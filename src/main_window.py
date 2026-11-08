"""
Main PySide6 Application Window (`DibellasWardrobeWindow`) for Dibella's Wardrobe.

Features:
- Dynamic on-demand (lazy) directory tree loading: instant startup even for massive MO2 directories
  with 500+ mods and 30,000+ .nif files.
- Non-blocking background search thread with debouncing.
- Smart NIF filters: hide `_0.nif` (low weight) and `gnd.nif`/`go.nif` (world models).
- Interactive 3D OpenGL Viewport (`OutfitViewportWidget`) powered by ModernGL + PyGLM with
  custom Skyrim GLSL shaders.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from PySide6.QtCore import QModelIndex, QRect, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig, LightingConfig, load_locales
from .nif_loader import LoadedNifData, NifLoader
from .texture_resolver import TextureResolver
from .viewport import OutfitViewportWidget

log = logging.getLogger("dibellas_wardrobe.ui")

ROLE_NIF_PATH = Qt.ItemDataRole.UserRole + 1
ROLE_FOLDER_PATH = Qt.ItemDataRole.UserRole + 2
ROLE_IS_LOADED = Qt.ItemDataRole.UserRole + 3
ROLE_ADDED_NIF_PATH = Qt.ItemDataRole.UserRole

TARGET_MOD_FOLDERS = {"meshes", "calientetools"}


def is_filtered_nif_filename(filename: str, hide_low_weight: bool = True) -> bool:
    """Return True if `filename` should be hidden by the active smart NIF filters."""
    lower = filename.lower()
    if not lower.endswith(".nif"):
        return True
    return hide_low_weight and lower[:-4].endswith("_0")


def is_filtered_folder_name(folder_name: str, hide_gnd_folders: bool = True) -> bool:
    """Return True if `folder_name` should be hidden by smart folder filters."""
    if not hide_gnd_folders:
        return False
    lower = folder_name.lower()
    return (
        lower in ("gnd", "_gnd", "ground")
        or lower.endswith("_gnd")
        or lower.startswith("gnd")
    )


def mod_has_mesh_or_caliente_folder(mod_path: Path | str) -> bool:
    """Return True if `mod_path` directly contains a 'meshes' or 'calienteTools' directory."""
    try:
        for entry in os.scandir(mod_path):
            if not entry.is_dir():
                continue
            lower = entry.name.lower()
            if lower in TARGET_MOD_FOLDERS:
                return True
            if lower == "data":
                try:
                    if any(
                        sub.is_dir() and sub.name.lower() in TARGET_MOD_FOLDERS
                        for sub in os.scandir(entry.path)
                    ):
                        return True
                except OSError:
                    pass
    except OSError:
        pass
    return False


class NifSearchThread(QThread):
    """
    Background worker that recursively searches `root_dir` for .nif files
    matching the search query without blocking the PySide6 UI thread.
    """

    matchFound = Signal(str, str, str)  # (mod_or_parent_name, file_name, full_path)

    def __init__(
        self,
        root_dir: Path,
        query: str,
        hide_0: bool,
        hide_gnd_folder: bool = True,
        meshes_and_caliente_only: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.root_dir = root_dir
        self.query = query.lower()
        self.hide_0 = hide_0
        self.hide_gnd_folder = hide_gnd_folder
        self.meshes_and_caliente_only = meshes_and_caliente_only
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            for root, dirs, files in os.walk(self.root_dir):
                if self._cancelled:
                    return

                if self.hide_gnd_folder:
                    dirs[:] = [d for d in dirs if not is_filtered_folder_name(d, True)]

                root_path = Path(root)
                if root_path == self.root_dir:
                    if self.query not in self.root_dir.name.lower():
                        dirs[:] = [d for d in dirs if self.query in d.lower()]
                    if self.meshes_and_caliente_only:
                        dirs[:] = [
                            d for d in dirs
                            if d.lower() in TARGET_MOD_FOLDERS
                            or mod_has_mesh_or_caliente_folder(self.root_dir / d)
                        ]
                elif root_path.parent == self.root_dir and self.meshes_and_caliente_only:
                    dirs[:] = [
                        d for d in dirs
                        if d.lower() in TARGET_MOD_FOLDERS or d.lower() == "data"
                    ]

                try:
                    rel_parts = root_path.relative_to(self.root_dir).parts
                    group_name = rel_parts[0] if rel_parts else root_path.name
                except Exception:
                    group_name = root_path.name

                if self.query not in group_name.lower():
                    continue

                for fname in files:
                    if self._cancelled:
                        return
                    if is_filtered_nif_filename(fname, self.hide_0):
                        continue

                    full_path = str((root_path / fname).resolve())
                    self.matchFound.emit(group_name, fname, full_path)
        except Exception as exc:
            log.warning("Search error: %s", exc)


class RefreshableGroupBox(QGroupBox):
    """QGroupBox featuring a compact refresh icon button in the top right corner."""

    def __init__(self, title: str, on_refresh, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        self.btn_refresh = QPushButton("↻", self)
        self.btn_refresh.setObjectName("refreshTreeBtn")
        self.btn_refresh.setToolTip("Refresh Tree")
        self.btn_refresh.setFixedSize(24, 20)
        self.btn_refresh.setStyleSheet(
            """
            QPushButton#refreshTreeBtn {
                background: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                color: #9da6be;
                font-size: 15px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton#refreshTreeBtn:hover {
                background-color: #2b3040;
                border: 1px solid #41485f;
                color: #f0d59c;
            }
            QPushButton#refreshTreeBtn:pressed {
                background-color: #1f222e;
                color: #dfb86c;
            }
            """
        )
        self.btn_refresh.clicked.connect(on_refresh)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.btn_refresh.move(max(0, self.width() - self.btn_refresh.width() - 8), 1)


class ElidedLabel(QLabel):
    """Label that elides text in the middle with '...' so it never overflows."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._full_text = text
        self.setToolTip(text)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(30)

    def setText(self, text: str) -> None:
        self._full_text = text
        self.setToolTip(text)
        self.update()

    def text(self) -> str:
        return self._full_text

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        metrics = painter.fontMetrics()
        elided = metrics.elidedText(self._full_text, Qt.TextElideMode.ElideMiddle, self.width())
        painter.setPen(QColor("#e4e7ef"))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided)


class AddedNifRowWidget(QWidget):
    """Row widget for a wardrobe NIF with visibility and remove/unlock controls."""

    def __init__(
        self,
        text: str,
        on_remove,
        on_select,
        on_toggle_visibility,
        on_double_click,
        visible: bool = True,
        remove_tooltip: str = "Remove this NIF",
        hide_tooltip: str = "Hide this NIF",
        show_tooltip: str = "Show this NIF",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._on_select = on_select
        self._on_double_click = on_double_click
        self._hide_tooltip = hide_tooltip
        self._show_tooltip = show_tooltip
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(4)

        self.label = ElidedLabel(text)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout.addWidget(self.label, stretch=1)

        self.btn_visibility = QPushButton()
        self.btn_visibility.setFixedSize(20, 20)
        self.btn_visibility.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: none;
                color: #aeb8cc;
                font-size: 13px;
                padding: 0px;
            }
            QPushButton:hover {
                color: #ffffff;
                background-color: #303543;
                border-radius: 3px;
            }
            """
        )
        self.btn_visibility.clicked.connect(on_toggle_visibility)
        layout.addWidget(self.btn_visibility, stretch=0)
        self.set_visible(visible)

        self.btn_remove = QPushButton("✕")
        self.btn_remove.setToolTip(remove_tooltip)
        self.btn_remove.setFixedSize(18, 18)
        self.btn_remove.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: none;
                color: #8f98b0;
                font-size: 11px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                color: #ff6b6b;
                background-color: #382424;
                border-radius: 3px;
            }
            QPushButton:pressed {
                color: #ff4040;
            }
            """
        )
        self.btn_remove.clicked.connect(on_remove)
        layout.addWidget(self.btn_remove, stretch=0)

    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_select()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_double_click()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def set_visible(self, visible: bool) -> None:
        self.btn_visibility.setText("◉" if visible else "○")
        self.btn_visibility.setToolTip(
            self._hide_tooltip if visible else self._show_tooltip
        )


class NeatTreeWidget(QTreeWidget):
    """Clean tree widget with compact indentation and subtle vertical guide lines."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setIndentation(14)
        self.setHeaderHidden(True)
        self.setAnimated(True)

    def drawBranches(self, painter: QPainter, rect: QRect, index: QModelIndex) -> None:
        if not index.isValid():
            super().drawBranches(painter, rect, index)
            return

        item = self.itemFromIndex(index)
        if item is None:
            super().drawBranches(painter, rect, index)
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        indent = self.indentation()
        depth = 0
        p = item.parent()
        while p is not None:
            depth += 1
            p = p.parent()

        y_top = rect.top()
        y_bottom = rect.bottom()
        y_mid = rect.center().y()

        # 1. Subtle vertical guide lines for ancestor levels
        guide_pen = QPen(QColor("#2d3342"))
        guide_pen.setWidth(1)
        painter.setPen(guide_pen)

        for d in range(depth):
            gx = rect.left() + d * indent + (indent // 2)
            painter.drawLine(gx, y_top, gx, y_bottom)

        # 2. Draw branch indicator for the current item
        is_nif = bool(item.data(0, ROLE_NIF_PATH))
        has_children = item.childCount() > 0 or not is_nif

        if has_children:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            arrow_pen = QPen(QColor("#9da6be"))
            arrow_pen.setWidth(1)
            painter.setPen(arrow_pen)

            cx = rect.left() + depth * indent + (indent // 2)
            cy = y_mid

            if item.isExpanded():
                painter.drawLine(cx - 3, cy - 2, cx, cy + 2)
                painter.drawLine(cx, cy + 2, cx + 3, cy - 2)
            else:
                painter.drawLine(cx - 2, cy - 3, cx + 2, cy)
                painter.drawLine(cx + 2, cy, cx - 2, cy + 3)
        else:
            # Subtle connector tick connecting parent guide line to the NIF item
            if depth > 0:
                conn_x = rect.left() + (depth - 1) * indent + (indent // 2)
                painter.drawLine(conn_x, y_mid, conn_x + (indent // 2) + 2, y_mid)

        painter.restore()


class DibellasWardrobeWindow(QMainWindow):
    """Main desktop window for Dibella's Wardrobe."""

    WINDOW_TITLE = "Dibella's Wardrobe"

    def __init__(
        self,
        initial_root_dir: Optional[Path | str] = None,
        config_path: Optional[Path | str] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(1520, 920)
        self.setWindowState(Qt.WindowState.WindowMaximized)

        self.config_path = config_path
        self._translations = load_locales()
        self.config = AppConfig.load(
            config_path,
            supported_languages=self._translations,
        )
        self.language = self.config.language
        self._i18n_bindings: List[tuple[QWidget, str, str]] = []

        cfg_folder = Path(self.config.default_folder) if self.config.default_folder else None

        if initial_root_dir:
            self.current_root_dir: Optional[Path] = Path(initial_root_dir).resolve()
        elif cfg_folder and cfg_folder.is_dir():
            self.current_root_dir = cfg_folder.resolve()
        else:
            self.current_root_dir = None

        extra_roots = [self.current_root_dir] if self.current_root_dir else None
        self.resolver = TextureResolver(extra_roots=extra_roots)
        self.loader = NifLoader(resolver=self.resolver)

        self._checked_nif_paths: Set[str] = set()
        self._added_nif_items: Dict[str, QListWidgetItem] = {}
        self._locked_nif_paths: Set[str] = {
            str(Path(path).resolve()) for path in self.config.locked_nifs if path
        }
        self._locked_nif_items: Dict[str, QListWidgetItem] = {}
        self._nif_visibility: Dict[str, bool] = {}
        self._highlighted_tree_items: List[QTreeWidgetItem] = []
        self._loaded_nif_cache: Dict[str, LoadedNifData] = {}
        self._updating_tree: bool = False
        self._search_thread: Optional[NifSearchThread] = None
        self._search_groups: Dict[str, QTreeWidgetItem] = {}

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(320)
        self._search_timer.timeout.connect(self._start_search_worker)

        self._apply_dark_theme()
        self._build_ui()
        self.populate_file_tree()
        self._autoload_locked_nifs()

    def _t(self, key: str, **values: object) -> str:
        translations = self._translations.get(self.language, {})
        fallback = next(iter(self._translations.values()))
        template = translations.get(key, fallback.get(key, key))
        return template.format(**values)

    def _bind_text(self, widget: QWidget, key: str, setter: str = "setText") -> None:
        self._i18n_bindings.append((widget, key, setter))
        getattr(widget, setter)(self._t(key))

    def _on_language_changed(self, _index: int) -> None:
        language = self.language_combo.currentData()
        if language not in self._translations or language == self.language:
            return
        self.language = language
        self.config.language = language
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        for widget, key, setter in self._i18n_bindings:
            getattr(widget, setter)(self._t(key))
        self._retranslate_tree_helpers()
        self._refresh_added_nifs()
        self._update_added_nif_count()

    def _retranslate_tree_helpers(self) -> None:
        helper_keys = ("loading", "no_nif_files")
        known_texts = {
            translations[key]
            for translations in self._translations.values()
            for key in helper_keys
        }

        def visit(item: QTreeWidgetItem) -> None:
            if item.text(0) in known_texts:
                key = "loading" if item.text(0) in {
                    translations["loading"]
                    for translations in self._translations.values()
                } else "no_nif_files"
                item.setText(0, self._t(key))
            for index in range(item.childCount()):
                visit(item.child(index))

        for index in range(self.file_tree.topLevelItemCount()):
            visit(self.file_tree.topLevelItem(index))

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #1b1d24;
                color: #e4e7ef;
                font-family: "Segoe UI", sans-serif;
                font-size: 10pt;
            }
            QGroupBox {
                border: 1px solid #303442;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: 600;
                color: #f0d59c;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #2b3040;
                border: 1px solid #41485f;
                border-radius: 5px;
                padding: 6px 12px;
                color: #eaeef7;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #384057;
                border-color: #d4af63;
            }
            QPushButton:pressed {
                background-color: #222634;
            }
            QPushButton:checked {
                background-color: #5b3b24;
                border-color: #e6bd68;
                color: #fff6e0;
            }
            QLineEdit, QTreeWidget {
                background-color: #14161c;
                border: 1px solid #2e3240;
                border-radius: 5px;
                padding: 4px;
                color: #e4e7ef;
            }
            QTreeWidget::item {
                padding: 3px 2px;
            }
            QTreeWidget::item:selected {
                background-color: #2f3852;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #20232d;
                color: #c9cfde;
                border: none;
                padding: 5px;
                font-weight: 600;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #2b2f3c;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #dfb86c;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            """
        )

    def _build_ui(self) -> None:
        self.menuBar().setVisible(False)

        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(0)

        # Global keyboard shortcuts
        self.shortcut_open = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_open.activated.connect(self._on_browse_root_folder)
        self.shortcut_filter = QShortcut(QKeySequence("Ctrl+F"), self)
        self.shortcut_filter.activated.connect(lambda: self._toggle_mod_dir_filter())

        # Main splitter: 4 columns
        # Column 1: Mod Directory / Filter + Outfit Meshes Tree
        # Column 2: Currently Added NIFs Box (Adjacent Column)
        # Column 3: 3D OpenGL Viewport
        # Column 4: Lighting & Skyrim Shader Controls Sidebar
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        # ------------------------------------------------------------------
        # COLUMN 1: Mod Directory / Filter + Outfit Meshes Tree
        # ------------------------------------------------------------------
        mesh_panel = QWidget()
        mesh_layout = QVBoxLayout(mesh_panel)
        mesh_layout.setContentsMargins(2, 2, 2, 2)
        mesh_layout.setSpacing(6)

        dir_group = QGroupBox()
        self._bind_text(dir_group, "mod_directory", "setTitle")
        dir_layout = QVBoxLayout(dir_group)

        path_row = QHBoxLayout()
        self.root_path_edit = QLineEdit(
            str(self.current_root_dir) if self.current_root_dir else ""
        )
        self.root_path_edit.setReadOnly(True)
        browse_btn = QPushButton()
        self._bind_text(browse_btn, "browse_folder")
        browse_btn.clicked.connect(self._on_browse_root_folder)
        path_row.addWidget(self.root_path_edit, stretch=1)
        path_row.addWidget(browse_btn)
        dir_layout.addLayout(path_row)

        cfg_chk = self.config.checkboxes
        self.chk_meshes_and_caliente_only = self._add_checkbox(
            dir_layout, self._t("show_meshes_caliente"),
            cfg_chk.meshes_and_caliente_only, self._on_filter_toggled
        )
        self._bind_text(self.chk_meshes_and_caliente_only, "show_meshes_caliente")
        self.chk_meshes_and_caliente_only.setToolTip(self._t("show_meshes_caliente_tip"))
        self._i18n_bindings.append(
            (self.chk_meshes_and_caliente_only, "show_meshes_caliente_tip", "setToolTip")
        )
        self.chk_hide_low_weight = self._add_checkbox(
            dir_layout, self._t("hide_low_weight"),
            cfg_chk.hide_low_weight, self._on_filter_toggled
        )
        self._bind_text(self.chk_hide_low_weight, "hide_low_weight")
        self.chk_hide_gnd_folder = self._add_checkbox(
            dir_layout, self._t("hide_gnd"),
            cfg_chk.hide_gnd_folder, self._on_filter_toggled
        )
        self._bind_text(self.chk_hide_gnd_folder, "hide_gnd")

        self.dir_group = dir_group
        self.dir_group.setVisible(cfg_chk.show_mod_dir_filter)
        mesh_layout.addWidget(dir_group)

        tree_group = RefreshableGroupBox(
            self._t("outfit_meshes"),
            self.populate_file_tree,
        )
        self._bind_text(tree_group, "outfit_meshes", "setTitle")
        self.btn_refresh_tree = tree_group.btn_refresh
        self._bind_text(self.btn_refresh_tree, "refresh_tree", "setToolTip")
        tree_layout = QVBoxLayout(tree_group)

        self.search_edit = QLineEdit()
        self._bind_text(self.search_edit, "search_mod", "setPlaceholderText")
        self.search_edit.textChanged.connect(self._on_search_text_changed)
        tree_layout.addWidget(self.search_edit)

        self.file_tree = NeatTreeWidget()
        self.file_tree.itemExpanded.connect(self._on_tree_item_expanded)
        self.file_tree.itemClicked.connect(self._on_tree_item_clicked)
        tree_layout.addWidget(self.file_tree, stretch=1)

        mesh_layout.addWidget(tree_group, stretch=1)
        self.mesh_panel = mesh_panel
        splitter.addWidget(mesh_panel)

        # ------------------------------------------------------------------
        # COLUMN 2: Currently Added NIFs Box (Adjacent Column)
        # ------------------------------------------------------------------
        added_panel = QWidget()
        added_layout = QVBoxLayout(added_panel)
        added_layout.setContentsMargins(2, 2, 2, 2)
        added_layout.setSpacing(6)

        locked_group = QGroupBox(self._t("locked_wardrobe", count=0))
        locked_group_layout = QVBoxLayout(locked_group)
        self.locked_nifs_list = QListWidget()
        self.locked_nifs_list.setMinimumHeight(72)
        self.locked_nifs_list.setMaximumHeight(210)
        self.locked_nifs_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.locked_nifs_list.itemClicked.connect(self._on_added_nif_clicked)
        locked_group_layout.addWidget(self.locked_nifs_list, stretch=1)
        added_layout.addWidget(locked_group)

        added_group = QGroupBox(self._t("wardrobe", count=0))
        added_group_layout = QVBoxLayout(added_group)
        self.added_nifs_list = QListWidget()
        self.added_nifs_list.setMinimumHeight(72)
        self.added_nifs_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.added_nifs_list.itemClicked.connect(self._on_added_nif_clicked)
        added_group_layout.addWidget(self.added_nifs_list, stretch=1)
        self.locked_nifs_group = locked_group

        self.btn_clear_all_added = QPushButton()
        self._bind_text(self.btn_clear_all_added, "clear_unlocked")
        self.btn_clear_all_added.clicked.connect(self.uncheck_all_nifs)
        added_group_layout.addWidget(self.btn_clear_all_added)

        self.added_nifs_group = added_group
        added_layout.addWidget(added_group, stretch=1)

        self.added_panel = added_panel
        splitter.addWidget(added_panel)

        # ------------------------------------------------------------------
        # COLUMN 3: 3D OpenGL Viewport
        # ------------------------------------------------------------------
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 4, 0, 4)
        center_layout.setSpacing(4)

        self.viewport = OutfitViewportWidget(self)

        cfg_light = self.config.lighting
        cfg_chk = self.config.checkboxes

        s = self.viewport.settings
        s.light_azimuth_deg = cfg_light.light_azimuth_deg
        s.light_elevation_deg = cfg_light.light_elevation_deg
        s.light_intensity = cfg_light.light_intensity
        s.ambient_intensity = cfg_light.ambient_intensity
        s.specular_multiplier = cfg_light.specular_multiplier
        s.env_reflection_multiplier = cfg_light.env_reflection_multiplier

        s.wireframe_overlay = cfg_chk.wireframe_overlay
        s.show_ground_grid = cfg_chk.show_ground_grid

        center_layout.addWidget(self.viewport, stretch=1)

        cam_hint_row = QHBoxLayout()
        cam_hint = QLabel()
        self._bind_text(cam_hint, "camera_controls")
        cam_hint.setStyleSheet("color: #8f98b0; font-size: 12px;")
        btn_reset_cam = QPushButton()
        self._bind_text(btn_reset_cam, "reset_camera")
        btn_reset_cam.setShortcut("R")
        btn_reset_cam.clicked.connect(self.viewport.reset_camera)

        self.btn_settings = QPushButton()
        self._bind_text(self.btn_settings, "settings")
        self.btn_settings.setShortcut("S")
        self.btn_settings.clicked.connect(lambda: self._toggle_settings_sidebar())

        cam_hint_row.addWidget(cam_hint)
        cam_hint_row.addStretch(1)
        cam_hint_row.addWidget(btn_reset_cam)
        cam_hint_row.addWidget(self.btn_settings)
        center_layout.addLayout(cam_hint_row)

        splitter.addWidget(center_panel)

        # ------------------------------------------------------------------
        # COLUMN 4: Lighting & Skyrim Shader Controls Sidebar
        # ------------------------------------------------------------------
        right_panel = QWidget()
        right_panel.setMinimumWidth(400)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        sidebar_header = QHBoxLayout()
        sidebar_title = QLabel()
        self._bind_text(sidebar_title, "settings_title")
        sidebar_title.setStyleSheet("font-weight: 600; color: #f0d59c; font-size: 13px;")
        btn_close_sidebar = QPushButton("✕")
        self._bind_text(btn_close_sidebar, "close_settings", "setToolTip")
        btn_close_sidebar.setFixedSize(22, 22)
        btn_close_sidebar.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: none;
                color: #8f98b0;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                color: #ff6b6b;
                background-color: #382424;
                border-radius: 3px;
            }
            """
        )
        btn_close_sidebar.clicked.connect(lambda: self._toggle_settings_sidebar(False))
        language_label = QLabel()
        self._bind_text(language_label, "language")
        self.language_combo = QComboBox()
        for language_code, translations in self._translations.items():
            self.language_combo.addItem(translations["language_name"], language_code)
        self.language_combo.setCurrentIndex(
            self.language_combo.findData(self.language)
        )
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        sidebar_header.addWidget(sidebar_title)
        sidebar_header.addWidget(language_label)
        sidebar_header.addWidget(self.language_combo)
        sidebar_header.addStretch(1)
        sidebar_header.addWidget(btn_close_sidebar)
        right_layout.addLayout(sidebar_header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)

        # Lighting Controls Group
        light_group = RefreshableGroupBox(
            self._t("lighting_hemisphere"),
            self._reset_lighting_controls,
        )
        self._bind_text(light_group, "lighting_hemisphere", "setTitle")
        light_form = QFormLayout(light_group)

        self.btn_reset_lighting = light_group.btn_refresh
        self._bind_text(self.btn_reset_lighting, "reset_lighting", "setToolTip")

        self.slider_light_azimuth, self.lbl_azimuth_val = self._add_slider_row(
            light_form, self._t("light_rotation"), -180, 180, int(round(cfg_light.light_azimuth_deg)),
            f"{int(round(cfg_light.light_azimuth_deg))}°", self._on_lighting_changed,
            "light_rotation",
        )
        self.slider_light_elevation, self.lbl_elevation_val = self._add_slider_row(
            light_form, self._t("light_elevation"), -60, 85, int(round(cfg_light.light_elevation_deg)),
            f"{int(round(cfg_light.light_elevation_deg))}°", self._on_lighting_changed,
            "light_elevation",
        )
        self.slider_key_intensity, self.lbl_key_val = self._add_slider_row(
            light_form, self._t("key_light_power"), 0, 200, int(round(cfg_light.light_intensity * 100)),
            f"{cfg_light.light_intensity:.2f}", self._on_lighting_changed,
            "key_light_power",
        )
        self.slider_ambient, self.lbl_ambient_val = self._add_slider_row(
            light_form, self._t("ambient_light"), 0, 100, int(round(cfg_light.ambient_intensity * 100)),
            f"{cfg_light.ambient_intensity:.2f}", self._on_lighting_changed,
            "ambient_light",
        )
        self.slider_specular, self.lbl_spec_val = self._add_slider_row(
            light_form, self._t("specular_scale"), 0, 300, int(round(cfg_light.specular_multiplier * 100)),
            f"{cfg_light.specular_multiplier:.2f}x", self._on_lighting_changed,
            "specular_scale",
        )
        self.slider_env, self.lbl_env_val = self._add_slider_row(
            light_form, self._t("env_reflection"), 0, 300, int(round(cfg_light.env_reflection_multiplier * 100)),
            f"{cfg_light.env_reflection_multiplier:.2f}x", self._on_lighting_changed,
            "env_reflection",
        )

        scroll_layout.addWidget(light_group)

        # Skyrim Shader Features Group
        shader_group = QGroupBox()
        self._bind_text(shader_group, "shader_options", "setTitle")
        shader_layout = QVBoxLayout(shader_group)

        self.chk_wireframe = self._add_checkbox(
            shader_layout, self._t("wireframe_overlay"),
            cfg_chk.wireframe_overlay, self._on_shader_toggles_changed
        )
        self._bind_text(self.chk_wireframe, "wireframe_overlay")
        self.chk_grid = self._add_checkbox(
            shader_layout, self._t("ground_grid"),
            cfg_chk.show_ground_grid, self._on_shader_toggles_changed
        )
        self._bind_text(self.chk_grid, "ground_grid")

        scroll_layout.addWidget(shader_group)
        scroll_layout.addStretch(1)

        scroll.setWidget(scroll_content)
        right_layout.addWidget(scroll, stretch=1)

        self.right_panel = right_panel
        splitter.addWidget(right_panel)
        self.right_panel.setVisible(False)

        mesh_panel.setMinimumWidth(260)
        added_panel.setMinimumWidth(180)
        center_panel.setMinimumWidth(300)

        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)
        splitter.setCollapsible(3, True)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 0)
        splitter.setStretchFactor(2, 1)
        splitter.setStretchFactor(3, 0)

        self.splitter = splitter
        self.splitter.setSizes([350, 260, 910, 0])

    def save_config(self, path: Optional[Path | str] = None) -> None:
        """Save current user settings back to JSON config."""
        target_path = path if path else self.config_path
        self.config.default_folder = (
            str(self.current_root_dir) if self.current_root_dir else ""
        )
        self.config.language = self.language

        # Lighting
        l = self.config.lighting
        l.light_azimuth_deg = float(self.slider_light_azimuth.value())
        l.light_elevation_deg = float(self.slider_light_elevation.value())
        l.light_intensity = self.slider_key_intensity.value() / 100.0
        l.ambient_intensity = self.slider_ambient.value() / 100.0
        l.specular_multiplier = self.slider_specular.value() / 100.0
        l.env_reflection_multiplier = self.slider_env.value() / 100.0

        # Checkboxes
        c = self.config.checkboxes
        c.meshes_and_caliente_only = self.chk_meshes_and_caliente_only.isChecked()
        c.hide_low_weight = self.chk_hide_low_weight.isChecked()
        c.hide_gnd_folder = self.chk_hide_gnd_folder.isChecked()
        c.show_mod_dir_filter = self.dir_group.isVisible()

        c.wireframe_overlay = self.chk_wireframe.isChecked()
        c.show_ground_grid = self.chk_grid.isChecked()
        self.config.locked_nifs = sorted(self._locked_nif_paths)

        self.config.save(target_path)

    def closeEvent(self, event) -> None:
        try:
            self.save_config()
        except Exception as exc:
            log.warning("Could not save config on close: %s", exc)
        super().closeEvent(event)

    def _toggle_mod_dir_filter(self, visible: Optional[bool] = None) -> None:
        """Toggle visibility of the Mod Directory & NIF Filter group to expand the outfit meshes box."""
        if visible is None:
            visible = not self.dir_group.isVisible()
        self.dir_group.setVisible(visible)
        if hasattr(self, "act_toggle_filter"):
            self.act_toggle_filter.setChecked(visible)

    def _toggle_settings_sidebar(self, visible: Optional[bool] = None) -> None:
        if visible is None:
            visible = not self.right_panel.isVisible()

        curr_sizes = self.splitter.sizes()
        total_w = sum(curr_sizes)
        mesh_w = curr_sizes[0]
        added_w = curr_sizes[1]

        if visible:
            if not self.right_panel.isVisible():
                self.right_panel.setVisible(True)
                right_target_w = min(420, max(400, total_w - mesh_w - added_w - 300))
                center_w = max(300, total_w - mesh_w - added_w - right_target_w)
                self.splitter.setSizes([mesh_w, added_w, center_w, right_target_w])
        else:
            if self.right_panel.isVisible():
                self.right_panel.setVisible(False)
                center_w = total_w - mesh_w - added_w
                self.splitter.setSizes([mesh_w, added_w, center_w, 0])

    def _add_slider_row(
        self,
        form: QFormLayout,
        label_text: str,
        min_val: int,
        max_val: int,
        init_val: int,
        val_text: str,
        callback,
        label_key: str,
    ) -> tuple[QSlider, QLabel]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(init_val)
        slider.valueChanged.connect(callback)

        value_label = QLabel(val_text)
        value_label.setMinimumWidth(46)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(value_label)

        label = QLabel(label_text)
        self._bind_text(label, label_key)
        form.addRow(label, row)
        return slider, value_label

    @staticmethod
    def _add_checkbox(layout: QVBoxLayout, text: str, checked: bool, callback) -> QCheckBox:
        chk = QCheckBox(text)
        chk.setChecked(checked)
        chk.toggled.connect(callback)
        layout.addWidget(chk)
        return chk

    def _create_folder_tree_item(self, parent: QTreeWidget | QTreeWidgetItem, name: str, path: str, color: str) -> QTreeWidgetItem:
        folder_item = QTreeWidgetItem(parent, [name])
        folder_item.setData(0, ROLE_FOLDER_PATH, path)
        folder_item.setData(0, ROLE_IS_LOADED, False)
        folder_item.setForeground(0, QColor(color))
        QTreeWidgetItem(folder_item, [self._t("loading"), ""])
        return folder_item

    @staticmethod
    def _create_nif_tree_item(parent: QTreeWidget | QTreeWidgetItem, name: str, path: str) -> QTreeWidgetItem:
        nif_item = QTreeWidgetItem(parent, [name])
        nif_item.setData(0, ROLE_NIF_PATH, path)
        nif_item.setToolTip(0, path)
        nif_item.setForeground(0, QColor("#dce3f0"))
        nif_item.setFlags(nif_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        return nif_item

    # ------------------------------------------------------------------
    # Dynamic On-Demand Tree Population & Background Search
    # ------------------------------------------------------------------

    def set_root_directory(self, folder: Path | str) -> None:
        target = Path(folder).resolve()
        if not target.is_dir():
            return
        self.current_root_dir = target
        self.root_path_edit.setText(str(target))
        self.resolver.add_search_root(target)
        self._refresh_added_nifs()
        self.populate_file_tree()
        try:
            self.save_config()
        except Exception:
            pass

    def _on_browse_root_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            self._t("select_mod_directory"),
            str(self.current_root_dir) if self.current_root_dir else "",
        )
        if chosen:
            self.set_root_directory(Path(chosen))

    def _on_filter_toggled(self) -> None:
        if self.search_edit.text().strip():
            self._start_search_worker()
        else:
            self.populate_file_tree()

    def _on_search_text_changed(self, text: str) -> None:
        query = text.strip()
        if not query:
            if self._search_thread and self._search_thread.isRunning():
                self._search_thread.cancel()
            self._search_timer.stop()
            self.populate_file_tree()
        else:
            self._search_timer.start(320)

    def _start_search_worker(self) -> None:
        query = self.search_edit.text().strip()
        if not query or self.current_root_dir is None:
            self.populate_file_tree()
            return

        if self._search_thread and self._search_thread.isRunning():
            self._search_thread.cancel()
            self._search_thread.wait(300)

        self._updating_tree = True
        self.file_tree.clear()
        self._updating_tree = False

        self._search_groups = {}
        self._search_thread = NifSearchThread(
            root_dir=self.current_root_dir,
            query=query,
            hide_0=self.chk_hide_low_weight.isChecked(),
            hide_gnd_folder=self.chk_hide_gnd_folder.isChecked(),
            meshes_and_caliente_only=self.chk_meshes_and_caliente_only.isChecked(),
            parent=self,
        )
        self._search_thread.matchFound.connect(self._on_search_match_found)
        self._search_thread.start()

    def _on_search_match_found(self, group_name: str, file_name: str, full_path: str) -> None:
        self._updating_tree = True
        if group_name not in self._search_groups:
            grp_item = QTreeWidgetItem(self.file_tree, [group_name])
            grp_item.setForeground(0, QColor("#f0d59c"))
            self._search_groups[group_name] = grp_item

        self._create_nif_tree_item(self._search_groups[group_name], file_name, full_path)
        self._updating_tree = False

    def populate_file_tree(self) -> None:
        r"""
        Dynamically populate top-level directory entries on demand.
        Does NOT recursively scan or expand, ensuring instant loading for
        large mod directories.
        """
        if self._search_thread and self._search_thread.isRunning():
            self._search_thread.cancel()
            self._search_thread.wait(300)

        self._updating_tree = True
        self.file_tree.clear()

        if self.current_root_dir is None:
            self._updating_tree = False
            return

        hide_0 = self.chk_hide_low_weight.isChecked()
        hide_gnd_folder = self.chk_hide_gnd_folder.isChecked()
        meshes_caliente_only = self.chk_meshes_and_caliente_only.isChecked()

        try:
            entries = sorted(
                list(os.scandir(self.current_root_dir)),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            )
        except Exception as exc:
            log.error("Failed to list %s: %s", self.current_root_dir, exc)
            self._updating_tree = False
            return

        root_is_mod = mod_has_mesh_or_caliente_folder(self.current_root_dir)

        for entry in entries:
            try:
                if entry.is_dir():
                    if is_filtered_folder_name(entry.name, hide_gnd_folder):
                        continue
                    if meshes_caliente_only:
                        if root_is_mod:
                            if entry.name.lower() not in TARGET_MOD_FOLDERS and entry.name.lower() != "data":
                                continue
                        elif not mod_has_mesh_or_caliente_folder(entry.path):
                            continue
                    self._create_folder_tree_item(self.file_tree, entry.name, entry.path, "#e0d2b4")
                elif entry.is_file() and not is_filtered_nif_filename(entry.name, hide_0):
                    self._create_nif_tree_item(self.file_tree, entry.name, entry.path)
            except OSError:
                continue

        self._updating_tree = False

    def _on_tree_item_expanded(self, item: QTreeWidgetItem) -> None:
        """
        Dynamically populate immediate children when a folder node is expanded.
        """
        if item.data(0, ROLE_IS_LOADED):
            return

        folder_path = item.data(0, ROLE_FOLDER_PATH)
        if not folder_path or not os.path.isdir(folder_path):
            return

        self._updating_tree = True
        item.takeChildren()

        hide_0 = self.chk_hide_low_weight.isChecked()
        hide_gnd_folder = self.chk_hide_gnd_folder.isChecked()
        meshes_caliente_only = self.chk_meshes_and_caliente_only.isChecked()

        is_mod_level = (
            self.current_root_dir is not None
            and Path(folder_path).parent == self.current_root_dir
            and not mod_has_mesh_or_caliente_folder(self.current_root_dir)
        )

        try:
            entries = sorted(
                list(os.scandir(folder_path)),
                key=lambda e: (not e.is_dir(), e.name.lower()),
            )
            added_any = False
            for entry in entries:
                if entry.is_dir():
                    if is_filtered_folder_name(entry.name, hide_gnd_folder):
                        continue
                    if meshes_caliente_only and is_mod_level:
                        if entry.name.lower() not in TARGET_MOD_FOLDERS and entry.name.lower() != "data":
                            continue
                    self._create_folder_tree_item(item, entry.name, entry.path, "#c5cee2")
                    added_any = True
                elif entry.is_file() and not is_filtered_nif_filename(entry.name, hide_0):
                    self._create_nif_tree_item(item, entry.name, entry.path)
                    added_any = True

            if not added_any:
                empty_item = QTreeWidgetItem(item, [self._t("no_nif_files")])
                empty_item.setForeground(0, QColor("#6c7285"))
                empty_item.setFlags(Qt.ItemFlag.NoItemFlags)

            item.setData(0, ROLE_IS_LOADED, True)
        except Exception as exc:
            log.warning("Error expanding %s: %s", folder_path, exc)
        finally:
            self._updating_tree = False

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._updating_tree:
            return

        nif_key = item.data(0, ROLE_NIF_PATH)
        if not nif_key:
            return

        if nif_key in self._locked_nif_paths:
            return

        if nif_key not in self._checked_nif_paths:
            self._load_and_attach_outfit_nif(nif_key)
        else:
            self._detach_outfit_nif(nif_key)

    def _load_and_attach_outfit_nif(self, nif_key: str, show_error: bool = True) -> bool:
        try:
            nif_data = self._loaded_nif_cache.get(nif_key)
            if nif_data is None:
                nif_data = self.loader.load_nif(nif_key)
                self._loaded_nif_cache[nif_key] = nif_data

            self.viewport.add_nif(nif_key, nif_data)
            self._checked_nif_paths.add(nif_key)
            self._nif_visibility.setdefault(nif_key, True)
            self.viewport.set_nif_visible(nif_key, self._nif_visibility[nif_key])
            self._add_nif_to_wardrobe_list(nif_key)
            return True
        except Exception as exc:
            log.error("Failed loading NIF %s: %s", nif_key, exc)
            if show_error:
                QMessageBox.warning(
                    self,
                    self._t("nif_load_error"),
                    self._t(
                        "nif_load_error_message",
                        filename=Path(nif_key).name,
                        error=exc,
                    ),
                )
            return False

    def _detach_outfit_nif(self, nif_key: str) -> None:
        was_locked = nif_key in self._locked_nif_paths
        self._checked_nif_paths.discard(nif_key)
        self._loaded_nif_cache.pop(nif_key, None)
        self._nif_visibility.pop(nif_key, None)
        self.viewport.remove_nif(nif_key)
        list_item = self._added_nif_items.pop(nif_key, None)
        if list_item is not None:
            self.added_nifs_list.takeItem(self.added_nifs_list.row(list_item))
        locked_item = self._locked_nif_items.pop(nif_key, None)
        if locked_item is not None:
            self.locked_nifs_list.takeItem(self.locked_nifs_list.row(locked_item))
        self._locked_nif_paths.discard(nif_key)
        if was_locked:
            self.save_config()
        self._update_added_nif_count()

    def uncheck_all_nifs(self) -> None:
        """Remove unlocked outfit meshes while preserving the locked wardrobe."""
        for nif_key in tuple(self._checked_nif_paths - self._locked_nif_paths):
            self._detach_outfit_nif(nif_key)
        self._refresh_added_nifs()

    def _mod_name_for_nif(self, nif_key: str) -> str:
        nif_path = Path(nif_key)
        if self.current_root_dir is None:
            return nif_path.parent.name
        try:
            relative_path = nif_path.resolve().relative_to(self.current_root_dir)
        except ValueError:
            return nif_path.parent.name
        if any((self.current_root_dir / name).is_dir() for name in ("meshes", "textures")):
            return self.current_root_dir.name
        return relative_path.parts[0] if len(relative_path.parts) > 1 else self.current_root_dir.name

    def _add_nif_to_wardrobe_list(self, nif_key: str) -> None:
        locked = nif_key in self._locked_nif_paths
        target_list = self.locked_nifs_list if locked else self.added_nifs_list
        target_items = self._locked_nif_items if locked else self._added_nif_items
        if nif_key in target_items:
            return
        item = QListWidgetItem()
        item.setData(ROLE_ADDED_NIF_PATH, nif_key)
        item.setToolTip(nif_key)

        display_text = f"{Path(nif_key).name} ({self._mod_name_for_nif(nif_key)})"
        row_widget = AddedNifRowWidget(
            text=display_text,
            on_remove=lambda _checked=False, k=nif_key: (
                self._unlock_nif(k) if locked else self._detach_outfit_nif(k)
            ),
            on_select=lambda k=nif_key: self._select_nif_in_tree(k),
            on_toggle_visibility=lambda _checked=False, k=nif_key: self._toggle_nif_visibility(k),
            on_double_click=lambda k=nif_key: (
                self._unlock_nif(k) if locked else self._lock_nif(k)
            ),
            visible=self._nif_visibility.get(nif_key, True),
            remove_tooltip=self._t("unlock_nif") if locked else self._t("remove_nif"),
            hide_tooltip=self._t("hide_nif"),
            show_tooltip=self._t("show_nif"),
        )
        row_widget.setToolTip(
            self._t("double_click_unlock")
            if locked
            else self._t("double_click_lock")
        )
        vp_width = max(100, target_list.viewport().width())
        item.setSizeHint(QSize(vp_width, 26))
        target_list.addItem(item)
        target_list.setItemWidget(item, row_widget)
        target_items[nif_key] = item
        self._update_added_nif_count()

    def _refresh_added_nifs(self) -> None:
        self.added_nifs_list.clear()
        self.locked_nifs_list.clear()
        self._added_nif_items.clear()
        self._locked_nif_items.clear()
        for nif_key in sorted(self._checked_nif_paths):
            self._add_nif_to_wardrobe_list(nif_key)
        self._update_added_nif_count()

    def _update_added_nif_count(self) -> None:
        self.added_nifs_group.setTitle(
            self._t("wardrobe", count=len(self._added_nif_items))
        )
        self.locked_nifs_group.setTitle(
            self._t("locked_wardrobe", count=len(self._locked_nif_items))
        )

    def _on_added_nif_clicked(self, item: QListWidgetItem) -> None:
        nif_key = item.data(ROLE_ADDED_NIF_PATH)
        if nif_key:
            self._select_nif_in_tree(nif_key)

    def _autoload_locked_nifs(self) -> None:
        """Load NIFs saved in the locked wardrobe without showing startup dialogs."""
        for nif_key in sorted(self._locked_nif_paths):
            if not Path(nif_key).is_file():
                log.warning("Locked NIF is not available: %s", nif_key)
                continue
            self._load_and_attach_outfit_nif(nif_key, show_error=False)

    def _lock_nif(self, nif_key: str) -> None:
        if nif_key not in self._checked_nif_paths or nif_key in self._locked_nif_paths:
            return
        self._locked_nif_paths.add(nif_key)
        self.save_config()
        self._refresh_added_nifs()

    def _unlock_nif(self, nif_key: str) -> None:
        if nif_key not in self._locked_nif_paths:
            return
        self._locked_nif_paths.discard(nif_key)
        self.save_config()
        self._refresh_added_nifs()

    def _toggle_nif_visibility(self, nif_key: str) -> None:
        visible = not self._nif_visibility.get(nif_key, True)
        self._nif_visibility[nif_key] = visible
        self.viewport.set_nif_visible(nif_key, visible)

        item = self._locked_nif_items.get(nif_key) or self._added_nif_items.get(nif_key)
        if item is not None:
            row_widget = (
                self.locked_nifs_list.itemWidget(item)
                if nif_key in self._locked_nif_items
                else self.added_nifs_list.itemWidget(item)
            )
            if isinstance(row_widget, AddedNifRowWidget):
                row_widget.set_visible(visible)

    def _tree_siblings(self, parent_item: Optional[QTreeWidgetItem]) -> List[QTreeWidgetItem]:
        if parent_item is not None:
            return [parent_item.child(i) for i in range(parent_item.childCount())]
        return [
            self.file_tree.topLevelItem(i)
            for i in range(self.file_tree.topLevelItemCount())
        ]

    def _select_nif_in_tree(self, nif_key: str) -> None:
        if self.current_root_dir is None:
            return
        try:
            nif_path = Path(nif_key).resolve()
            relative_path = nif_path.relative_to(self.current_root_dir)
        except ValueError:
            return

        if self.search_edit.text():
            self.search_edit.clear()

        parent_item: Optional[QTreeWidgetItem] = None
        current_path = self.current_root_dir
        for folder_name in relative_path.parts[:-1]:
            current_path /= folder_name
            parent_item = next(
                (
                    candidate
                    for candidate in self._tree_siblings(parent_item)
                    if candidate.data(0, ROLE_FOLDER_PATH)
                    and Path(candidate.data(0, ROLE_FOLDER_PATH)).resolve()
                    == current_path.resolve()
                ),
                None,
            )
            if parent_item is None:
                return
            self._on_tree_item_expanded(parent_item)
            parent_item.setExpanded(True)

        target_item = next(
            (
                candidate
                for candidate in self._tree_siblings(parent_item)
                if candidate.data(0, ROLE_NIF_PATH)
                and Path(candidate.data(0, ROLE_NIF_PATH)).resolve() == nif_path
            ),
            None,
        )
        if target_item is None:
            return

        for previous in self._highlighted_tree_items:
            previous.setBackground(0, QBrush())
        self._highlighted_tree_items = []
        ancestor: Optional[QTreeWidgetItem] = target_item
        while ancestor is not None:
            ancestor.setBackground(0, QColor("#5b3b24"))
            self._highlighted_tree_items.append(ancestor)
            ancestor = ancestor.parent()
        self.file_tree.setCurrentItem(target_item)
        self.file_tree.scrollToItem(target_item)

    # ------------------------------------------------------------------
    # Lighting & Shader Updates
    # ------------------------------------------------------------------

    def _on_lighting_changed(self) -> None:
        az = float(self.slider_light_azimuth.value())
        el = float(self.slider_light_elevation.value())
        key_i = self.slider_key_intensity.value() / 100.0
        amb_i = self.slider_ambient.value() / 100.0
        spec_m = self.slider_specular.value() / 100.0
        env_m = self.slider_env.value() / 100.0

        self.lbl_azimuth_val.setText(f"{int(az)}°")
        self.lbl_elevation_val.setText(f"{int(el)}°")
        self.lbl_key_val.setText(f"{key_i:.2f}")
        self.lbl_ambient_val.setText(f"{amb_i:.2f}")
        self.lbl_spec_val.setText(f"{spec_m:.2f}x")
        self.lbl_env_val.setText(f"{env_m:.2f}x")

        s = self.viewport.settings
        s.light_azimuth_deg = az
        s.light_elevation_deg = el
        s.light_intensity = key_i
        s.ambient_intensity = amb_i
        s.specular_multiplier = spec_m
        s.env_reflection_multiplier = env_m
        self.viewport.update()

    def _reset_lighting_controls(self) -> None:
        defaults = LightingConfig()
        self.slider_light_azimuth.setValue(int(round(defaults.light_azimuth_deg)))
        self.slider_light_elevation.setValue(int(round(defaults.light_elevation_deg)))
        self.slider_key_intensity.setValue(int(round(defaults.light_intensity * 100)))
        self.slider_ambient.setValue(int(round(defaults.ambient_intensity * 100)))
        self.slider_specular.setValue(int(round(defaults.specular_multiplier * 100)))
        self.slider_env.setValue(int(round(defaults.env_reflection_multiplier * 100)))
        self._on_lighting_changed()

    def _on_shader_toggles_changed(self) -> None:
        s = self.viewport.settings
        s.wireframe_overlay = self.chk_wireframe.isChecked()
        s.show_ground_grid = self.chk_grid.isChecked()
        self.viewport.update()
