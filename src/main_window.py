"""
Main PySide6 Application Window (`DibellasWardrobeWindow`) for Dibella's Wardrobe.

Features:
- Dynamic on-demand (lazy) directory tree loading: instant startup even for massive MO2 directories.
- Non-blocking background search thread with debouncing.
- Smart filters: hide low weight, hide GND folders, hide folders without NIF/HKX.
- Interactive 3D OpenGL Viewport powered by ModernGL + PyGLM with custom Skyrim GLSL shaders.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from PySide6.QtCore import QPoint, QRect, QSignalBlocker, QSize, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QCompleter,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
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

from .asset_scanner import (
    NifSearchThread,
    directory_contains_nif_or_hkx,
    discover_mod_textures_for_nif,
    get_mod_name_for_nif,
    is_filtered_folder_name,
    is_filtered_nif_filename,
    normalize_mods_root,
)
from .config import AppConfig, LightingConfig, ViewportEffectsConfig, load_locales
from .mod_scanner import ModScanner, is_animation_hkx
from .nif_loader import LoadedNifData, NifLoader
from .texture_popover import NifTexturePopover, resolve_shape_texture_path
from .texture_resolver import TextureResolver
from .ui_widgets import (
    HEADER_ACTIONS_STYLE,
    NIF_DRAG_MIME_TYPE,
    ROLE_ADDED_NIF_PATH,
    ROLE_FOLDER_PATH,
    ROLE_HKX_PATH,
    ROLE_IS_LOADED,
    ROLE_MOD_RELATIVE_PATH,
    ROLE_MOD_ROOT_PATH,
    ROLE_NIF_PATH,
    ROLE_OVERWRITE_FALLBACK,
    ROLE_OVERWRITE_RELATIVE_PATH,
    AddedNifRowWidget,
    AnimationPlaybackToolbar,
    ElidedLabel,
    NeatTreeWidget,
    NifDropPanel,
    PresetNifListWidget,
    RefreshableGroupBox,
)
from .viewport import OutfitViewportWidget

log = logging.getLogger("dibellas_wardrobe.ui")

_ROLE_SEARCH_FOLDER = Qt.ItemDataRole.UserRole + 9

__all__ = [
    "DibellasWardrobeWindow",
    "AddedNifRowWidget",
    "AnimationPlaybackToolbar",
    "ElidedLabel",
    "NeatTreeWidget",
    "NifSearchThread",
    "NifTexturePopover",
    "PresetNifListWidget",
    "RefreshableGroupBox",
    "ROLE_ADDED_NIF_PATH",
    "ROLE_FOLDER_PATH",
    "ROLE_HKX_PATH",
    "ROLE_IS_LOADED",
    "ROLE_MOD_RELATIVE_PATH",
    "ROLE_MOD_ROOT_PATH",
    "ROLE_NIF_PATH",
    "ROLE_OVERWRITE_FALLBACK",
    "ROLE_OVERWRITE_RELATIVE_PATH",
    "NIF_DRAG_MIME_TYPE",
    "directory_contains_nif_or_hkx",
    "is_filtered_folder_name",
    "is_filtered_nif_filename",
    "normalize_mods_root",
]


class DibellasWardrobeWindow(QMainWindow):
    """Main window with outfit mesh tree, preset manager, 3D viewport, and settings."""

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
            initial_folder = normalize_mods_root(Path(initial_root_dir).resolve())
            self.current_root_dir: Optional[Path] = (
                initial_folder if initial_folder.is_dir() else None
            )
        elif cfg_folder and cfg_folder.is_dir():
            self.current_root_dir = normalize_mods_root(cfg_folder.resolve())
        else:
            self.current_root_dir = None

        self.mod_scanner = (
            ModScanner(self.current_root_dir)
            if self.current_root_dir and self.current_root_dir.name.casefold() == "mods"
            else None
        )
        extra_roots = [self.current_root_dir] if self.current_root_dir else None
        self.resolver = TextureResolver(extra_roots=extra_roots)
        if self.mod_scanner:
            self.resolver.configure_mo2_paths(
                self.current_root_dir,
                self.mod_scanner.overwrite_root,
            )
        self.loader = NifLoader(resolver=self.resolver)

        self._checked_nif_paths: Set[str] = set()
        self._added_nif_items: Dict[str, QListWidgetItem] = {}
        self._locked_presets: Dict[int, Set[str]] = {
            int(page): {str(Path(path).resolve()) for path in paths if path}
            for page, paths in self.config.locked_presets.items()
            if page.isdigit()
        }
        if not self._locked_presets:
            self._locked_presets = {1: set()}
        self._locked_preset_diffuse_textures: Dict[int, Dict[str, Dict[str, str]]] = {
            int(page): {
                str(Path(mesh).resolve()): shapes.copy()
                for mesh, shapes in meshes.items()
            }
            for page, meshes in self.config.locked_preset_diffuse_textures.items()
        }
        self._current_locked_preset = min(self._locked_presets)
        self._locked_preset_names: Dict[int, str] = {
            int(page): name.strip()
            for page, name in self.config.locked_preset_names.items()
            if page.isdigit() and name.strip()
        }
        self._locked_nif_items: Dict[str, QListWidgetItem] = {}
        self._nif_visibility: Dict[str, bool] = {}
        self._highlighted_tree_items: List[QTreeWidgetItem] = []
        self._loaded_nif_cache: Dict[str, LoadedNifData] = {}
        self._updating_tree: bool = False
        self._search_thread: Optional[NifSearchThread] = None
        self._search_groups: Dict[str, QTreeWidgetItem] = {}
        self._search_folder_groups: Dict[tuple[str, tuple[str, ...]], QTreeWidgetItem] = {}
        self._startup_locked_nif_refresh_pending = False
        self._active_hkx_path: Optional[str] = None

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(320)
        self._search_timer.timeout.connect(self._start_search_worker)

        self._apply_dark_theme()
        self.texture_popover = NifTexturePopover(
            on_apply_texture=self._on_apply_nif_texture,
            on_reset_textures=self._on_reset_nif_textures,
            translate_fn=self._t,
            parent=self,
        )
        self._build_ui()
        self.populate_file_tree()
        self._autoload_locked_nifs()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._startup_locked_nif_refresh_pending:
            self._startup_locked_nif_refresh_pending = False
            QTimer.singleShot(0, self._refresh_added_nifs)

    def closeEvent(self, event) -> None:
        try:
            self.save_config()
        except Exception as exc:
            log.warning("Could not save config on close: %s", exc)
        self.viewport.smp.close()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Localization and Theming
    # ------------------------------------------------------------------

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
        if hasattr(self, "locked_preset_combo"):
            self.locked_preset_combo.setToolTip(self._t("locked_preset_search_tip"))
            self.locked_preset_combo.lineEdit().setPlaceholderText(
                self._t("locked_preset_search")
            )
            self._refresh_locked_preset_selector()
        if hasattr(self, "btn_rename_locked_preset"):
            self.btn_rename_locked_preset.setToolTip(self._t("rename_locked_preset"))
        if hasattr(self, "btn_add_locked_preset"):
            self.btn_add_locked_preset.setToolTip(self._t("add_locked_preset"))
        for group in self.findChildren(RefreshableGroupBox):
            if group.btn_minimize is not None:
                group.set_minimize_tooltips(
                    self._t("minimize"),
                    self._t("restore"),
                )
        if hasattr(self, "anim_toolbar"):
            self.anim_toolbar.set_translations(
                self._t("anim_play"),
                self._t("anim_pause"),
                self._t("anim_stop"),
                self._t("anim_stop_tip"),
            )
            self._apply_smp_texts()
            self.texture_popover.retranslate_ui()
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
                loading_texts = {
                    translations["loading"]
                    for translations in self._translations.values()
                }
                key = "loading" if item.text(0) in loading_texts else "no_nif_files"
                item.setText(0, self._t(key))
            for index in range(item.childCount()):
                visit(item.child(index))

        for index in range(self.file_tree.topLevelItemCount()):
            visit(self.file_tree.topLevelItem(index))

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #1b1d24;
                color: #e4e7ef;
                font-family: 'Segoe UI', system-ui, sans-serif;
                font-size: 10pt;
            }
            QGroupBox {
                border: 1px solid #363c50;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: 600;
                color: #f0d59c;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLineEdit {
                background-color: #242834;
                border: 1px solid #3c445a;
                border-radius: 4px;
                color: #e4e7ef;
                padding: 4px 8px;
            }
            QLineEdit:focus {
                border-color: #f0d59c;
            }
            QPushButton {
                background-color: #2e3444;
                border: 1px solid #454d68;
                border-radius: 4px;
                color: #e4e7ef;
                padding: 5px 12px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #3b4358;
                border-color: #f0d59c;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #242834;
            }
            QCheckBox {
                spacing: 6px;
                color: #c5cee2;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #2e3444;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #b58d3d;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #f0d59c;
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }
            QTreeWidget, QListWidget {
                background-color: #171920;
                border: 1px solid #2e3444;
                border-radius: 4px;
                color: #c5cee2;
            }
            QTreeWidget::item:hover, QListWidget::item:hover {
                background-color: #242937;
            }
            QTreeWidget::item:selected, QListWidget::item:selected {
                background-color: #343c52;
                color: #ffffff;
            }
            QScrollBar:vertical {
                background: #171920;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #363c50;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #505874;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QSplitter::handle {
                background-color: #242834;
                width: 4px;
            }
            """
        )

    # ------------------------------------------------------------------
    # UI Layout Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.menuBar().setVisible(False)
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(0)

        # Global shortcuts
        self.shortcut_open = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_open.activated.connect(self._on_browse_root_folder)
        self.shortcut_filter = QShortcut(QKeySequence("Ctrl+F"), self)
        self.shortcut_filter.activated.connect(lambda: self._toggle_mod_dir_filter())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(splitter, stretch=1)

        # COLUMN 1: Mod Directory & File Tree
        mesh_panel = NifDropPanel(self._on_nif_dropped_on_mods)
        mesh_layout = QVBoxLayout(mesh_panel)
        mesh_layout.setContentsMargins(2, 2, 2, 2)
        mesh_layout.setSpacing(6)

        dir_group = RefreshableGroupBox(self._t("mod_directory"), minimizable=True)
        self._bind_text(dir_group, "mod_directory", "setTitle")
        dir_content = QWidget()
        dir_layout = QVBoxLayout(dir_content)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_group_layout = QVBoxLayout(dir_group)
        dir_group_layout.addWidget(dir_content)
        dir_group.set_collapsible_content(dir_content)
        dir_group.set_minimize_tooltips(self._t("minimize"), self._t("restore"))

        path_row = QHBoxLayout()
        self.root_path_edit = QLineEdit(str(self.current_root_dir) if self.current_root_dir else "")
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
        self._i18n_bindings.append((self.chk_meshes_and_caliente_only, "show_meshes_caliente_tip", "setToolTip"))

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
        mesh_layout.addWidget(dir_group)

        tree_group = RefreshableGroupBox(self._t("outfit_meshes"), self._refresh_file_tree)
        self._bind_text(tree_group, "outfit_meshes", "setTitle")
        self.btn_refresh_tree = tree_group.btn_refresh
        assert self.btn_refresh_tree is not None
        self._bind_text(self.btn_refresh_tree, "refresh_tree", "setToolTip")
        tree_layout = QVBoxLayout(tree_group)

        self.search_edit = QLineEdit()
        self._bind_text(self.search_edit, "search_mod", "setPlaceholderText")
        self.search_edit.textChanged.connect(self._on_search_text_changed)
        tree_layout.addWidget(self.search_edit)

        self.file_tree = NeatTreeWidget()
        self.file_tree.setDragDropMode(NeatTreeWidget.DragDropMode.DragDrop)
        self.file_tree.setDragEnabled(True)
        self.file_tree.setAcceptDrops(True)
        self.file_tree.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.file_tree.setDropIndicatorShown(True)
        self.file_tree.on_nif_dropped = self._on_nif_dropped_on_mods
        self.file_tree.itemExpanded.connect(self._on_tree_item_expanded)
        self.file_tree.itemClicked.connect(self._on_tree_item_clicked)
        self.file_tree.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
        tree_layout.addWidget(self.file_tree, stretch=1)

        mesh_layout.addWidget(tree_group, stretch=1)
        mesh_panel.enable_drop_targets()
        self.mesh_panel = mesh_panel
        splitter.addWidget(mesh_panel)

        # COLUMN 2: Presets & Wardrobe Box
        added_panel = QWidget()
        added_layout = QVBoxLayout(added_panel)
        added_layout.setContentsMargins(2, 2, 2, 2)
        added_layout.setSpacing(6)

        locked_group = RefreshableGroupBox(self._t("locked_wardrobe", count=0), minimizable=True)
        locked_content = QWidget()
        locked_group_layout = QVBoxLayout(locked_content)
        locked_group_layout.setContentsMargins(0, 0, 0, 0)
        locked_group_box_layout = QVBoxLayout(locked_group)
        locked_group_box_layout.addWidget(locked_content)
        locked_group.set_collapsible_content(locked_content)
        locked_group.set_minimize_tooltips(self._t("minimize"), self._t("restore"))

        self.locked_nifs_list = PresetNifListWidget()
        self.locked_nifs_list.on_nif_dropped = self._on_nif_dropped_on_preset
        self.locked_nifs_list.setMinimumHeight(0)
        self.locked_nifs_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        preset_bar = QHBoxLayout()
        preset_bar.setContentsMargins(0, 0, 0, 0)
        preset_bar.setSpacing(4)
        self.locked_preset_combo = QComboBox()
        self.locked_preset_combo.setEditable(True)
        self.locked_preset_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.locked_preset_combo.setFixedHeight(26)
        self.locked_preset_combo.setToolTip(self._t("locked_preset_search_tip"))
        self.locked_preset_combo.lineEdit().setPlaceholderText(self._t("locked_preset_search"))
        completer = QCompleter(self.locked_preset_combo.model(), self.locked_preset_combo)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.locked_preset_combo.setCompleter(completer)
        self.locked_preset_combo.currentIndexChanged.connect(self._on_locked_preset_changed)
        preset_bar.addWidget(self.locked_preset_combo, stretch=1)

        preset_actions = QFrame()
        preset_actions.setObjectName("headerActions")
        preset_actions.setStyleSheet(HEADER_ACTIONS_STYLE)
        preset_actions_layout = QHBoxLayout(preset_actions)
        preset_actions_layout.setContentsMargins(1, 1, 1, 1)
        preset_actions_layout.setSpacing(2)

        self.btn_rename_locked_preset = QPushButton("✎")
        self.btn_rename_locked_preset.setObjectName("headerActionButton")
        self.btn_rename_locked_preset.setFixedSize(22, 22)
        self.btn_rename_locked_preset.setToolTip(self._t("rename_locked_preset"))
        self.btn_rename_locked_preset.clicked.connect(self._rename_locked_preset)
        preset_actions_layout.addWidget(self.btn_rename_locked_preset)

        self.btn_add_locked_preset = QPushButton("+")
        self.btn_add_locked_preset.setObjectName("headerActionButton")
        self.btn_add_locked_preset.setFixedSize(22, 22)
        self.btn_add_locked_preset.setToolTip(self._t("add_locked_preset"))
        self.btn_add_locked_preset.clicked.connect(self._add_locked_preset)
        preset_actions_layout.addWidget(self.btn_add_locked_preset)
        preset_bar.addWidget(preset_actions)

        locked_group_layout.addLayout(preset_bar)
        self.locked_nifs_list.itemClicked.connect(self._on_added_nif_clicked)
        locked_group_layout.addWidget(self.locked_nifs_list, stretch=1)
        self._refresh_locked_preset_selector()
        added_layout.addWidget(locked_group)

        added_group = QGroupBox(self._t("wardrobe", count=0))
        added_group_layout = QVBoxLayout(added_group)
        self.added_nifs_list = PresetNifListWidget()
        self.added_nifs_list.on_nif_dropped = self._on_nif_dropped_on_wardrobe
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

        # COLUMN 3: 3D OpenGL Viewport
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 4, 0, 4)
        center_layout.setSpacing(4)

        self.viewport = OutfitViewportWidget(self)
        overlay = self.viewport.cbpc_overlay
        self._bind_text(overlay.checkbox, "cbpc_enable")
        for key in ("gravity", "breasts", "butt", "belly", "thigh"):
            locale_key = f"cbpc_{key}"
            self._bind_text(overlay.captions[key], locale_key)
            slider = overlay.slider if key == "gravity" else overlay.softness_sliders[key]
            self._bind_text(slider, locale_key, "setAccessibleName")
        self._init_viewport_settings()
        center_layout.addWidget(self.viewport, stretch=1)

        self.anim_toolbar = AnimationPlaybackToolbar(
            on_toggle_play=self.viewport.toggle_play_pause,
            on_seek=self.viewport.seek_animation,
            on_close=self._on_stop_animation,
            parent=center_panel,
        )
        self.anim_toolbar.set_translations(
            self._t("anim_play"),
            self._t("anim_pause"),
            self._t("anim_stop"),
            self._t("anim_stop_tip"),
        )
        self._apply_smp_texts()
        self.viewport.smpStateChanged.connect(self._on_smp_state_changed)
        self.viewport.set_smp_enabled(self.config.smp.enabled)
        self.anim_toolbar.setVisible(False)
        self.viewport.animationTimeChanged.connect(self.anim_toolbar.update_time)
        self.viewport.animationStateChanged.connect(self.anim_toolbar.update_state)
        self.viewport.animationStopped.connect(self._on_viewport_animation_stopped)
        center_layout.addWidget(self.anim_toolbar)

        cam_hint_row = QHBoxLayout()
        cam_hint = QLabel()
        self._bind_text(cam_hint, "camera_controls")
        cam_hint.setStyleSheet("color: #8f98b0; font-size: 12px;")
        btn_reset_cam = QPushButton()
        self._bind_text(btn_reset_cam, "reset_camera")
        btn_reset_cam.setShortcut("R")
        btn_reset_cam.clicked.connect(self.viewport.reset_camera)

        self.btn_settings = QPushButton("⚙")
        self.btn_settings.setFixedSize(30, 30)
        self.btn_settings.setStyleSheet(
            """
            QPushButton {
                background-color: #2e3444;
                border: 1px solid #454d68;
                border-radius: 4px;
                color: #e4e7ef;
                font-size: 16px;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #3b4358;
                border-color: #f0d59c;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #242834;
            }
            """
        )
        self._bind_text(self.btn_settings, "settings", "setToolTip")
        self.btn_settings.setShortcut("S")
        self.btn_settings.clicked.connect(lambda: self._toggle_settings_sidebar())

        cam_hint_row.addWidget(cam_hint)
        cam_hint_row.addStretch(1)
        cam_hint_row.addWidget(btn_reset_cam)
        cam_hint_row.addWidget(self.btn_settings)
        center_layout.addLayout(cam_hint_row)

        splitter.addWidget(center_panel)

        # COLUMN 4: Settings Sidebar
        self._build_settings_sidebar(splitter)

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

    def _init_viewport_settings(self) -> None:
        cfg_light = self.config.lighting
        cfg_effects = self.config.effects
        cfg_chk = self.config.checkboxes
        s = self.viewport.settings

        s.light_azimuth_deg = cfg_light.light_azimuth_deg
        s.light_elevation_deg = cfg_light.light_elevation_deg
        s.light_intensity = cfg_light.light_intensity
        s.ambient_intensity = cfg_light.ambient_intensity
        s.specular_multiplier = cfg_light.specular_multiplier
        s.env_reflection_multiplier = cfg_light.env_reflection_multiplier

        s.enable_ssr = cfg_effects.enable_ssr
        s.ssr_step_size = cfg_effects.ssr_step_size
        s.ssr_max_steps = cfg_effects.ssr_max_steps
        s.ssr_thickness = cfg_effects.ssr_thickness
        s.enable_ao = cfg_effects.enable_ao
        s.ao_radius = cfg_effects.ao_radius
        s.ao_intensity = cfg_effects.ao_intensity
        s.ao_bias = cfg_effects.ao_bias
        s.enable_ssss = cfg_effects.enable_ssss
        s.ssss_radius = cfg_effects.ssss_radius
        s.ssss_falloff_color = cfg_effects.ssss_falloff_color

        cb = self.config.cbpc
        self.viewport.set_cbpc_gravity(cb.gravity)
        for group, factor in cb.softness.items():
            self.viewport.set_cbpc_softness(group, factor)
        self.viewport.set_cbpc_enabled(cb.enabled)

        s.wireframe_overlay = cfg_chk.wireframe_overlay
        s.show_ground_grid = cfg_chk.show_ground_grid

    def _build_settings_sidebar(self, splitter: QSplitter) -> None:
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

        self.language_combo = QComboBox()
        for language_code, translations in self._translations.items():
            self.language_combo.addItem(translations["language_name"], language_code)
        self.language_combo.setCurrentIndex(self.language_combo.findData(self.language))
        self.language_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.language_combo.setFixedHeight(28)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)

        sidebar_header.addWidget(sidebar_title)
        sidebar_header.addStretch(1)
        sidebar_header.addWidget(self.language_combo)
        sidebar_header.addWidget(btn_close_sidebar)
        right_layout.addLayout(sidebar_header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)

        cfg_light = self.config.lighting
        cfg_effects = self.config.effects
        cfg_chk = self.config.checkboxes

        physics_group = QGroupBox()
        self._bind_text(physics_group, "physics_settings", "setTitle")
        physics_layout = QVBoxLayout(physics_group)
        physics_layout.addWidget(self.viewport.cbpc_overlay)
        self.viewport.cbpc_overlay.show()
        scroll_layout.addWidget(physics_group)

        # Lighting Group
        light_group = RefreshableGroupBox(self._t("lighting_hemisphere"), self._reset_lighting_controls)
        self._bind_text(light_group, "lighting_hemisphere", "setTitle")
        light_form = QFormLayout(light_group)
        self.btn_reset_lighting = light_group.btn_refresh
        assert self.btn_reset_lighting is not None
        self._bind_text(self.btn_reset_lighting, "reset_lighting", "setToolTip")

        self.slider_light_azimuth, self.lbl_azimuth_val = self._add_slider_row(
            light_form, self._t("light_rotation"), -180, 180, int(round(cfg_light.light_azimuth_deg)),
            f"{int(round(cfg_light.light_azimuth_deg))}°", self._on_lighting_changed, "light_rotation",
        )
        self.slider_light_elevation, self.lbl_elevation_val = self._add_slider_row(
            light_form, self._t("light_elevation"), -60, 85, int(round(cfg_light.light_elevation_deg)),
            f"{int(round(cfg_light.light_elevation_deg))}°", self._on_lighting_changed, "light_elevation",
        )
        self.slider_key_intensity, self.lbl_key_val = self._add_slider_row(
            light_form, self._t("key_light_power"), 0, 200, int(round(cfg_light.light_intensity * 100)),
            f"{cfg_light.light_intensity:.2f}", self._on_lighting_changed, "key_light_power",
        )
        self.slider_ambient, self.lbl_ambient_val = self._add_slider_row(
            light_form, self._t("ambient_light"), 0, 100, int(round(cfg_light.ambient_intensity * 100)),
            f"{cfg_light.ambient_intensity:.2f}", self._on_lighting_changed, "ambient_light",
        )
        self.slider_specular, self.lbl_spec_val = self._add_slider_row(
            light_form, self._t("specular_scale"), 0, 300, int(round(cfg_light.specular_multiplier * 100)),
            f"{cfg_light.specular_multiplier:.2f}x", self._on_lighting_changed, "specular_scale",
        )
        self.slider_env, self.lbl_env_val = self._add_slider_row(
            light_form, self._t("env_reflection"), 0, 300, int(round(cfg_light.env_reflection_multiplier * 100)),
            f"{cfg_light.env_reflection_multiplier:.2f}x", self._on_lighting_changed, "env_reflection",
        )
        scroll_layout.addWidget(light_group)

        # Meshes / Shader Options
        shader_group = QGroupBox()
        self._bind_text(shader_group, "shader_options", "setTitle")
        shader_layout = QVBoxLayout(shader_group)
        self.chk_wireframe = self._add_checkbox(
            shader_layout, self._t("wireframe_overlay"),
            cfg_chk.wireframe_overlay, self._on_shader_toggles_changed,
        )
        self._bind_text(self.chk_wireframe, "wireframe_overlay")
        self.chk_grid = self._add_checkbox(
            shader_layout, self._t("ground_grid"),
            cfg_chk.show_ground_grid, self._on_shader_toggles_changed,
        )
        self._bind_text(self.chk_grid, "ground_grid")
        scroll_layout.addWidget(shader_group)

        # SSR Group
        ssr_group = RefreshableGroupBox(self._t("ssr_options"), self._reset_ssr_controls)
        self._bind_text(ssr_group, "ssr_options", "setTitle")
        assert ssr_group.btn_refresh is not None
        self._bind_text(ssr_group.btn_refresh, "reset_ssr", "setToolTip")
        ssr_layout = QVBoxLayout(ssr_group)
        self.chk_ssr = self._add_checkbox(
            ssr_layout, self._t("ssr_enabled"), cfg_effects.enable_ssr, self._on_effects_changed
        )
        self._bind_text(self.chk_ssr, "ssr_enabled")
        ssr_form = QFormLayout()
        self.slider_ssr_step, self.lbl_ssr_step = self._add_slider_row(
            ssr_form, self._t("ssr_step_size"), 1, 50, int(round(cfg_effects.ssr_step_size * 10)),
            f"{cfg_effects.ssr_step_size:.1f}", self._on_effects_changed, "ssr_step_size",
        )
        self.slider_ssr_steps, self.lbl_ssr_steps = self._add_slider_row(
            ssr_form, self._t("ssr_max_steps"), 8, 128, int(cfg_effects.ssr_max_steps),
            str(cfg_effects.ssr_max_steps), self._on_effects_changed, "ssr_max_steps",
        )
        self.slider_ssr_thickness, self.lbl_ssr_thickness = self._add_slider_row(
            ssr_form, self._t("ssr_thickness"), 1, 100, int(round(cfg_effects.ssr_thickness * 10)),
            f"{cfg_effects.ssr_thickness:.1f}", self._on_effects_changed, "ssr_thickness",
        )
        ssr_layout.addLayout(ssr_form)
        scroll_layout.addWidget(ssr_group)

        # AO Group
        ao_group = RefreshableGroupBox(self._t("ao_options"), self._reset_ao_controls)
        self._bind_text(ao_group, "ao_options", "setTitle")
        assert ao_group.btn_refresh is not None
        self._bind_text(ao_group.btn_refresh, "reset_ao", "setToolTip")
        ao_layout = QVBoxLayout(ao_group)
        self.chk_ao = self._add_checkbox(
            ao_layout, self._t("ao_enabled"), cfg_effects.enable_ao, self._on_effects_changed
        )
        self._bind_text(self.chk_ao, "ao_enabled")
        ao_form = QFormLayout()
        self.slider_ao_radius, self.lbl_ao_radius = self._add_slider_row(
            ao_form, self._t("ao_radius"), 1, 40, int(round(cfg_effects.ao_radius)),
            f"{cfg_effects.ao_radius:.0f}", self._on_effects_changed, "ao_radius",
        )
        self.slider_ao_intensity, self.lbl_ao_intensity = self._add_slider_row(
            ao_form, self._t("ao_intensity"), 0, 300, int(round(cfg_effects.ao_intensity * 100)),
            f"{cfg_effects.ao_intensity:.2f}", self._on_effects_changed, "ao_intensity",
        )
        self.slider_ao_bias, self.lbl_ao_bias = self._add_slider_row(
            ao_form, self._t("ao_bias"), 1, 100, int(round(cfg_effects.ao_bias * 1000)),
            f"{cfg_effects.ao_bias:.3f}", self._on_effects_changed, "ao_bias",
        )
        ao_layout.addLayout(ao_form)
        scroll_layout.addWidget(ao_group)

        # SSSS Group
        ssss_group = RefreshableGroupBox(self._t("ssss_options"), self._reset_ssss_controls)
        self._bind_text(ssss_group, "ssss_options", "setTitle")
        assert ssss_group.btn_refresh is not None
        self._bind_text(ssss_group.btn_refresh, "reset_ssss", "setToolTip")
        ssss_layout = QVBoxLayout(ssss_group)
        self.chk_ssss = self._add_checkbox(
            ssss_layout, self._t("ssss_enabled"), cfg_effects.enable_ssss, self._on_effects_changed
        )
        self._bind_text(self.chk_ssss, "ssss_enabled")
        ssss_form = QFormLayout()
        self.slider_ssss_radius, self.lbl_ssss_radius = self._add_slider_row(
            ssss_form, self._t("ssss_radius"), 1, 24, int(round(cfg_effects.ssss_radius)),
            f"{cfg_effects.ssss_radius:.0f}", self._on_effects_changed, "ssss_radius",
        )
        self.btn_ssss_falloff = QPushButton()
        self.btn_ssss_falloff.clicked.connect(self._choose_ssss_falloff_color)
        falloff_row = QWidget()
        falloff_layout = QHBoxLayout(falloff_row)
        falloff_layout.setContentsMargins(0, 0, 0, 0)
        falloff_layout.addWidget(self.btn_ssss_falloff)
        falloff_label = QLabel()
        self._bind_text(falloff_label, "ssss_falloff_color")
        ssss_form.addRow(falloff_label, falloff_row)
        ssss_layout.addLayout(ssss_form)
        scroll_layout.addWidget(ssss_group)

        self._update_ssss_falloff_button()
        self._on_effects_changed()

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        right_layout.addWidget(scroll, stretch=1)

        self.right_panel = right_panel
        right_layout.activate()
        scroll_content.ensurePolished()
        scroll_layout.activate()
        scroll_content.adjustSize()
        content_width = max(
            scroll_content.minimumSizeHint().width(),
            scroll_layout.minimumSize().width(),
        )
        self._settings_sidebar_min_width = content_width + 12
        right_panel.setMinimumWidth(self._settings_sidebar_min_width)
        splitter.addWidget(right_panel)
        self.right_panel.setVisible(False)

    def _toggle_mod_dir_filter(self, visible: Optional[bool] = None) -> None:
        """Minimize or restore the Mod Directory controls."""
        if visible is None:
            visible = self.dir_group._collapsed
        self.dir_group.set_collapsed(not visible)

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
                self.right_panel.ensurePolished()
                scroll = self.right_panel.findChild(QScrollArea)
                if scroll is not None and scroll.widget() is not None:
                    content = scroll.widget()
                    content.ensurePolished()
                    if content.layout() is not None:
                        content.layout().activate()
                        content.adjustSize()
                        self._settings_sidebar_min_width = max(
                            content.minimumSizeHint().width(),
                            content.layout().minimumSize().width(),
                        ) + 12
                        self.right_panel.setMinimumWidth(self._settings_sidebar_min_width)
                right_target_w = self._settings_sidebar_min_width
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
        i18n_key: Optional[str] = None,
    ) -> tuple[QSlider, QLabel]:
        label = QLabel(label_text)
        if i18n_key:
            self._bind_text(label, i18n_key)
        label.setStyleSheet("color: #c5cee2; font-size: 12px;")

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(init_val)
        slider.valueChanged.connect(callback)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        val_label = QLabel(val_text)
        val_label.setFixedWidth(52)
        val_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        val_label.setStyleSheet("color: #f0d59c; font-family: monospace; font-size: 11px;")

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(slider, stretch=1)
        row_layout.addWidget(val_label)
        form.addRow(label, row_widget)
        return slider, val_label

    @staticmethod
    def _add_checkbox(layout: QVBoxLayout, text: str, checked: bool, callback) -> QCheckBox:
        chk = QCheckBox(text)
        chk.setChecked(checked)
        chk.toggled.connect(callback)
        layout.addWidget(chk)
        return chk

    def _create_folder_tree_item(
        self,
        parent: QTreeWidget | QTreeWidgetItem,
        name: str,
        path: str,
        color: str,
        mod_root: Optional[Path] = None,
        mod_relative_path: tuple[str, ...] = (),
        overwrite_fallback: bool = False,
        overwrite_relative_path: tuple[str, ...] = (),
    ) -> QTreeWidgetItem:
        folder_item = QTreeWidgetItem(parent, [name])
        folder_item.setData(0, ROLE_FOLDER_PATH, path)
        folder_item.setData(0, ROLE_IS_LOADED, False)
        if mod_root is not None:
            folder_item.setData(0, ROLE_MOD_ROOT_PATH, str(mod_root))
            folder_item.setData(0, ROLE_MOD_RELATIVE_PATH, mod_relative_path)
        if overwrite_fallback:
            folder_item.setData(0, ROLE_OVERWRITE_FALLBACK, True)
            folder_item.setData(0, ROLE_OVERWRITE_RELATIVE_PATH, overwrite_relative_path)
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

    @staticmethod
    def _create_hkx_tree_item(parent: QTreeWidget | QTreeWidgetItem, name: str, path: str) -> QTreeWidgetItem:
        label = name if name.startswith("[HKX]") else f"[HKX] {name}"
        hkx_item = QTreeWidgetItem(parent, [label])
        hkx_item.setData(0, ROLE_HKX_PATH, path)
        hkx_item.setToolTip(0, path)
        hkx_item.setForeground(0, QColor("#7fd4ff"))
        hkx_item.setFlags(hkx_item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        return hkx_item

    # ------------------------------------------------------------------
    # Dynamic On-Demand Tree Population & Search
    # ------------------------------------------------------------------

    def set_root_directory(self, folder: Path | str) -> None:
        target = normalize_mods_root(Path(folder).resolve())
        if not target.is_dir():
            return
        self.current_root_dir = target
        self.mod_scanner = (
            ModScanner(target) if target.name.casefold() == "mods" else None
        )
        self.resolver.configure_mo2_paths(
            target if self.mod_scanner else None,
            self.mod_scanner.overwrite_root if self.mod_scanner else None,
        )
        self.root_path_edit.setText(str(target))
        if not self.mod_scanner:
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

    def _refresh_file_tree(self) -> None:
        if self._search_thread and self._search_thread.isRunning():
            self._search_thread.cancel()
            self._search_thread.wait()
        if self.mod_scanner and self.current_root_dir:
            self.mod_scanner = ModScanner(self.current_root_dir)
            self.resolver.configure_mo2_paths(
                self.current_root_dir, self.mod_scanner.overwrite_root
            )
            for nif_path in self._checked_nif_paths:
                owner = self.mod_scanner.owner_for_nif(nif_path)
                if owner is not None:
                    self.resolver.register_nif_owner(nif_path, owner)
        self.resolver.clear_cache()
        self._on_filter_toggled()

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

        directory_contains_nif_or_hkx.cache_clear()
        self._updating_tree = True
        self._clear_file_tree()
        self._updating_tree = False

        self._search_thread = NifSearchThread(
            root_dir=self.current_root_dir,
            query=query,
            hide_0=self.chk_hide_low_weight.isChecked(),
            hide_gnd_folder=self.chk_hide_gnd_folder.isChecked(),
            meshes_and_caliente_only=self.chk_meshes_and_caliente_only.isChecked(),
            mod_scanner=self.mod_scanner,
            parent=self,
        )
        self._search_thread.matchFound.connect(self._on_search_match_found)
        self._search_thread.finished.connect(self._auto_expand_search_results)
        self._search_thread.start()

    def _on_search_match_found(
        self,
        group_name: str,
        relative_directory: str,
        file_name: str,
        full_path: str,
    ) -> None:
        self._updating_tree = True
        if group_name not in self._search_groups:
            grp_item = QTreeWidgetItem(self.file_tree, [group_name])
            grp_item.setForeground(0, QColor("#f0d59c"))
            grp_item.setData(0, _ROLE_SEARCH_FOLDER, True)
            self._search_groups[group_name] = grp_item

        parent_item = self._search_groups[group_name]
        directory_parts = tuple(
            part for part in relative_directory.replace("\\", "/").split("/") if part
        )
        accumulated_parts: tuple[str, ...] = ()
        for folder_name in directory_parts:
            accumulated_parts += (folder_name.casefold(),)
            folder_key = (group_name.casefold(), accumulated_parts)
            folder_item = self._search_folder_groups.get(folder_key)
            if folder_item is None:
                folder_item = QTreeWidgetItem(parent_item, [folder_name])
                folder_item.setForeground(0, QColor("#d7c7a6"))
                folder_item.setData(0, ROLE_IS_LOADED, True)
                folder_item.setData(0, _ROLE_SEARCH_FOLDER, True)
                self._search_folder_groups[folder_key] = folder_item
            parent_item = folder_item

        if full_path.casefold().endswith(".hkx"):
            self._create_hkx_tree_item(parent_item, file_name, full_path)
        else:
            self._create_nif_tree_item(parent_item, file_name, full_path)
        self._updating_tree = False

    def _auto_expand_search_results(self) -> None:
        """Expand search-result folder chains with only one visible child folder."""
        for group_item in self._search_groups.values():
            self._auto_expand_search_item(group_item)

    def _auto_expand_search_item(self, item: QTreeWidgetItem) -> None:
        with QSignalBlocker(self.file_tree):
            while item.childCount() == 1:
                child = item.child(0)
                if not child.data(0, _ROLE_SEARCH_FOLDER):
                    break
                child.setExpanded(True)
                item = child

    def populate_file_tree(self) -> None:
        if self._search_thread and self._search_thread.isRunning():
            self._search_thread.cancel()
            self._search_thread.wait(300)

        self._updating_tree = True
        self._clear_file_tree()

        if self.current_root_dir is None:
            self._updating_tree = False
            return

        directory_contains_nif_or_hkx.cache_clear()
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

        is_mo2_root = bool(self.mod_scanner and self.mod_scanner.is_mods_root)

        for entry in entries:
            try:
                if entry.is_dir():
                    if is_filtered_folder_name(entry.name, hide_gnd_folder):
                        continue
                    if (
                        meshes_caliente_only
                        and not directory_contains_nif_or_hkx(entry.path)
                        and not (
                            is_mo2_root
                            and self.mod_scanner
                            and self.mod_scanner.has_mod_overlay(entry.path)
                        )
                    ):
                        continue
                    self._create_folder_tree_item(
                        self.file_tree,
                        entry.name,
                        entry.path,
                        "#e0d2b4",
                        mod_root=Path(entry.path) if is_mo2_root else None,
                    )
                elif entry.is_file():
                    if not is_filtered_nif_filename(entry.name, hide_0, hide_gnd_folder):
                        self._create_nif_tree_item(self.file_tree, entry.name, entry.path)
                    elif is_animation_hkx(entry.name):
                        self._create_hkx_tree_item(self.file_tree, entry.name, entry.path)
            except OSError:
                continue

        if self.mod_scanner:
            self._ensure_overwrite_fallback_item()
        self._updating_tree = False

    def _clear_file_tree(self) -> None:
        self._highlighted_tree_items.clear()
        self._search_groups.clear()
        self._search_folder_groups.clear()
        self.file_tree.clear()

    def _on_tree_item_expanded(self, item: QTreeWidgetItem) -> None:
        if item.data(0, _ROLE_SEARCH_FOLDER):
            self._auto_expand_search_item(item)
            return

        if item.data(0, ROLE_IS_LOADED):
            return

        folder_path = item.data(0, ROLE_FOLDER_PATH)
        mod_root_value = item.data(0, ROLE_MOD_ROOT_PATH)
        is_overwrite_fallback = bool(item.data(0, ROLE_OVERWRITE_FALLBACK))
        if not folder_path or (
            not os.path.isdir(folder_path)
            and not mod_root_value
            and not is_overwrite_fallback
        ):
            return

        self._updating_tree = True
        item.takeChildren()

        hide_0 = self.chk_hide_low_weight.isChecked()
        hide_gnd_folder = self.chk_hide_gnd_folder.isChecked()
        meshes_caliente_only = self.chk_meshes_and_caliente_only.isChecked()

        try:
            mod_root = Path(mod_root_value).resolve() if mod_root_value else None
            mod_relative_path = tuple(item.data(0, ROLE_MOD_RELATIVE_PATH) or ())
            fallback_relative_path = tuple(item.data(0, ROLE_OVERWRITE_RELATIVE_PATH) or ())
            local_entries = []
            if os.path.isdir(folder_path) and not is_overwrite_fallback:
                local_entries = sorted(
                    list(os.scandir(folder_path)),
                    key=lambda e: (not e.is_dir(), e.name.lower()),
                )

            overlay_entries = []
            if self.mod_scanner and mod_root is not None:
                overlay_entries = self.mod_scanner.mod_overlay_entries(
                    mod_root, mod_relative_path
                )
                self._ensure_overwrite_fallback_item()
            elif self.mod_scanner and is_overwrite_fallback:
                overlay_entries = self.mod_scanner.fallback_entries(fallback_relative_path)

            merged_entries = {
                entry.name.casefold(): (entry.name, entry.is_dir(), entry.path, False)
                for entry in local_entries
            }
            for entry in overlay_entries:
                entry_key = (
                    str(entry.path).casefold()
                    if is_overwrite_fallback and not entry.is_dir
                    else entry.name.casefold()
                )
                merged_entries[entry_key] = (
                    entry.name,
                    entry.is_dir,
                    entry.path,
                    not entry.is_dir,
                )
            entries = sorted(
                merged_entries.values(),
                key=lambda entry: (not entry[1], entry[0].casefold()),
            )
            added_any = False
            for name, is_dir, entry_path, is_overwritten in entries:
                if is_dir:
                    if is_filtered_folder_name(name, hide_gnd_folder):
                        continue
                    if (
                        meshes_caliente_only
                        and not directory_contains_nif_or_hkx(str(entry_path))
                        and not (
                            self.mod_scanner
                            and mod_root is not None
                            and self.mod_scanner.has_mod_overlay(
                                mod_root, mod_relative_path + (name,)
                            )
                        )
                    ):
                        continue
                    relative_child_path = mod_relative_path + (name,)
                    self._create_folder_tree_item(
                        item,
                        name,
                        str(entry_path),
                        "#c5cee2",
                        mod_root=mod_root,
                        mod_relative_path=relative_child_path if mod_root else (),
                        overwrite_fallback=is_overwrite_fallback,
                        overwrite_relative_path=(
                            fallback_relative_path + (name,) if is_overwrite_fallback else ()
                        ),
                    )
                    added_any = True
                elif name.casefold().endswith(".nif") and not is_filtered_nif_filename(
                    name, hide_0, hide_gnd_folder
                ):
                    display_name = f"[{name}]" if is_overwritten else name
                    self._create_nif_tree_item(item, display_name, str(entry_path))
                    added_any = True
                elif is_animation_hkx(name, mod_relative_path or fallback_relative_path):
                    display_name = f"[{name}]" if is_overwritten else name
                    self._create_hkx_tree_item(item, display_name, str(entry_path))
                    added_any = True

            if not added_any:
                empty_item = QTreeWidgetItem(item, [self._t("no_nif_files")])
                empty_item.setForeground(0, QColor("#6c7285"))
                empty_item.setFlags(Qt.ItemFlag.NoItemFlags)

            item.setData(0, ROLE_IS_LOADED, True)
            self._auto_expand_single_folder_child(item)
        except Exception as exc:
            log.warning("Error expanding %s: %s", folder_path, exc)
        finally:
            self._updating_tree = False

    def _auto_expand_single_folder_child(self, item: QTreeWidgetItem) -> None:
        """Expand a folder chain while each folder has one visible folder child."""
        if item.childCount() != 1:
            return

        child = item.child(0)
        if not child.data(0, ROLE_FOLDER_PATH):
            return

        child.setExpanded(True)
        self._on_tree_item_expanded(child)

    def _ensure_overwrite_fallback_item(self) -> None:
        if not self.mod_scanner or not self.mod_scanner.has_fallback():
            return
        for index in range(self.file_tree.topLevelItemCount()):
            item = self.file_tree.topLevelItem(index)
            if item.data(0, ROLE_OVERWRITE_FALLBACK):
                return
        overwrite_root = self.mod_scanner.overwrite_root
        if overwrite_root is None:
            return
        meshes_root = overwrite_root / "meshes"
        fallback_path = meshes_root if meshes_root.is_dir() else overwrite_root
        self._create_folder_tree_item(
            self.file_tree,
            "[Overwrite]",
            str(fallback_path),
            "#f0d59c",
            overwrite_fallback=True,
        )

    # ------------------------------------------------------------------
    # Tree Events and Animation Playback
    # ------------------------------------------------------------------

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._updating_tree:
            return

        hkx_key = item.data(0, ROLE_HKX_PATH)
        if hkx_key:
            self._load_and_play_animation(hkx_key)
            return

        nif_key = item.data(0, ROLE_NIF_PATH)
        if not nif_key or nif_key in self._locked_presets.get(self._current_locked_preset, set()):
            return

        if nif_key not in self._checked_nif_paths:
            self._load_and_attach_outfit_nif(nif_key)
        else:
            self._detach_outfit_nif(nif_key, remove_from_presets=False)

    def _on_tree_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._updating_tree:
            return
        hkx_key = item.data(0, ROLE_HKX_PATH)
        if hkx_key:
            self._load_and_play_animation(hkx_key)

    def _format_animation_metadata(self, hkx_path: Path) -> str:
        mod_name = self._mod_name_for_nif(str(hkx_path))
        parent_name = hkx_path.parent.name
        if parent_name and parent_name.casefold() != mod_name.casefold():
            return f"{mod_name} ({parent_name}) - {hkx_path.name}"
        return f"{mod_name} - {hkx_path.name}"

    def _load_and_play_animation(self, hkx_key: str, show_error: bool = True) -> bool:
        resolved_key = str(Path(hkx_key).resolve())
        try:
            loaded = self.viewport.load_animation(resolved_key, auto_play=True)
            self._active_hkx_path = resolved_key
            meta_label = self._format_animation_metadata(loaded.filepath)
            self.anim_toolbar.set_metadata(meta_label, loaded.duration)
            self.anim_toolbar.lbl_metadata.setToolTip(resolved_key)
            self.anim_toolbar.update_state(self.viewport.is_animation_playing)
            self.anim_toolbar.setVisible(True)
            return True
        except Exception as exc:
            log.error("Failed loading HKX animation %s: %s", hkx_key, exc)
            if show_error:
                QMessageBox.warning(
                    self,
                    self._t("hkx_load_error"),
                    self._t("hkx_load_error_message", filename=Path(hkx_key).name, error=exc),
                )
            return False

    def _on_stop_animation(self) -> None:
        self._active_hkx_path = None
        self.viewport.stop_animation()
        self.anim_toolbar.setVisible(False)

    def _apply_smp_texts(self) -> None:
        overlay = self.viewport.cbpc_overlay
        overlay.set_smp_texts(self._t("smp_enable"), self._t("smp_tip"))
        smp = self.viewport.smp
        overlay.set_smp_state(smp.enabled, smp.available, self._t("smp_unavailable"))

    def _on_smp_state_changed(self, enabled: bool, simulating: bool, available: bool) -> None:
        self.viewport.cbpc_overlay.set_smp_state(enabled, available, self._t("smp_unavailable"))

    def _on_viewport_animation_stopped(self) -> None:
        self._active_hkx_path = None
        if hasattr(self, "anim_toolbar"):
            self.anim_toolbar.setVisible(False)

    # ------------------------------------------------------------------
    # NIF Load / Detach & Wardrobe Management
    # ------------------------------------------------------------------

    def _load_and_attach_outfit_nif(self, nif_key: str, show_error: bool = True) -> bool:
        try:
            owner = self.mod_scanner.owner_for_nif(nif_key) if self.mod_scanner else None
            if owner is not None:
                self.resolver.register_nif_owner(nif_key, owner)
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
                    self._t("nif_load_error_message", filename=Path(nif_key).name, error=exc),
                )
            return False

    def _detach_outfit_nif(self, nif_key: str, remove_from_presets: bool = True) -> None:
        if (
            hasattr(self, "texture_popover")
            and self.texture_popover.isVisible()
            and self.texture_popover.current_nif_key == nif_key
        ):
            self.texture_popover.close()
        was_locked = remove_from_presets and any(
            nif_key in paths for paths in self._locked_presets.values()
        )
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
        if remove_from_presets:
            for page, paths in self._locked_presets.items():
                paths.discard(nif_key)
                self._locked_preset_diffuse_textures.get(page, {}).pop(nif_key, None)
        if was_locked:
            self.save_config()
        self._update_added_nif_count()

    def uncheck_all_nifs(self) -> None:
        """Remove unlocked outfit meshes while preserving the locked wardrobe."""
        current_preset = self._locked_presets.get(self._current_locked_preset, set())
        for nif_key in tuple(self._checked_nif_paths - current_preset):
            self._detach_outfit_nif(nif_key, remove_from_presets=False)
        self._refresh_added_nifs()

    def _mod_name_for_nif(self, nif_key: str) -> str:
        return get_mod_name_for_nif(nif_key, self.mod_scanner, self.current_root_dir)

    def _add_nif_to_wardrobe_list(self, nif_key: str) -> None:
        locked = nif_key in self._locked_presets.get(self._current_locked_preset, set())
        target_list = self.locked_nifs_list if locked else self.added_nifs_list
        target_items = self._locked_nif_items if locked else self._added_nif_items
        if nif_key in target_items:
            return

        item = QListWidgetItem()
        item.setData(ROLE_ADDED_NIF_PATH, nif_key)
        item.setToolTip(nif_key)

        nif_name = Path(nif_key).name
        if self.mod_scanner and self.mod_scanner.is_overwrite_nif(nif_key):
            nif_name = f"[{nif_name}]"
        display_text = f"{nif_name} ({self._mod_name_for_nif(nif_key)})"

        row_widget = AddedNifRowWidget(
            nif_path=nif_key,
            text=display_text,
            on_select=lambda k=nif_key: self._select_nif_in_tree(k),
            on_toggle_visibility=lambda _checked=False, k=nif_key: self._toggle_nif_visibility(k),
            on_double_click=lambda k=nif_key: (
                self._unlock_nif(k) if locked
                else self._detach_outfit_nif(k, remove_from_presets=False)
            ),
            on_texture_select=lambda anchor, k=nif_key: self._open_nif_texture_popover(k, anchor),
            on_right_click=None if locked else lambda k=nif_key: self._lock_nif(k),
            texture_tooltip=self._t("tex_change_tip"),
            visible=self._nif_visibility.get(nif_key, True),
            hide_tooltip=self._t("hide_nif"),
            show_tooltip=self._t("show_nif"),
        )
        row_widget.setToolTip(
            self._t("double_click_unlock") if locked else self._t("wardrobe_actions")
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
        self.added_nifs_group.setTitle(self._t("wardrobe", count=len(self._added_nif_items)))
        self.locked_nifs_group.setTitle(
            self._t(
                "locked_wardrobe",
                count=len(self._locked_nif_items),
                page=self._current_locked_preset,
            )
        )
        locked_row_height = sum(
            max(
                self.locked_nifs_list.item(index).sizeHint().height(),
                self.locked_nifs_list.sizeHintForRow(index),
            )
            for index in range(self.locked_nifs_list.count())
        )
        locked_list_height = max(36, locked_row_height + 2 * self.locked_nifs_list.frameWidth() + 2)
        self.locked_nifs_list.setFixedHeight(locked_list_height)
        self.locked_nifs_list.setVisible(True)

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    def _refresh_locked_preset_selector(self) -> None:
        pages = sorted(self._locked_presets)
        if not pages:
            pages = [1]
            self._locked_presets[1] = set()
        if self._current_locked_preset not in self._locked_presets:
            self._current_locked_preset = pages[0]

        with QSignalBlocker(self.locked_preset_combo):
            self.locked_preset_combo.clear()
            for page in pages:
                self.locked_preset_combo.addItem(
                    self._locked_preset_display_name(page),
                    page,
                )
            index = self.locked_preset_combo.findData(self._current_locked_preset)
            self.locked_preset_combo.setCurrentIndex(max(0, index))

    def _locked_preset_display_name(self, page: int) -> str:
        name = self._locked_preset_names.get(
            page,
            self._t("locked_preset_default_name", page=page),
        )
        return f"{page}: {name}"

    def _on_locked_preset_changed(self, index: int) -> None:
        if index < 0:
            return
        page = self.locked_preset_combo.itemData(index)
        if not isinstance(page, int) or page == self._current_locked_preset:
            return
        self._activate_locked_preset(page)

    def _activate_locked_preset(self, page: int) -> None:
        previous_page = self._current_locked_preset
        previous_paths = self._locked_presets.get(previous_page, set()).copy()
        next_paths = self._locked_presets[page]
        self._current_locked_preset = page
        for nif_key in sorted(previous_paths - next_paths):
            if nif_key in self._checked_nif_paths:
                self._detach_outfit_nif(nif_key, remove_from_presets=False)
        self._load_current_preset()
        self._refresh_added_nifs()
        self.save_config()

    def _load_current_preset(self) -> None:
        paths = self._locked_presets.get(self._current_locked_preset, set())
        for nif_key in sorted(paths - self._checked_nif_paths):
            self._load_and_attach_outfit_nif(nif_key)
        for nif_key in sorted(paths & self._checked_nif_paths):
            self._restore_preset_diffuse_textures(nif_key)

    def _rename_locked_preset(self) -> None:
        page = self._current_locked_preset
        current_name = self._locked_preset_names.get(
            page,
            self._t("locked_preset_default_name", page=page),
        )
        new_name, accepted = QInputDialog.getText(
            self,
            self._t("rename_locked_preset"),
            self._t("rename_locked_preset_prompt"),
            text=current_name,
        )
        new_name = new_name.strip()
        if not accepted or not new_name:
            return
        self._locked_preset_names[page] = new_name
        self._refresh_locked_preset_selector()
        self.save_config()

    def _add_locked_preset(self) -> None:
        page = 1
        while page in self._locked_presets:
            page += 1
        self._locked_presets[page] = set()
        self._activate_locked_preset(page)
        self._refresh_locked_preset_selector()

    def _on_added_nif_clicked(self, item: QListWidgetItem) -> None:
        nif_key = item.data(ROLE_ADDED_NIF_PATH)
        if nif_key:
            self._select_nif_in_tree(nif_key)

    # ------------------------------------------------------------------
    # Texture Popover Integration
    # ------------------------------------------------------------------

    def _open_nif_texture_popover(self, nif_key: str, anchor_widget: QWidget) -> None:
        nif_data = self._loaded_nif_cache.get(nif_key)
        if nif_data is None:
            try:
                nif_data = self.loader.load_nif(nif_key)
                self._loaded_nif_cache[nif_key] = nif_data
            except Exception as exc:
                log.error("Could not inspect NIF %s for textures: %s", nif_key, exc)
                return

        mod_textures = discover_mod_textures_for_nif(
            nif_key, self.mod_scanner, self.current_root_dir
        )
        self.texture_popover.show_for_nif(
            nif_key=nif_key,
            nif_data=nif_data,
            mod_textures=mod_textures,
            anchor_widget=anchor_widget,
        )

    def _on_apply_nif_texture(
        self,
        nif_key: str,
        shape_indices: List[int],
        slot: str,
        dds_path: Path,
    ) -> None:
        nif_data = self._loaded_nif_cache.get(nif_key)
        if nif_data is None or not shape_indices:
            return

        multi_shape = len(shape_indices) > 1
        target_dir = dds_path.resolve().parent
        applied_diffuse: Dict[str, str] = {}

        for sh_idx in shape_indices:
            if not (0 <= sh_idx < len(nif_data.shapes)):
                continue
            sh = nif_data.shapes[sh_idx]
            chosen_path = (
                resolve_shape_texture_path(sh, slot, dds_path, target_dir)
                if multi_shape
                else dds_path
            )
            try:
                res_tex = self.resolver.load_from_file(slot, chosen_path)
            except Exception as exc:
                log.warning("Could not decode texture %s: %s", chosen_path, exc)
                continue
            self.viewport.update_nif_shape_texture(nif_key, sh_idx, slot, res_tex)
            if slot == "Diffuse" and res_tex.resolved_path is not None:
                applied_diffuse[str(sh_idx)] = str(res_tex.resolved_path.resolve())
        if applied_diffuse and nif_key in self._locked_presets.get(self._current_locked_preset, set()):
            meshes = self._locked_preset_diffuse_textures.setdefault(self._current_locked_preset, {})
            meshes.setdefault(nif_key, {}).update(applied_diffuse)
            self.save_config()

    def _on_reset_nif_textures(self, nif_key: str) -> None:
        nif_data = self._loaded_nif_cache.get(nif_key)
        self.viewport.reset_nif_textures(nif_key, nif_data)
        if nif_key in self._locked_presets.get(self._current_locked_preset, set()):
            self._capture_preset_diffuse_textures(nif_key)
            self.save_config()

    def _capture_preset_diffuse_textures(self, nif_key: str) -> None:
        nif_data = self._loaded_nif_cache.get(nif_key)
        if nif_data is None:
            return
        shapes = {}
        for index, shape in enumerate(nif_data.shapes):
            texture = shape.shader.resolved_textures.get("Diffuse")
            if texture is not None and not texture.is_fallback and texture.resolved_path is not None:
                shapes[str(index)] = str(texture.resolved_path.resolve())
        meshes = self._locked_preset_diffuse_textures.setdefault(self._current_locked_preset, {})
        if shapes:
            meshes[nif_key] = shapes
        else:
            meshes.pop(nif_key, None)

    def _restore_preset_diffuse_textures(self, nif_key: str) -> None:
        nif_data = self._loaded_nif_cache.get(nif_key)
        if nif_data is None:
            return
        meshes = self._locked_preset_diffuse_textures.get(self._current_locked_preset, {})
        if nif_key not in meshes:
            for index, shape in enumerate(nif_data.shapes):
                default = shape.shader.default_resolved_textures.get("Diffuse")
                if default is None:
                    default = self.resolver.load_or_fallback("Diffuse", "", nif_key)
                self.viewport.update_nif_shape_texture(nif_key, index, "Diffuse", default)
            self._capture_preset_diffuse_textures(nif_key)
            return
        saved = meshes[nif_key]
        for index, shape in enumerate(nif_data.shapes):
            path = saved.get(str(index))
            texture = shape.shader.default_resolved_textures.get("Diffuse")
            if path:
                try:
                    texture = self.resolver.load_from_file("Diffuse", path)
                except (OSError, ValueError) as exc:
                    log.warning("Could not restore preset diffuse texture %s for %s: %s", path, nif_key, exc)
            if texture is None:
                texture = self.resolver.load_or_fallback("Diffuse", "", nif_key)
            self.viewport.update_nif_shape_texture(nif_key, index, "Diffuse", texture)
        for index in saved:
            if int(index) >= len(nif_data.shapes):
                log.warning("Preset diffuse texture shape %s no longer exists in %s", index, nif_key)

    # ------------------------------------------------------------------
    # Drag-and-Drop and Lock Management
    # ------------------------------------------------------------------

    def _autoload_locked_nifs(self) -> None:
        """Load NIFs saved in presets without showing startup error dialogs."""
        missing_nifs = {
            nif_key
            for paths in self._locked_presets.values()
            for nif_key in paths
            if not Path(nif_key).is_file()
        }
        if missing_nifs:
            for nif_key in sorted(missing_nifs):
                log.warning("Removing unavailable locked NIF from config: %s", nif_key)
            for paths in self._locked_presets.values():
                paths.difference_update(missing_nifs)
            for meshes in self._locked_preset_diffuse_textures.values():
                for nif_key in missing_nifs:
                    meshes.pop(nif_key, None)
            self.save_config()

        for nif_key in sorted(self._locked_presets.get(self._current_locked_preset, set())):
            if self._load_and_attach_outfit_nif(nif_key, show_error=False):
                self._restore_preset_diffuse_textures(nif_key)
        self._startup_locked_nif_refresh_pending = bool(self._locked_nif_items)

    def _lock_nif(self, nif_key: str) -> None:
        if (
            nif_key not in self._checked_nif_paths
            or nif_key in self._locked_presets.get(self._current_locked_preset, set())
        ):
            return
        self._locked_presets.setdefault(self._current_locked_preset, set()).add(nif_key)
        self._capture_preset_diffuse_textures(nif_key)
        self.save_config()
        self._refresh_added_nifs()

    def _on_nif_dropped_on_preset(self, nif_key: str) -> None:
        if Path(nif_key).suffix.casefold() != ".nif":
            return
        if nif_key not in self._checked_nif_paths:
            if not self._load_and_attach_outfit_nif(nif_key):
                return
        self._lock_nif(nif_key)

    def _on_nif_dropped_on_wardrobe(self, nif_key: str) -> None:
        if nif_key in self._locked_presets.get(self._current_locked_preset, set()):
            self._unlock_nif(nif_key)
        elif Path(nif_key).suffix.casefold() == ".nif" and nif_key not in self._checked_nif_paths:
            self._load_and_attach_outfit_nif(nif_key)

    def _on_nif_dropped_on_mods(self, nif_key: str) -> None:
        if nif_key in self._checked_nif_paths:
            self._detach_outfit_nif(nif_key)

    def _unlock_nif(self, nif_key: str) -> None:
        page_paths = self._locked_presets.get(self._current_locked_preset)
        if page_paths is None or nif_key not in page_paths:
            return
        page_paths.discard(nif_key)
        self._locked_preset_diffuse_textures.get(self._current_locked_preset, {}).pop(nif_key, None)
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

    # ------------------------------------------------------------------
    # Tree Navigation and Highlighting
    # ------------------------------------------------------------------

    def _tree_siblings(self, parent_item: Optional[QTreeWidgetItem]) -> List[QTreeWidgetItem]:
        if parent_item is not None:
            return [parent_item.child(i) for i in range(parent_item.childCount())]
        return [
            self.file_tree.topLevelItem(i)
            for i in range(self.file_tree.topLevelItemCount())
        ]

    def _find_tree_item(
        self, parent_item: Optional[QTreeWidgetItem], predicate
    ) -> Optional[QTreeWidgetItem]:
        return next((it for it in self._tree_siblings(parent_item) if predicate(it)), None)

    def _expand_path_and_select(
        self,
        start_item: Optional[QTreeWidgetItem],
        path_segments: tuple[str, ...],
        resolved_nif: Path,
        rel_path_role: int,
    ) -> None:
        if start_item is None:
            return
        parent_item = start_item
        accumulated: tuple[str, ...] = ()
        for folder_name in path_segments:
            self._on_tree_item_expanded(parent_item)
            accumulated += (folder_name,)
            parent_item = self._find_tree_item(
                parent_item,
                lambda it: tuple(
                    str(part).casefold() for part in (it.data(0, rel_path_role) or ())
                ) == tuple(part.casefold() for part in accumulated),
            )
            if parent_item is None:
                return
            parent_item.setExpanded(True)

        self._on_tree_item_expanded(parent_item)
        target = self._find_tree_item(
            parent_item,
            lambda it: bool(it.data(0, ROLE_NIF_PATH))
            and Path(it.data(0, ROLE_NIF_PATH)).resolve() == resolved_nif,
        )
        if target:
            self._highlight_tree_item(target)

    def _select_nif_in_tree(self, nif_key: str) -> None:
        if self.current_root_dir is None:
            return

        resolved_nif = Path(nif_key).resolve()
        overlay_target = (
            self.mod_scanner.target_for_nif(nif_key) if self.mod_scanner else None
        )
        fallback_target = (
            self.mod_scanner.fallback_target_for_nif(nif_key) if self.mod_scanner else None
        )

        if self.search_edit.text():
            self.search_edit.clear()

        if overlay_target is not None:
            mod_root, target_parts = overlay_target
            start_item = self._find_tree_item(
                None,
                lambda it: bool(it.data(0, ROLE_MOD_ROOT_PATH))
                and Path(it.data(0, ROLE_MOD_ROOT_PATH)).resolve() == mod_root,
            )
            self._expand_path_and_select(
                start_item, target_parts[:-1], resolved_nif, ROLE_MOD_RELATIVE_PATH
            )
        elif fallback_target is not None:
            start_item = self._find_tree_item(
                None,
                lambda it: bool(it.data(0, ROLE_OVERWRITE_FALLBACK)),
            )
            self._expand_path_and_select(
                start_item, fallback_target[:-1], resolved_nif, ROLE_OVERWRITE_RELATIVE_PATH
            )
        else:
            try:
                relative_path = resolved_nif.relative_to(self.current_root_dir.resolve())
            except ValueError:
                return

            parent_item: Optional[QTreeWidgetItem] = None
            current_path = self.current_root_dir
            for folder_name in relative_path.parts[:-1]:
                current_path /= folder_name
                parent_item = self._find_tree_item(
                    parent_item,
                    lambda it: bool(it.data(0, ROLE_FOLDER_PATH))
                    and Path(it.data(0, ROLE_FOLDER_PATH)).resolve() == current_path.resolve(),
                )
                if parent_item is None:
                    return
                self._on_tree_item_expanded(parent_item)
                parent_item.setExpanded(True)

            leaf = self._find_tree_item(
                parent_item,
                lambda it: bool(it.data(0, ROLE_NIF_PATH))
                and Path(it.data(0, ROLE_NIF_PATH)).resolve() == resolved_nif,
            )
            if leaf:
                self._highlight_tree_item(leaf)

    def _highlight_tree_item(self, target_item: QTreeWidgetItem) -> None:
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
    # Lighting, Shader & Viewport Effects
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

    def _on_effects_changed(self) -> None:
        settings = self.viewport.settings
        settings.enable_ssr = self.chk_ssr.isChecked()
        settings.ssr_step_size = self.slider_ssr_step.value() / 10.0
        settings.ssr_max_steps = self.slider_ssr_steps.value()
        settings.ssr_thickness = self.slider_ssr_thickness.value() / 10.0
        settings.enable_ao = self.chk_ao.isChecked()
        settings.ao_radius = float(self.slider_ao_radius.value())
        settings.ao_intensity = self.slider_ao_intensity.value() / 100.0
        settings.ao_bias = self.slider_ao_bias.value() / 1000.0
        settings.enable_ssss = self.chk_ssss.isChecked()
        settings.ssss_radius = float(self.slider_ssss_radius.value())

        self.lbl_ssr_step.setText(f"{settings.ssr_step_size:.1f}")
        self.lbl_ssr_steps.setText(str(settings.ssr_max_steps))
        self.lbl_ssr_thickness.setText(f"{settings.ssr_thickness:.1f}")
        self.lbl_ao_radius.setText(f"{settings.ao_radius:.0f}")
        self.lbl_ao_intensity.setText(f"{settings.ao_intensity:.2f}")
        self.lbl_ao_bias.setText(f"{settings.ao_bias:.3f}")
        self.lbl_ssss_radius.setText(f"{settings.ssss_radius:.0f}")

        for control in (self.slider_ssr_step, self.slider_ssr_steps, self.slider_ssr_thickness):
            control.setEnabled(settings.enable_ssr)
        for control in (self.slider_ao_radius, self.slider_ao_intensity, self.slider_ao_bias):
            control.setEnabled(settings.enable_ao)
        self.slider_ssss_radius.setEnabled(settings.enable_ssss)
        self.btn_ssss_falloff.setEnabled(settings.enable_ssss)
        self.viewport.update()

    def _reset_ssr_controls(self) -> None:
        defaults = ViewportEffectsConfig()
        self.chk_ssr.setChecked(defaults.enable_ssr)
        self.slider_ssr_step.setValue(int(round(defaults.ssr_step_size * 10)))
        self.slider_ssr_steps.setValue(defaults.ssr_max_steps)
        self.slider_ssr_thickness.setValue(int(round(defaults.ssr_thickness * 10)))
        self._on_effects_changed()

    def _reset_ao_controls(self) -> None:
        defaults = ViewportEffectsConfig()
        self.chk_ao.setChecked(defaults.enable_ao)
        self.slider_ao_radius.setValue(int(round(defaults.ao_radius)))
        self.slider_ao_intensity.setValue(int(round(defaults.ao_intensity * 100)))
        self.slider_ao_bias.setValue(int(round(defaults.ao_bias * 1000)))
        self._on_effects_changed()

    def _reset_ssss_controls(self) -> None:
        defaults = ViewportEffectsConfig()
        self.chk_ssss.setChecked(defaults.enable_ssss)
        self.slider_ssss_radius.setValue(int(round(defaults.ssss_radius)))
        self.viewport.settings.ssss_falloff_color = defaults.ssss_falloff_color
        self._update_ssss_falloff_button()
        self._on_effects_changed()

    def _update_ssss_falloff_button(self) -> None:
        color = QColor.fromRgbF(*self.viewport.settings.ssss_falloff_color)
        self.btn_ssss_falloff.setText(color.name().upper())
        self.btn_ssss_falloff.setStyleSheet(
            f"background-color: {color.name()}; color: #14161c;"
        )

    def _choose_ssss_falloff_color(self) -> None:
        initial = QColor.fromRgbF(*self.viewport.settings.ssss_falloff_color)
        color = QColorDialog.getColor(initial, self, self._t("ssss_falloff_color"))
        if not color.isValid():
            return
        self.viewport.settings.ssss_falloff_color = (
            color.redF(),
            color.greenF(),
            color.blueF(),
        )
        self._update_ssss_falloff_button()
        self.viewport.update()

    # ------------------------------------------------------------------
    # Config Persistence
    # ------------------------------------------------------------------

    def save_config(self, path: Optional[Path | str] = None) -> None:
        target_path = path if path else self.config_path
        self.config.default_folder = (
            str(self.current_root_dir) if self.current_root_dir else ""
        )
        self.config.language = self.language

        light = self.config.lighting
        light.light_azimuth_deg = float(self.slider_light_azimuth.value())
        light.light_elevation_deg = float(self.slider_light_elevation.value())
        light.light_intensity = self.slider_key_intensity.value() / 100.0
        light.ambient_intensity = self.slider_ambient.value() / 100.0
        light.specular_multiplier = self.slider_specular.value() / 100.0
        light.env_reflection_multiplier = self.slider_env.value() / 100.0

        eff = self.config.effects
        eff.enable_ssr = self.chk_ssr.isChecked()
        eff.ssr_step_size = self.slider_ssr_step.value() / 10.0
        eff.ssr_max_steps = self.slider_ssr_steps.value()
        eff.ssr_thickness = self.slider_ssr_thickness.value() / 10.0
        eff.enable_ao = self.chk_ao.isChecked()
        eff.ao_radius = float(self.slider_ao_radius.value())
        eff.ao_intensity = self.slider_ao_intensity.value() / 100.0
        eff.ao_bias = self.slider_ao_bias.value() / 1000.0
        eff.enable_ssss = self.chk_ssss.isChecked()
        eff.ssss_radius = float(self.slider_ssss_radius.value())
        eff.ssss_falloff_color = self.viewport.settings.ssss_falloff_color

        c = self.config.checkboxes
        c.meshes_and_caliente_only = self.chk_meshes_and_caliente_only.isChecked()
        c.hide_low_weight = self.chk_hide_low_weight.isChecked()
        c.hide_gnd_folder = self.chk_hide_gnd_folder.isChecked()
        c.show_mod_dir_filter = self.dir_group.isVisible()
        c.wireframe_overlay = self.chk_wireframe.isChecked()
        c.show_ground_grid = self.chk_grid.isChecked()

        cb = self.config.cbpc
        cb.enabled = self.viewport.cbpc.enabled
        cb.gravity = self.viewport.cbpc.gravity
        cb.softness = self.viewport.cbpc.softness.copy()
        self.config.smp.enabled = self.viewport.smp.enabled

        self.config.locked_presets = {
            str(page): sorted(paths)
            for page, paths in self._locked_presets.items()
        }
        self.config.locked_preset_diffuse_textures = {
            str(page): {
                nif_key: shapes.copy()
                for nif_key, shapes in meshes.items()
                if nif_key in self._locked_presets.get(page, set()) and shapes
            }
            for page, meshes in self._locked_preset_diffuse_textures.items()
            if page in self._locked_presets and meshes
        }
        self.config.locked_preset_names = {
            str(page): name.strip()
            for page, name in self._locked_preset_names.items()
            if name.strip() and page in self._locked_presets
        }
        self.config.save(target_path)
