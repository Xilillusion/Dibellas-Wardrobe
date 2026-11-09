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

from PySide6.QtCore import QModelIndex, QPoint, QRect, QSignalBlocker, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QBrush, QGuiApplication, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QColorDialog,
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
    QStyle,
    QStyleOptionGroupBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .config import AppConfig, LightingConfig, ViewportEffectsConfig, load_locales
from .mod_scanner import ModScanner, is_animation_hkx
from .nif_loader import LoadedNifData, NifLoader
from .texture_resolver import TextureResolver
from .viewport import OutfitViewportWidget

log = logging.getLogger("dibellas_wardrobe.ui")

ROLE_NIF_PATH = Qt.ItemDataRole.UserRole + 1
ROLE_FOLDER_PATH = Qt.ItemDataRole.UserRole + 2
ROLE_IS_LOADED = Qt.ItemDataRole.UserRole + 3
ROLE_ADDED_NIF_PATH = Qt.ItemDataRole.UserRole
ROLE_MOD_ROOT_PATH = Qt.ItemDataRole.UserRole + 4
ROLE_MOD_RELATIVE_PATH = Qt.ItemDataRole.UserRole + 5
ROLE_OVERWRITE_FALLBACK = Qt.ItemDataRole.UserRole + 6
ROLE_OVERWRITE_RELATIVE_PATH = Qt.ItemDataRole.UserRole + 7
ROLE_HKX_PATH = Qt.ItemDataRole.UserRole + 8

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


def normalize_mods_root(folder: Path) -> Path:
    """Use MO2's mods subdirectory when the user selected the MO2 instance root."""
    if (
        folder.name.casefold() != "mods"
        and (folder / "mods").is_dir()
        and (folder / "overwrite").is_dir()
    ):
        return folder / "mods"
    return folder


class NifSearchThread(QThread):
    """
    Background worker that recursively searches `root_dir` for .nif files
    matching the search query without blocking the PySide6 UI thread.
    """

    matchFound = Signal(str, str, str, str)  # (mod name, relative directory, file name, path)

    def __init__(
        self,
        root_dir: Path,
        query: str,
        hide_0: bool,
        hide_gnd_folder: bool = True,
        meshes_and_caliente_only: bool = True,
        mod_scanner: Optional[ModScanner] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.root_dir = root_dir
        self.query = query.lower()
        self.hide_0 = hide_0
        self.hide_gnd_folder = hide_gnd_folder
        self.meshes_and_caliente_only = meshes_and_caliente_only
        self.mod_scanner = mod_scanner
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
                    lower_fname = fname.casefold()
                    if lower_fname.endswith(".nif"):
                        if is_filtered_nif_filename(fname, self.hide_0):
                            continue
                    elif is_animation_hkx(fname, rel_parts if "rel_parts" in locals() else ()):
                        pass
                    else:
                        continue
                    if self.mod_scanner:
                        try:
                            relative_parts = root_path.relative_to(
                                self.root_dir
                            ).parts
                            if relative_parts:
                                mod_path = self.root_dir / relative_parts[0]
                                if self.mod_scanner.is_overridden_mod_file(
                                    mod_path, relative_parts[1:] + (fname,)
                                ):
                                    continue
                        except ValueError:
                            pass

                    full_path = str((root_path / fname).resolve())
                    relative_dirs = (
                        rel_parts[1:]
                        if self.mod_scanner and rel_parts
                        else rel_parts
                    )
                    self.matchFound.emit(
                        group_name,
                        str(Path(*relative_dirs)) if relative_dirs else "",
                        fname,
                        full_path,
                    )
            if self.mod_scanner:
                for group_name, file_name, nif_path in self.mod_scanner.search_entries():
                    if self._cancelled:
                        return
                    if self.query not in group_name.casefold():
                        continue
                    if nif_path.name.casefold().endswith(".nif") and is_filtered_nif_filename(
                        nif_path.name, self.hide_0
                    ):
                        continue
                    target = self.mod_scanner.target_for_nif(nif_path)
                    relative_parts = (
                        target[1]
                        if target
                        else self.mod_scanner.fallback_target_for_nif(nif_path)
                    )
                    if self.hide_gnd_folder and relative_parts and any(
                        is_filtered_folder_name(part, True)
                        for part in relative_parts[:-1]
                    ):
                        continue
                    relative_dirs = target[1][:-1] if target else (
                        relative_parts[:-1] if relative_parts else ()
                    )
                    self.matchFound.emit(
                        group_name,
                        str(Path(*relative_dirs)) if relative_dirs else "",
                        file_name,
                        str(nif_path),
                    )
        except Exception as exc:
            log.warning("Search error: %s", exc)


class RefreshableGroupBox(QGroupBox):
    """Group box with boxed refresh and optional minimize header actions."""

    def __init__(
        self,
        title: str,
        on_refresh=None,
        minimizable: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(title, parent)
        self._collapsible_content: Optional[QWidget] = None
        self._collapsed = False

        self.header_actions = QFrame(self)
        self.header_actions.setObjectName("headerActions")
        self.header_actions.setStyleSheet(
            """
            QFrame#headerActions {
                background-color: #1b1d24;
                border: 1px solid #41485f;
                border-radius: 4px;
            }
            QPushButton#headerActionButton {
                background: transparent;
                border: none;
                border-radius: 3px;
                color: #9da6be;
                font-size: 14px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton#headerActionButton:hover {
                background-color: #2b3040;
                color: #f0d59c;
            }
            QPushButton#headerActionButton:pressed {
                background-color: #1f222e;
                color: #dfb86c;
            }
            """
        )
        actions_layout = QHBoxLayout(self.header_actions)
        actions_layout.setContentsMargins(1, 1, 1, 1)
        actions_layout.setSpacing(2)

        self.btn_refresh: Optional[QPushButton] = None
        if on_refresh is not None:
            self.btn_refresh = QPushButton("↻", self.header_actions)
            self.btn_refresh.setObjectName("headerActionButton")
            self.btn_refresh.setToolTip("Refresh")
            self.btn_refresh.setFixedSize(22, 22)
            self.btn_refresh.clicked.connect(on_refresh)
            actions_layout.addWidget(self.btn_refresh)

        self.btn_minimize: Optional[QPushButton] = None
        if minimizable:
            self.btn_minimize = QPushButton("−", self.header_actions)
            self.btn_minimize.setObjectName("headerActionButton")
            self.btn_minimize.setToolTip("Minimize")
            self.btn_minimize.setFixedSize(22, 22)
            self.btn_minimize.clicked.connect(self._toggle_content)
            actions_layout.addWidget(self.btn_minimize)

        self.header_actions.adjustSize()
        self.header_actions.setVisible(on_refresh is not None or minimizable)

    def set_collapsible_content(self, content: QWidget) -> None:
        self._collapsible_content = content

    def setTitle(self, title: str) -> None:
        super().setTitle(title)
        if hasattr(self, "header_actions"):
            QTimer.singleShot(0, self._position_header_actions)

    def set_collapsed(self, collapsed: bool) -> None:
        if self._collapsible_content is None or self.btn_minimize is None:
            return
        self._collapsed = collapsed
        self._collapsible_content.setVisible(not collapsed)
        self.btn_minimize.setText("+" if collapsed else "−")
        self._update_minimize_tooltip()

    def set_minimize_tooltips(self, minimize: str, restore: str) -> None:
        self._minimize_tooltip = minimize
        self._restore_tooltip = restore
        self._update_minimize_tooltip()

    def _toggle_content(self) -> None:
        if self._collapsible_content is None or self.btn_minimize is None:
            return
        self.set_collapsed(not self._collapsed)

    def _update_minimize_tooltip(self) -> None:
        if self.btn_minimize is None:
            return
        self.btn_minimize.setToolTip(
            getattr(self, "_restore_tooltip", "Restore")
            if self._collapsed
            else getattr(self, "_minimize_tooltip", "Minimize")
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_header_actions()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._position_header_actions)

    def _position_header_actions(self) -> None:
        if self.header_actions.isVisible():
            option = QStyleOptionGroupBox()
            self.initStyleOption(option)
            title_rect = self.style().subControlRect(
                QStyle.ComplexControl.CC_GroupBox,
                option,
                QStyle.SubControl.SC_GroupBoxLabel,
                self,
            )
            self.header_actions.move(
                max(0, self.width() - self.header_actions.width() - 8),
                max(0, title_rect.center().y() - self.header_actions.height() // 2),
            )


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
        on_right_click=None,
        visible: bool = True,
        remove_tooltip: str = "Remove this NIF",
        hide_tooltip: str = "Hide this NIF",
        show_tooltip: str = "Show this NIF",
        show_remove: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._on_select = on_select
        self._on_double_click = on_double_click
        self._on_right_click = on_right_click
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

        if show_remove:
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
        elif event.button() == Qt.MouseButton.RightButton and self._on_right_click is not None:
            self._on_right_click(self)
            event.accept()

    def contextMenuEvent(self, event) -> None:
        if self._on_right_click is not None:
            self._on_right_click(self)
            event.accept()
            return
        super().contextMenuEvent(event)

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
        is_leaf_file = bool(
            item.data(0, ROLE_NIF_PATH) or item.data(0, ROLE_HKX_PATH)
        )
        has_children = item.childCount() > 0 or not is_leaf_file

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


class AnimationPlaybackToolbar(QFrame):
    """
    Docked bottom playback toolbar for `.hkx` animations.

    Displays metadata (`[Mod Name (Parent Directory)] - [Filename.hkx]`), a formatted
    timestamp counter (`MM:SS.CC / MM:SS.CC`), an interactive timeline scrubber, and
    Play/Pause, Loop, and Close/Reset controls.
    """

    SCRUB_STEPS = 1000

    def __init__(
        self,
        on_toggle_play,
        on_toggle_loop,
        on_seek,
        on_close,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._on_seek = on_seek
        self._duration: float = 0.0
        self._is_playing: bool = False
        self._play_text = "▶"
        self._pause_text = "Ⅱ"
        self._play_tooltip = "Play"
        self._pause_tooltip = "Pause"

        self.setObjectName("animPlaybackToolbar")
        self.setStyleSheet(
            """
            QFrame#animPlaybackToolbar {
                background-color: #161820;
                border: 1px solid #30364a;
                border-radius: 6px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(4)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(8)

        self.lbl_metadata = QLabel("")
        self.lbl_metadata.setStyleSheet(
            "color: #f0d59c; font-weight: 600; font-size: 12px;"
        )
        self.lbl_metadata.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )

        self.lbl_time = QLabel("00:00.00 / 00:00.00")
        self.lbl_time.setStyleSheet(
            "color: #9da6be; font-family: 'Consolas', monospace; font-size: 12px;"
        )
        self.lbl_time.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        top_row.addWidget(self.lbl_metadata, stretch=1)
        top_row.addWidget(self.lbl_time)
        layout.addLayout(top_row)

        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)
        bottom_row.setSpacing(8)

        self.btn_play_pause = QPushButton(self._pause_text)
        self.btn_play_pause.setFixedSize(32, 30)
        self.btn_play_pause.clicked.connect(on_toggle_play)

        self.slider_timeline = QSlider(Qt.Orientation.Horizontal)
        self.slider_timeline.setRange(0, self.SCRUB_STEPS)
        self.slider_timeline.setValue(0)
        self.slider_timeline.valueChanged.connect(self._on_slider_value_changed)

        self.chk_loop = QCheckBox("Loop")
        self.chk_loop.setChecked(True)
        self.chk_loop.toggled.connect(on_toggle_loop)

        self.btn_close = QPushButton("×")
        self.btn_close.setFixedSize(32, 30)
        self.btn_close.clicked.connect(on_close)

        bottom_row.addWidget(self.btn_play_pause)
        bottom_row.addWidget(self.slider_timeline, stretch=1)
        bottom_row.addWidget(self.chk_loop)
        bottom_row.addWidget(self.btn_close)
        layout.addLayout(bottom_row)

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        mins = int(seconds // 60.0)
        rem = seconds - mins * 60.0
        return f"{mins:02d}:{rem:05.2f}"

    def set_translations(
        self,
        play_text: str,
        pause_text: str,
        loop_text: str,
        close_text: str,
        close_tooltip: str,
    ) -> None:
        self._play_text = "▶"
        self._pause_text = "Ⅱ"
        self._play_tooltip = play_text
        self._pause_tooltip = pause_text
        self.btn_play_pause.setText(
            self._pause_text if self._is_playing else self._play_text
        )
        self.btn_play_pause.setToolTip(
            self._pause_tooltip if self._is_playing else self._play_tooltip
        )
        self.chk_loop.setText(loop_text)
        self.btn_close.setText("×")
        self.btn_close.setToolTip(close_tooltip or close_text)
        self.btn_close.setAccessibleName(close_text)

    def set_metadata(self, label_text: str, duration: float) -> None:
        self._duration = max(0.0, float(duration))
        self.lbl_metadata.setText(label_text)
        self.lbl_metadata.setToolTip(label_text)
        self.update_time(0.0, self._duration)

    def update_state(self, is_playing: bool, is_looping: bool) -> None:
        self._is_playing = bool(is_playing)
        self.btn_play_pause.setText(
            self._pause_text if self._is_playing else self._play_text
        )
        self.btn_play_pause.setToolTip(
            self._pause_tooltip if self._is_playing else self._play_tooltip
        )
        with QSignalBlocker(self.chk_loop):
            self.chk_loop.setChecked(bool(is_looping))

    def update_time(self, current_time: float, duration: float) -> None:
        self._duration = max(0.0, float(duration))
        cur_str = self.format_timestamp(current_time)
        dur_str = self.format_timestamp(self._duration)
        self.lbl_time.setText(f"{cur_str} / {dur_str}")

        if not self.slider_timeline.isSliderDown():
            with QSignalBlocker(self.slider_timeline):
                if self._duration > 0.0:
                    ratio = max(0.0, min(1.0, current_time / self._duration))
                    self.slider_timeline.setValue(int(round(ratio * self.SCRUB_STEPS)))
                else:
                    self.slider_timeline.setValue(0)

    def _on_slider_value_changed(self, value: int) -> None:
        if self._duration <= 0.0:
            return
        target_time = (float(value) / float(self.SCRUB_STEPS)) * self._duration
        self._on_seek(target_time)


ROLE_TEXTURE_PATH = Qt.ItemDataRole.UserRole + 20
_BODY_SHAPE_TOKENS = (
    "3ba",
    "3bbb",
    "cbbe",
    "bhunp",
    "unp",
    "ube",
    "femalebody",
    "basebody",
    "femaleshands",
    "femalefeet",
)


def _is_likely_body_shape(shape_name: str) -> bool:
    low = shape_name.casefold().strip()
    if low in {"body", "feet", "hands", "head"}:
        return True
    return any(tok in low for tok in _BODY_SHAPE_TOKENS)


def _is_diffuse_texture(filename_lower: str) -> bool:
    stem = filename_lower[:-4] if filename_lower.endswith(".dds") else filename_lower
    return not stem.endswith(("_n", "_msn", "_s", "_sk", "_g", "_m", "_em"))


class NifTexturePopover(QFrame):
    """
    Small floating popover bar anchored to the right of a Wardrobe/Locked NIF item,
    allowing live diffuse texture switching per shape from the mod's textures or disk.
    """

    ALL_OUTFIT_SHAPES = -1

    def __init__(
        self,
        on_apply_texture,
        on_reset_textures,
        translate_fn,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            parent,
            Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint,
        )
        self._on_apply_texture = on_apply_texture
        self._on_reset_textures = on_reset_textures
        self._t = translate_fn

        self.current_nif_key: Optional[str] = None
        self.current_nif_data: Optional[LoadedNifData] = None
        self._mod_textures: List[tuple[str, Path]] = []
        self._anchor_widget: Optional[QWidget] = None

        self.setObjectName("nifTexturePopover")
        self.setFixedWidth(310)
        self.setFixedHeight(340)
        self.setStyleSheet(
            """
            QFrame#nifTexturePopover {
                background-color: #161820;
                border: 1px solid #3d455e;
                border-radius: 6px;
            }
            """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(4)

        self.lbl_title = ElidedLabel("")
        self.lbl_title.setStyleSheet("color: #f0d59c; font-weight: 600; font-size: 12px;")
        header_row.addWidget(self.lbl_title, stretch=1)

        self.btn_reset = QPushButton("↻")
        self.btn_reset.setFixedSize(22, 22)
        self.btn_reset.setStyleSheet(
            """
            QPushButton {
                background-color: #262b38;
                border: 1px solid #41485f;
                border-radius: 4px;
                color: #d8deec;
                font-size: 12px;
                font-weight: 700;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #32394a;
                color: #ffffff;
            }
            """
        )
        self.btn_reset.clicked.connect(self._handle_reset)
        header_row.addWidget(self.btn_reset)

        self.btn_close = QPushButton("×")
        self.btn_close.setFixedSize(22, 22)
        self.btn_close.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: none;
                color: #8f98b0;
                font-weight: bold;
                font-size: 14px;
                padding: 0px;
            }
            QPushButton:hover {
                color: #ff6b6b;
                background-color: #382424;
                border-radius: 3px;
            }
            """
        )
        self.btn_close.clicked.connect(self.close)
        header_row.addWidget(self.btn_close)
        layout.addLayout(header_row)

        selectors_row = QHBoxLayout()
        selectors_row.setContentsMargins(0, 0, 0, 0)
        selectors_row.setSpacing(4)

        self.combo_shape = QComboBox()
        self.combo_shape.setFixedHeight(26)
        self.combo_shape.currentIndexChanged.connect(self._on_selector_changed)
        selectors_row.addWidget(self.combo_shape, stretch=3)

        layout.addLayout(selectors_row)

        self.lbl_current_tex = ElidedLabel("")
        self.lbl_current_tex.setStyleSheet("color: #9da6be; font-size: 11px;")
        layout.addWidget(self.lbl_current_tex)

        self.filter_edit = QLineEdit()
        self.filter_edit.setFixedHeight(24)
        self.filter_edit.textChanged.connect(self._populate_list)
        layout.addWidget(self.filter_edit)

        self.texture_list = QListWidget()
        self.texture_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.texture_list.itemClicked.connect(self._on_texture_item_clicked)
        layout.addWidget(self.texture_list, stretch=1)

        self.btn_browse = QPushButton()
        self.btn_browse.setFixedHeight(26)
        self.btn_browse.clicked.connect(self._handle_browse)
        layout.addWidget(self.btn_browse)

        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        self.btn_reset.setToolTip(self._t("tex_reset_tip"))
        self.btn_close.setToolTip(self._t("tex_close_tip"))
        self.filter_edit.setPlaceholderText(self._t("tex_filter_placeholder"))
        self.btn_browse.setText(self._t("tex_browse_dds"))

        if self.current_nif_data is not None:
            self._populate_shape_combo(preserve_selection=True)
            self._refresh_current_label()
            self._populate_list()

    def selected_shape_index(self) -> int:
        data = self.combo_shape.currentData()
        return int(data) if data is not None else 0

    def selected_slot(self) -> str:
        return "Diffuse"

    def outfit_shape_indices(self) -> List[int]:
        if self.current_nif_data is None:
            return []
        non_body = [
            idx
            for idx, sh in enumerate(self.current_nif_data.shapes)
            if not _is_likely_body_shape(sh.name)
        ]
        return non_body if non_body else list(range(len(self.current_nif_data.shapes)))

    def target_shape_indices(self) -> List[int]:
        if self.current_nif_data is None or not self.current_nif_data.shapes:
            return []
        sel = self.selected_shape_index()
        if sel == self.ALL_OUTFIT_SHAPES:
            return self.outfit_shape_indices()
        if 0 <= sel < len(self.current_nif_data.shapes):
            return [sel]
        return [0]

    def _populate_shape_combo(self, preserve_selection: bool = False) -> None:
        prev = self.combo_shape.currentData() if preserve_selection else None
        with QSignalBlocker(self.combo_shape):
            self.combo_shape.clear()
            if self.current_nif_data is None:
                return
            outfit_indices = self.outfit_shape_indices()
            if len(outfit_indices) >= 2:
                self.combo_shape.addItem(
                    self._t("tex_all_shapes"),
                    self.ALL_OUTFIT_SHAPES,
                )
            for idx, sh in enumerate(self.current_nif_data.shapes):
                self.combo_shape.addItem(f"{idx}: {sh.name}", idx)

            if prev is not None:
                found_idx = self.combo_shape.findData(prev)
                if found_idx >= 0:
                    self.combo_shape.setCurrentIndex(found_idx)
                    return

            default_shape = outfit_indices[0] if outfit_indices else 0
            found_idx = self.combo_shape.findData(default_shape)
            self.combo_shape.setCurrentIndex(max(0, found_idx))

    def show_for_nif(
        self,
        nif_key: str,
        nif_data: LoadedNifData,
        mod_textures: List[tuple[str, Path]],
        anchor_widget: QWidget,
    ) -> None:
        self.current_nif_key = nif_key
        self.current_nif_data = nif_data
        self._mod_textures = list(mod_textures)
        self._anchor_widget = anchor_widget

        self.lbl_title.setText(Path(nif_key).name)
        self.lbl_title.setToolTip(nif_key)
        self._populate_shape_combo(preserve_selection=False)
        with QSignalBlocker(self.filter_edit):
            self.filter_edit.clear()

        self._refresh_current_label()
        self._populate_list()
        self._position_next_to(anchor_widget)
        self.show()
        self.raise_()
        self.activateWindow()

    def _position_next_to(self, anchor_widget: QWidget) -> None:
        global_pos = anchor_widget.mapToGlobal(QPoint(anchor_widget.width() + 6, -4))
        screen = (
            anchor_widget.screen()
            or QGuiApplication.screenAt(global_pos)
            or QGuiApplication.primaryScreen()
        )
        if screen is not None:
            avail = screen.availableGeometry()
            x = min(
                max(avail.left() + 4, global_pos.x()),
                avail.right() - self.width() - 4,
            )
            y = min(
                max(avail.top() + 4, global_pos.y()),
                avail.bottom() - self.height() - 4,
            )
            self.move(x, y)
        else:
            self.move(global_pos)

    def _active_resolved_texture(self):
        if self.current_nif_data is None:
            return None
        indices = self.target_shape_indices()
        if not indices:
            return None
        sh = self.current_nif_data.shapes[indices[0]]
        return sh.shader.resolved_textures.get(self.selected_slot())

    def _refresh_current_label(self) -> None:
        res_tex = self._active_resolved_texture()
        if res_tex is not None and not res_tex.is_fallback and res_tex.resolved_path is not None:
            name = res_tex.resolved_path.name
            tooltip = str(res_tex.resolved_path)
        elif res_tex is not None and res_tex.raw_nif_path:
            name = Path(res_tex.raw_nif_path.replace("\\", "/")).name
            tooltip = res_tex.raw_nif_path
        else:
            name = self._t("tex_missing")
            tooltip = name
        self.lbl_current_tex.setText(self._t("tex_current", name=name))
        self.lbl_current_tex.setToolTip(tooltip)

    def _populate_list(self) -> None:
        self.texture_list.clear()
        res_tex = self._active_resolved_texture()
        active_path = (
            res_tex.resolved_path.resolve()
            if res_tex is not None and not res_tex.is_fallback and res_tex.resolved_path is not None
            else None
        )
        active_parent = active_path.parent if active_path is not None else None
        filter_query = self.filter_edit.text().strip().casefold()

        diffuse_entries = [
            entry
            for entry in self._mod_textures
            if _is_diffuse_texture(entry[1].name.casefold())
        ]
        sorted_entries = sorted(
            diffuse_entries,
            key=lambda entry: (
                0 if active_parent is not None and entry[1].parent == active_parent else 1,
                entry[0].casefold(),
            ),
        )
        selected_item: Optional[QListWidgetItem] = None

        for label, path in sorted_entries:
            if filter_query and filter_query not in label.casefold():
                continue
            item = QListWidgetItem(label)
            item.setData(ROLE_TEXTURE_PATH, str(path))
            item.setToolTip(str(path))
            self.texture_list.addItem(item)
            if active_path is not None and path.resolve() == active_path:
                selected_item = item

        if self.texture_list.count() == 0:
            empty_item = QListWidgetItem(self._t("tex_no_mod_textures"))
            empty_item.setFlags(Qt.ItemFlag.NoItemFlags)
            empty_item.setForeground(QColor("#7e869c"))
            self.texture_list.addItem(empty_item)
        elif selected_item is not None:
            with QSignalBlocker(self.texture_list):
                self.texture_list.setCurrentItem(selected_item)
                self.texture_list.scrollToItem(selected_item)

    def _on_selector_changed(self, _index: int = 0) -> None:
        self._refresh_current_label()
        self._populate_list()

    def _on_texture_item_clicked(self, item: QListWidgetItem) -> None:
        raw_path = item.data(ROLE_TEXTURE_PATH)
        if not raw_path or self.current_nif_key is None:
            return
        self._apply_dds_path(Path(raw_path))

    def _apply_dds_path(self, dds_path: Path) -> None:
        if self.current_nif_key is None:
            return
        self._on_apply_texture(
            self.current_nif_key,
            self.target_shape_indices(),
            self.selected_slot(),
            dds_path,
        )
        self._refresh_current_label()
        self._populate_list()

    def _handle_reset(self) -> None:
        if self.current_nif_key is None:
            return
        self._on_reset_textures(self.current_nif_key)
        self._refresh_current_label()
        self._populate_list()

    def _handle_browse(self) -> None:
        if self.current_nif_key is None:
            return
        res_tex = self._active_resolved_texture()
        start_dir = ""
        if res_tex is not None and res_tex.resolved_path is not None:
            start_dir = str(res_tex.resolved_path.parent)
        elif self._mod_textures:
            start_dir = str(self._mod_textures[0][1].parent)
        elif self.current_nif_key:
            start_dir = str(Path(self.current_nif_key).parent)

        anchor = self._anchor_widget
        chosen, _ = QFileDialog.getOpenFileName(
            self.parentWidget(),
            self._t("tex_select_dds_title"),
            start_dir,
            "DDS Textures (*.dds)",
        )
        if chosen:
            chosen_path = Path(chosen)
            if _is_diffuse_texture(chosen_path.name.casefold()):
                self._apply_dds_path(chosen_path)
            else:
                QMessageBox.warning(
                    self,
                    self._t("tex_select_dds_title"),
                    self._t("tex_non_diffuse_error"),
                )
        if anchor is not None and self.current_nif_key is not None:
            self._position_next_to(anchor)
            self.show()
            self.raise_()
            self.activateWindow()


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
            int(page): {
                str(Path(path).resolve()) for path in paths if path
            }
            for page, paths in self.config.locked_presets.items()
            if page.isdigit()
        }
        if not self._locked_presets:
            self._locked_presets = {
                1: {
                    str(Path(path).resolve())
                    for path in self.config.locked_nifs
                    if path
                }
            }
        self._current_locked_preset = min(self._locked_presets)
        self._locked_nif_paths: Set[str] = set().union(
            *self._locked_presets.values()
        )
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
                self._t("anim_loop"),
                self._t("anim_stop"),
                self._t("anim_stop_tip"),
            )
        if hasattr(self, "texture_popover"):
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

        dir_group = RefreshableGroupBox(
            self._t("mod_directory"),
            minimizable=True,
        )
        self._bind_text(dir_group, "mod_directory", "setTitle")
        dir_content = QWidget()
        dir_layout = QVBoxLayout(dir_content)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_group_layout = QVBoxLayout(dir_group)
        dir_group_layout.addWidget(dir_content)
        dir_group.set_collapsible_content(dir_content)
        dir_group.set_minimize_tooltips(
            self._t("minimize"),
            self._t("restore"),
        )

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
        self.dir_group.setVisible(True)
        mesh_layout.addWidget(dir_group)

        tree_group = RefreshableGroupBox(
            self._t("outfit_meshes"),
            self.populate_file_tree,
        )
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
        self.file_tree.itemExpanded.connect(self._on_tree_item_expanded)
        self.file_tree.itemClicked.connect(self._on_tree_item_clicked)
        self.file_tree.itemDoubleClicked.connect(self._on_tree_item_double_clicked)
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

        locked_group = RefreshableGroupBox(
            self._t("locked_wardrobe", count=0),
            minimizable=True,
        )
        locked_content = QWidget()
        locked_group_layout = QVBoxLayout(locked_content)
        locked_group_layout.setContentsMargins(0, 0, 0, 0)
        locked_group_box_layout = QVBoxLayout(locked_group)
        locked_group_box_layout.addWidget(locked_content)
        locked_group.set_collapsible_content(locked_content)
        locked_group.set_minimize_tooltips(
            self._t("minimize"),
            self._t("restore"),
        )
        self.locked_nifs_list = QListWidget()
        self.locked_nifs_list.setMinimumHeight(0)
        self.locked_nifs_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        preset_bar = QHBoxLayout()
        preset_bar.setContentsMargins(0, 0, 0, 0)
        preset_bar.setSpacing(4)
        self.locked_preset_combo = QComboBox()
        self.locked_preset_combo.setFixedHeight(26)
        self.locked_preset_combo.currentIndexChanged.connect(
            self._on_locked_preset_changed
        )
        preset_bar.addWidget(self.locked_preset_combo, stretch=1)
        self.btn_add_locked_preset = QPushButton("+")
        self.btn_add_locked_preset.setFixedSize(28, 26)
        self.btn_add_locked_preset.setToolTip(self._t("add_locked_preset"))
        self.btn_add_locked_preset.clicked.connect(self._add_locked_preset)
        preset_bar.addWidget(self.btn_add_locked_preset)
        locked_group_layout.addLayout(preset_bar)
        self.locked_nifs_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.locked_nifs_list.customContextMenuRequested.connect(
            lambda pos: self._on_wardrobe_list_context_menu(self.locked_nifs_list, pos)
        )
        self.locked_nifs_list.itemClicked.connect(self._on_added_nif_clicked)
        locked_group_layout.addWidget(self.locked_nifs_list, stretch=1)
        self._refresh_locked_preset_selector()
        added_layout.addWidget(locked_group)

        added_group = QGroupBox(self._t("wardrobe", count=0))
        added_group_layout = QVBoxLayout(added_group)
        self.added_nifs_list = QListWidget()
        self.added_nifs_list.setMinimumHeight(72)
        self.added_nifs_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.added_nifs_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.added_nifs_list.customContextMenuRequested.connect(
            lambda pos: self._on_wardrobe_list_context_menu(self.added_nifs_list, pos)
        )
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

        cfg_effects = self.config.effects
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

        s.wireframe_overlay = cfg_chk.wireframe_overlay
        s.show_ground_grid = cfg_chk.show_ground_grid

        center_layout.addWidget(self.viewport, stretch=1)

        self.anim_toolbar = AnimationPlaybackToolbar(
            on_toggle_play=self.viewport.toggle_play_pause,
            on_toggle_loop=self.viewport.set_animation_loop,
            on_seek=self.viewport.seek_animation,
            on_close=self._on_stop_animation,
            parent=center_panel,
        )
        self.anim_toolbar.set_translations(
            self._t("anim_play"),
            self._t("anim_pause"),
            self._t("anim_loop"),
            self._t("anim_stop"),
            self._t("anim_stop_tip"),
        )
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
        self.language_combo = QComboBox()
        for language_code, translations in self._translations.items():
            self.language_combo.addItem(translations["language_name"], language_code)
        self.language_combo.setCurrentIndex(
            self.language_combo.findData(self.language)
        )
        self.language_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
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

        # Lighting Controls Group
        light_group = RefreshableGroupBox(
            self._t("lighting_hemisphere"),
            self._reset_lighting_controls,
        )
        self._bind_text(light_group, "lighting_hemisphere", "setTitle")
        light_form = QFormLayout(light_group)

        self.btn_reset_lighting = light_group.btn_refresh
        assert self.btn_reset_lighting is not None
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

        ssr_group = RefreshableGroupBox(
            self._t("ssr_options"),
            self._reset_ssr_controls,
        )
        self._bind_text(ssr_group, "ssr_options", "setTitle")
        assert ssr_group.btn_refresh is not None
        self._bind_text(ssr_group.btn_refresh, "reset_ssr", "setToolTip")
        ssr_layout = QVBoxLayout(ssr_group)
        self.chk_ssr = self._add_checkbox(
            ssr_layout,
            self._t("ssr_enabled"),
            cfg_effects.enable_ssr,
            self._on_effects_changed,
        )
        self._bind_text(self.chk_ssr, "ssr_enabled")
        ssr_form = QFormLayout()
        self.slider_ssr_step, self.lbl_ssr_step = self._add_slider_row(
            ssr_form,
            self._t("ssr_step_size"),
            1,
            50,
            int(round(cfg_effects.ssr_step_size * 10)),
            f"{cfg_effects.ssr_step_size:.1f}",
            self._on_effects_changed,
            "ssr_step_size",
        )
        self.slider_ssr_steps, self.lbl_ssr_steps = self._add_slider_row(
            ssr_form,
            self._t("ssr_max_steps"),
            8,
            128,
            int(cfg_effects.ssr_max_steps),
            str(cfg_effects.ssr_max_steps),
            self._on_effects_changed,
            "ssr_max_steps",
        )
        self.slider_ssr_thickness, self.lbl_ssr_thickness = self._add_slider_row(
            ssr_form,
            self._t("ssr_thickness"),
            1,
            100,
            int(round(cfg_effects.ssr_thickness * 10)),
            f"{cfg_effects.ssr_thickness:.1f}",
            self._on_effects_changed,
            "ssr_thickness",
        )
        ssr_layout.addLayout(ssr_form)
        scroll_layout.addWidget(ssr_group)

        ao_group = RefreshableGroupBox(
            self._t("ao_options"),
            self._reset_ao_controls,
        )
        self._bind_text(ao_group, "ao_options", "setTitle")
        assert ao_group.btn_refresh is not None
        self._bind_text(ao_group.btn_refresh, "reset_ao", "setToolTip")
        ao_layout = QVBoxLayout(ao_group)
        self.chk_ao = self._add_checkbox(
            ao_layout,
            self._t("ao_enabled"),
            cfg_effects.enable_ao,
            self._on_effects_changed,
        )
        self._bind_text(self.chk_ao, "ao_enabled")
        ao_form = QFormLayout()
        self.slider_ao_radius, self.lbl_ao_radius = self._add_slider_row(
            ao_form,
            self._t("ao_radius"),
            1,
            40,
            int(round(cfg_effects.ao_radius)),
            f"{cfg_effects.ao_radius:.0f}",
            self._on_effects_changed,
            "ao_radius",
        )
        self.slider_ao_intensity, self.lbl_ao_intensity = self._add_slider_row(
            ao_form,
            self._t("ao_intensity"),
            0,
            300,
            int(round(cfg_effects.ao_intensity * 100)),
            f"{cfg_effects.ao_intensity:.2f}",
            self._on_effects_changed,
            "ao_intensity",
        )
        self.slider_ao_bias, self.lbl_ao_bias = self._add_slider_row(
            ao_form,
            self._t("ao_bias"),
            1,
            100,
            int(round(cfg_effects.ao_bias * 1000)),
            f"{cfg_effects.ao_bias:.3f}",
            self._on_effects_changed,
            "ao_bias",
        )
        ao_layout.addLayout(ao_form)
        scroll_layout.addWidget(ao_group)

        ssss_group = RefreshableGroupBox(
            self._t("ssss_options"),
            self._reset_ssss_controls,
        )
        self._bind_text(ssss_group, "ssss_options", "setTitle")
        assert ssss_group.btn_refresh is not None
        self._bind_text(ssss_group.btn_refresh, "reset_ssss", "setToolTip")
        ssss_layout = QVBoxLayout(ssss_group)
        self.chk_ssss = self._add_checkbox(
            ssss_layout,
            self._t("ssss_enabled"),
            cfg_effects.enable_ssss,
            self._on_effects_changed,
        )
        self._bind_text(self.chk_ssss, "ssss_enabled")
        ssss_form = QFormLayout()
        self.slider_ssss_radius, self.lbl_ssss_radius = self._add_slider_row(
            ssss_form,
            self._t("ssss_radius"),
            1,
            24,
            int(round(cfg_effects.ssss_radius)),
            f"{cfg_effects.ssss_radius:.0f}",
            self._on_effects_changed,
            "ssss_radius",
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

        effects = self.config.effects
        effects.enable_ssr = self.chk_ssr.isChecked()
        effects.ssr_step_size = self.slider_ssr_step.value() / 10.0
        effects.ssr_max_steps = self.slider_ssr_steps.value()
        effects.ssr_thickness = self.slider_ssr_thickness.value() / 10.0
        effects.enable_ao = self.chk_ao.isChecked()
        effects.ao_radius = float(self.slider_ao_radius.value())
        effects.ao_intensity = self.slider_ao_intensity.value() / 100.0
        effects.ao_bias = self.slider_ao_bias.value() / 1000.0
        effects.enable_ssss = self.chk_ssss.isChecked()
        effects.ssss_radius = float(self.slider_ssss_radius.value())
        effects.ssss_falloff_color = self.viewport.settings.ssss_falloff_color

        # Checkboxes
        c = self.config.checkboxes
        c.meshes_and_caliente_only = self.chk_meshes_and_caliente_only.isChecked()
        c.hide_low_weight = self.chk_hide_low_weight.isChecked()
        c.hide_gnd_folder = self.chk_hide_gnd_folder.isChecked()
        c.show_mod_dir_filter = self.dir_group.isVisible()

        c.wireframe_overlay = self.chk_wireframe.isChecked()
        c.show_ground_grid = self.chk_grid.isChecked()
        self.config.locked_presets = {
            str(page): sorted(paths)
            for page, paths in sorted(self._locked_presets.items())
        }
        self.config.locked_nifs = sorted(self._locked_nif_paths)

        self.config.save(target_path)

    def closeEvent(self, event) -> None:
        try:
            self.save_config()
        except Exception as exc:
            log.warning("Could not save config on close: %s", exc)
        super().closeEvent(event)

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
                        self.right_panel.setMinimumWidth(
                            self._settings_sidebar_min_width
                        )
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
        label_key: str,
    ) -> tuple[QSlider, QLabel]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(init_val)
        slider.setMinimumWidth(80)
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
    # Dynamic On-Demand Tree Population & Background Search
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
        self._search_folder_groups = {}
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
                self._search_folder_groups[folder_key] = folder_item
            parent_item = folder_item

        if full_path.casefold().endswith(".hkx"):
            self._create_hkx_tree_item(parent_item, file_name, full_path)
        else:
            self._create_nif_tree_item(parent_item, file_name, full_path)
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
                if self.mod_scanner and self.mod_scanner.is_mods_root:
                    if not entry.is_dir():
                        continue
                    if is_filtered_folder_name(entry.name, hide_gnd_folder):
                        continue
                    if meshes_caliente_only and not mod_has_mesh_or_caliente_folder(entry.path):
                        continue
                    self._create_folder_tree_item(
                        self.file_tree,
                        entry.name,
                        entry.path,
                        "#e0d2b4",
                        mod_root=Path(entry.path),
                    )
                    continue
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
                elif entry.is_file() and is_animation_hkx(entry.name):
                    self._create_hkx_tree_item(self.file_tree, entry.name, entry.path)
            except OSError:
                continue

        if self.mod_scanner:
            self._ensure_overwrite_fallback_item()
        self._updating_tree = False

    def _on_tree_item_expanded(self, item: QTreeWidgetItem) -> None:
        """
        Dynamically populate immediate children when a folder node is expanded.
        """
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

        is_mod_level = (
            self.current_root_dir is not None
            and Path(folder_path).parent == self.current_root_dir
            and not mod_has_mesh_or_caliente_folder(self.current_root_dir)
        )

        try:
            mod_root = Path(mod_root_value).resolve() if mod_root_value else None
            mod_relative_path = tuple(
                item.data(0, ROLE_MOD_RELATIVE_PATH) or ()
            )
            fallback_relative_path = tuple(
                item.data(0, ROLE_OVERWRITE_RELATIVE_PATH) or ()
            )
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
                overlay_entries = self.mod_scanner.fallback_entries(
                    fallback_relative_path
                )

            merged_entries = {
                entry.name.casefold(): (entry.name, entry.is_dir(), entry.path, False)
                for entry in local_entries
            }
            for entry in overlay_entries:
                merged_entries[entry.name.casefold()] = (
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
                    if meshes_caliente_only and (
                        is_mod_level or (mod_root is not None and not mod_relative_path)
                    ):
                        if name.lower() not in TARGET_MOD_FOLDERS and name.lower() != "data":
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
                            fallback_relative_path + (name,)
                            if is_overwrite_fallback
                            else ()
                        ),
                    )
                    added_any = True
                elif name.casefold().endswith(".nif") and not is_filtered_nif_filename(name, hide_0):
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
        except Exception as exc:
            log.warning("Error expanding %s: %s", folder_path, exc)
        finally:
            self._updating_tree = False

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

    def _on_tree_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._updating_tree:
            return

        hkx_key = item.data(0, ROLE_HKX_PATH)
        if hkx_key:
            self._load_and_play_animation(hkx_key)
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
            self.anim_toolbar.update_state(
                self.viewport.is_animation_playing,
                self.viewport.is_animation_looping,
            )
            self.anim_toolbar.setVisible(True)
            return True
        except Exception as exc:
            log.error("Failed loading HKX animation %s: %s", hkx_key, exc)
            if show_error:
                QMessageBox.warning(
                    self,
                    self._t("hkx_load_error"),
                    self._t(
                        "hkx_load_error_message",
                        filename=Path(hkx_key).name,
                        error=exc,
                    ),
                )
            return False

    def _on_stop_animation(self) -> None:
        self._active_hkx_path = None
        self.viewport.stop_animation()
        self.anim_toolbar.setVisible(False)

    def _on_viewport_animation_stopped(self) -> None:
        self._active_hkx_path = None
        if hasattr(self, "anim_toolbar"):
            self.anim_toolbar.setVisible(False)

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
                    self._t(
                        "nif_load_error_message",
                        filename=Path(nif_key).name,
                        error=exc,
                    ),
                )
            return False

    def _detach_outfit_nif(self, nif_key: str) -> None:
        if (
            hasattr(self, "texture_popover")
            and self.texture_popover.isVisible()
            and self.texture_popover.current_nif_key == nif_key
        ):
            self.texture_popover.close()
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
        owner = self.mod_scanner.owner_for_nif(nif_key) if self.mod_scanner else None
        if owner is not None:
            return owner.name
        if (
            self.mod_scanner
            and self.mod_scanner.fallback_target_for_nif(nif_key) is not None
        ):
            return "[Overwrite]"
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
        if locked and nif_key not in self._locked_presets.get(
            self._current_locked_preset, set()
        ):
            return
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
            text=display_text,
            on_remove=lambda _checked=False, k=nif_key: (
                self._unlock_nif(k) if locked else self._detach_outfit_nif(k)
            ),
            on_select=lambda k=nif_key: self._select_nif_in_tree(k),
            on_toggle_visibility=lambda _checked=False, k=nif_key: self._toggle_nif_visibility(k),
            on_double_click=lambda k=nif_key: (
                self._unlock_nif(k) if locked else self._lock_nif(k)
            ),
            on_right_click=lambda widget, k=nif_key: self._open_nif_texture_popover(
                k, widget
            ),
            visible=self._nif_visibility.get(nif_key, True),
            remove_tooltip=self._t("unlock_nif") if locked else self._t("remove_nif"),
            hide_tooltip=self._t("hide_nif"),
            show_tooltip=self._t("show_nif"),
            show_remove=not locked,
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
        locked_list_height = locked_row_height + 2 * self.locked_nifs_list.frameWidth() + 2
        self.locked_nifs_list.setFixedHeight(locked_list_height)
        self.locked_nifs_list.setVisible(bool(self._locked_nif_items))

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
                    self._t("locked_preset_page", page=page),
                    page,
                )
            index = self.locked_preset_combo.findData(self._current_locked_preset)
            self.locked_preset_combo.setCurrentIndex(max(0, index))

    def _on_locked_preset_changed(self, index: int) -> None:
        if index < 0:
            return
        page = self.locked_preset_combo.itemData(index)
        if not isinstance(page, int) or page == self._current_locked_preset:
            return
        self._current_locked_preset = page
        self._refresh_added_nifs()

    def _add_locked_preset(self) -> None:
        page = 1
        while page in self._locked_presets:
            page += 1
        self._locked_presets[page] = set()
        self._current_locked_preset = page
        self._refresh_locked_preset_selector()
        self._refresh_added_nifs()
        self.save_config()

    def _remove_empty_locked_preset(self, page: int) -> None:
        if self._locked_presets.get(page) or len(self._locked_presets) <= 1:
            return
        del self._locked_presets[page]
        if self._current_locked_preset == page:
            self._current_locked_preset = min(self._locked_presets)
        self._refresh_locked_preset_selector()

    def _on_added_nif_clicked(self, item: QListWidgetItem) -> None:
        nif_key = item.data(ROLE_ADDED_NIF_PATH)
        if nif_key:
            self._select_nif_in_tree(nif_key)

    def _on_wardrobe_list_context_menu(
        self, list_widget: QListWidget, pos: QPoint
    ) -> None:
        item = list_widget.itemAt(pos)
        if item is None:
            return
        nif_key = item.data(ROLE_ADDED_NIF_PATH)
        if not nif_key:
            return
        anchor_widget = list_widget.itemWidget(item) or list_widget
        self._open_nif_texture_popover(str(nif_key), anchor_widget)

    def _open_nif_texture_popover(
        self, nif_key: str, anchor_widget: QWidget
    ) -> None:
        nif_data = self._loaded_nif_cache.get(nif_key)
        if nif_data is None:
            try:
                nif_data = self.loader.load_nif(nif_key)
                self._loaded_nif_cache[nif_key] = nif_data
            except Exception as exc:
                log.error("Could not inspect NIF %s for textures: %s", nif_key, exc)
                return

        mod_textures = self._discover_mod_textures_for_nif(nif_key)
        self.texture_popover.show_for_nif(
            nif_key=nif_key,
            nif_data=nif_data,
            mod_textures=mod_textures,
            anchor_widget=anchor_widget,
        )

    def _discover_mod_textures_for_nif(
        self, nif_key: str
    ) -> List[tuple[str, Path]]:
        owner = self.mod_scanner.owner_for_nif(nif_key) if self.mod_scanner else None
        if owner is not None:
            mod_root = owner
        elif (
            self.mod_scanner
            and self.mod_scanner.fallback_target_for_nif(nif_key) is not None
            and self.mod_scanner.overwrite_root is not None
        ):
            mod_root = self.mod_scanner.overwrite_root
        elif self.current_root_dir is not None:
            mod_root = self.current_root_dir
            is_mod_directory = any(
                (self.current_root_dir / folder).is_dir()
                for folder in ("meshes", "textures", "Data/meshes", "Data/textures")
            )
            if not is_mod_directory:
                try:
                    relative = Path(nif_key).resolve().relative_to(
                        self.current_root_dir
                    )
                except ValueError:
                    return []
                if len(relative.parts) > 1:
                    mod_root = self.current_root_dir / relative.parts[0]
        else:
            mod_root = Path(nif_key).resolve().parent

        texture_roots = [
            mod_root / "textures",
            mod_root / "Data" / "textures",
        ]
        results_by_label: Dict[str, tuple[str, Path]] = {}

        for texture_root in texture_roots:
            if not texture_root.is_dir():
                continue
            for root, dirs, files in os.walk(texture_root):
                dirs.sort(key=str.casefold)
                root_path = Path(root)
                for file_name in sorted(files, key=str.casefold):
                    if not file_name.casefold().endswith(".dds"):
                        continue
                    full_path = (root_path / file_name).resolve()
                    try:
                        rel_label = full_path.relative_to(texture_root).as_posix()
                    except ValueError:
                        rel_label = full_path.name
                    label_parts = Path(rel_label).parts
                    if label_parts and label_parts[0].casefold() == mod_root.name.casefold():
                        rel_label = (
                            Path(*label_parts[1:]).as_posix()
                            if len(label_parts) > 1
                            else full_path.name
                        )
                    label_key = rel_label.casefold()
                    if label_key not in results_by_label:
                        results_by_label[label_key] = (rel_label, full_path)

        return list(results_by_label.values())

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

        for sh_idx in shape_indices:
            if not (0 <= sh_idx < len(nif_data.shapes)):
                continue
            chosen_path = dds_path
            if multi_shape:
                sh = nif_data.shapes[sh_idx]
                cur_res = sh.shader.resolved_textures.get(slot)
                cur_name = ""
                if cur_res is not None and cur_res.resolved_path is not None:
                    cur_name = cur_res.resolved_path.name
                elif cur_res is not None and cur_res.raw_nif_path:
                    cur_name = Path(cur_res.raw_nif_path.replace("\\", "/")).name
                if cur_name:
                    sibling = target_dir / cur_name
                    if sibling.is_file():
                        chosen_path = sibling
            try:
                res_tex = self.resolver.load_from_file(slot, chosen_path)
            except Exception as exc:
                log.warning("Could not decode texture %s: %s", chosen_path, exc)
                continue
            self.viewport.update_nif_shape_texture(nif_key, sh_idx, slot, res_tex)

    def _on_reset_nif_textures(self, nif_key: str) -> None:
        nif_data = self._loaded_nif_cache.get(nif_key)
        self.viewport.reset_nif_textures(nif_key, nif_data)

    def _autoload_locked_nifs(self) -> None:
        """Load NIFs saved in the locked wardrobe without showing startup dialogs."""
        missing_nifs = {
            nif_key
            for nif_key in self._locked_nif_paths
            if not Path(nif_key).is_file()
        }
        if missing_nifs:
            for nif_key in sorted(missing_nifs):
                log.warning("Removing unavailable locked NIF from config: %s", nif_key)
            self._locked_nif_paths.difference_update(missing_nifs)
            self.save_config()

        for nif_key in sorted(self._locked_nif_paths):
            self._load_and_attach_outfit_nif(nif_key, show_error=False)
        self._startup_locked_nif_refresh_pending = bool(self._locked_nif_items)

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
        overlay_target = (
            self.mod_scanner.target_for_nif(nif_key) if self.mod_scanner else None
        )
        fallback_target = (
            self.mod_scanner.fallback_target_for_nif(nif_key)
            if self.mod_scanner
            else None
        )
        if overlay_target is not None:
            if self.search_edit.text():
                self.search_edit.clear()
            mod_root, target_parts = overlay_target
            mod_item = next(
                (
                    self.file_tree.topLevelItem(index)
                    for index in range(self.file_tree.topLevelItemCount())
                    if self.file_tree.topLevelItem(index).data(0, ROLE_MOD_ROOT_PATH)
                    and Path(
                        self.file_tree.topLevelItem(index).data(0, ROLE_MOD_ROOT_PATH)
                    ).resolve()
                    == mod_root
                ),
                None,
            )
            if mod_item is None:
                return
            parent_item = mod_item
            relative_parts: tuple[str, ...] = ()
            for folder_name in target_parts[:-1]:
                self._on_tree_item_expanded(parent_item)
                relative_parts += (folder_name,)
                parent_item = next(
                    (
                        parent_item.child(index)
                        for index in range(parent_item.childCount())
                        if tuple(
                            str(part).casefold()
                            for part in (
                                parent_item.child(index).data(
                                    0, ROLE_MOD_RELATIVE_PATH
                                )
                                or ()
                            )
                        )
                        == tuple(part.casefold() for part in relative_parts)
                    ),
                    None,
                )
                if parent_item is None:
                    return
                parent_item.setExpanded(True)
            self._on_tree_item_expanded(parent_item)
            target_item = next(
                (
                    parent_item.child(index)
                    for index in range(parent_item.childCount())
                    if parent_item.child(index).data(0, ROLE_NIF_PATH)
                    and Path(
                        parent_item.child(index).data(0, ROLE_NIF_PATH)
                    ).resolve()
                    == Path(nif_key).resolve()
                ),
                None,
            )
            if target_item is None:
                return
            self._highlight_tree_item(target_item)
            return
        if fallback_target is not None:
            if self.search_edit.text():
                self.search_edit.clear()
            fallback_item = next(
                (
                    self.file_tree.topLevelItem(index)
                    for index in range(self.file_tree.topLevelItemCount())
                    if self.file_tree.topLevelItem(index).data(
                        0, ROLE_OVERWRITE_FALLBACK
                    )
                ),
                None,
            )
            if fallback_item is None:
                return
            parent_item = fallback_item
            relative_parts: tuple[str, ...] = ()
            for folder_name in fallback_target[:-1]:
                self._on_tree_item_expanded(parent_item)
                relative_parts += (folder_name,)
                parent_item = next(
                    (
                        parent_item.child(index)
                        for index in range(parent_item.childCount())
                        if tuple(
                            str(part).casefold()
                            for part in (
                                parent_item.child(index).data(
                                    0, ROLE_OVERWRITE_RELATIVE_PATH
                                )
                                or ()
                            )
                        )
                        == tuple(part.casefold() for part in relative_parts)
                    ),
                    None,
                )
                if parent_item is None:
                    return
                parent_item.setExpanded(True)
            self._on_tree_item_expanded(parent_item)
            target_item = next(
                (
                    parent_item.child(index)
                    for index in range(parent_item.childCount())
                    if parent_item.child(index).data(0, ROLE_NIF_PATH)
                    and Path(
                        parent_item.child(index).data(0, ROLE_NIF_PATH)
                    ).resolve()
                    == Path(nif_key).resolve()
                ),
                None,
            )
            if target_item is None:
                return
            self._highlight_tree_item(target_item)
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

        self._highlight_tree_item(target_item)

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
        color = QColorDialog.getColor(
            initial,
            self,
            self._t("ssss_falloff_color"),
        )
        if not color.isValid():
            return
        self.viewport.settings.ssss_falloff_color = (
            color.redF(),
            color.greenF(),
            color.blueF(),
        )
        self._update_ssss_falloff_button()
        self.viewport.update()
