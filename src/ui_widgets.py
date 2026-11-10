"""
Custom UI widgets and components for Dibella's Wardrobe.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from PySide6.QtCore import (
    QModelIndex,
    QMimeData,
    QPoint,
    QRect,
    Qt,
    QTimer,
)
from PySide6.QtGui import (
    QColor,
    QDrag,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyle,
    QStyleOptionGroupBox,
    QTreeWidget,
    QWidget,
)

log = logging.getLogger("dibellas_wardrobe.ui_widgets")

ROLE_NIF_PATH = Qt.ItemDataRole.UserRole + 1
ROLE_FOLDER_PATH = Qt.ItemDataRole.UserRole + 2
ROLE_IS_LOADED = Qt.ItemDataRole.UserRole + 3
ROLE_ADDED_NIF_PATH = Qt.ItemDataRole.UserRole
ROLE_MOD_ROOT_PATH = Qt.ItemDataRole.UserRole + 4
ROLE_MOD_RELATIVE_PATH = Qt.ItemDataRole.UserRole + 5
ROLE_OVERWRITE_FALLBACK = Qt.ItemDataRole.UserRole + 6
ROLE_OVERWRITE_RELATIVE_PATH = Qt.ItemDataRole.UserRole + 7
ROLE_HKX_PATH = Qt.ItemDataRole.UserRole + 8
NIF_DRAG_MIME_TYPE = "application/x-dibellas-wardrobe-nif"

HEADER_ACTIONS_STYLE = """
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


def _nif_path_from_mime_data(mime_data: QMimeData) -> Optional[str]:
    """Extract a NIF filesystem path from drag-and-drop MIME data."""
    if not mime_data.hasFormat(NIF_DRAG_MIME_TYPE):
        return None
    try:
        return bytes(mime_data.data(NIF_DRAG_MIME_TYPE)).decode("utf-8")
    except UnicodeDecodeError:
        log.warning("Ignoring malformed NIF drag data")
        return None


def _start_nif_drag(source: QWidget, nif_path: str) -> None:
    """Initiate a Qt move drag operation carrying a NIF path."""
    mime_data = QMimeData()
    mime_data.setData(NIF_DRAG_MIME_TYPE, nif_path.encode("utf-8"))
    drag = QDrag(source)
    drag.setMimeData(mime_data)
    drag.exec(Qt.DropAction.MoveAction)


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


class RefreshableGroupBox(QGroupBox):
    """
    QGroupBox with an embedded header action bar (refresh and collapse buttons)
    aligned with the native title text.
    """

    def __init__(
        self,
        title: str = "",
        on_refresh: Optional[Callable[[], None]] = None,
        minimizable: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(title, parent)
        self._on_refresh = on_refresh
        self._minimizable = minimizable
        self._collapsible_content: Optional[QWidget] = None
        self._collapsed = False
        self._minimize_tooltip = "Minimize"
        self._restore_tooltip = "Restore"

        self.header_actions = QFrame(self)
        self.header_actions.setObjectName("headerActions")
        self.header_actions.setStyleSheet(HEADER_ACTIONS_STYLE)

        self.header_actions_layout = QHBoxLayout(self.header_actions)
        self.header_actions_layout.setContentsMargins(1, 1, 1, 1)
        self.header_actions_layout.setSpacing(2)

        self.btn_refresh: Optional[QPushButton] = None
        if self._on_refresh is not None:
            self.btn_refresh = QPushButton("↻", self.header_actions)
            self.btn_refresh.setObjectName("headerActionButton")
            self.btn_refresh.setFixedSize(22, 22)
            self.btn_refresh.setToolTip("Refresh")
            self.btn_refresh.clicked.connect(self._on_refresh)
            self.header_actions_layout.addWidget(self.btn_refresh)

        self.btn_minimize: Optional[QPushButton] = None
        if self._minimizable:
            self.btn_minimize = QPushButton("−", self.header_actions)
            self.btn_minimize.setObjectName("headerActionButton")
            self.btn_minimize.setFixedSize(22, 22)
            self.btn_minimize.setToolTip(self._minimize_tooltip)
            self.btn_minimize.clicked.connect(self._toggle_content)
            self.header_actions_layout.addWidget(self.btn_minimize)

        has_actions = bool(self.btn_refresh or self.btn_minimize)
        self.header_actions.setVisible(has_actions)

    def set_collapsible_content(self, content: QWidget) -> None:
        self._collapsible_content = content
        self.set_collapsed(self._collapsed)

    def setTitle(self, title: str) -> None:
        super().setTitle(title)
        self._position_header_actions()

    def set_collapsed(self, collapsed: bool) -> None:
        self._collapsed = collapsed
        if self._collapsible_content is not None:
            self._collapsible_content.setVisible(not collapsed)
        if self.btn_minimize is not None:
            self.btn_minimize.setText("+" if collapsed else "−")
            self._update_minimize_tooltip()
        self._position_header_actions()

    def set_minimize_tooltips(self, minimize: str, restore: str) -> None:
        self._minimize_tooltip = minimize
        self._restore_tooltip = restore
        self._update_minimize_tooltip()

    def _toggle_content(self) -> None:
        self.set_collapsed(not self._collapsed)

    def _update_minimize_tooltip(self) -> None:
        if self.btn_minimize is not None:
            tip = self._restore_tooltip if self._collapsed else self._minimize_tooltip
            self.btn_minimize.setToolTip(tip)

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


class AddedNifRowWidget(QWidget):
    """Row widget for a wardrobe NIF with visibility, texture selector, and drag controls."""

    def __init__(
        self,
        nif_path: str,
        text: str,
        on_select,
        on_toggle_visibility,
        on_double_click,
        on_texture_select,
        on_right_click=None,
        texture_tooltip: str = "Change outfit textures",
        visible: bool = True,
        hide_tooltip: str = "Hide this NIF",
        show_tooltip: str = "Show this NIF",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._nif_path = nif_path
        self._drag_start_position: Optional[QPoint] = None
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

        self.btn_texture_select = QPushButton(">")
        self.btn_texture_select.setFixedSize(20, 20)
        self.btn_texture_select.setToolTip(texture_tooltip)
        self.btn_texture_select.setStyleSheet(
            """
            QPushButton {
                background: transparent;
                border: none;
                color: #aeb8cc;
                font-size: 13px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                color: #f0d59c;
                background-color: #303543;
                border-radius: 3px;
            }
            """
        )
        self.btn_texture_select.clicked.connect(
            lambda _checked=False: on_texture_select(self.btn_texture_select)
        )
        layout.addWidget(self.btn_texture_select, stretch=0)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton and self._on_right_click:
            event.accept()
            self._on_right_click()
            return
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_position = event.position().toPoint()
            self._on_select()

    def mouseMoveEvent(self, event) -> None:
        if (
            self._drag_start_position is not None
            and event.buttons() & Qt.MouseButton.LeftButton
            and (event.position().toPoint() - self._drag_start_position).manhattanLength()
            >= QApplication.startDragDistance()
        ):
            self._drag_start_position = None
            _start_nif_drag(self, self._nif_path)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_start_position = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            event.accept()
            self._on_double_click()
            return
        super().mouseDoubleClickEvent(event)

    def set_visible(self, visible: bool) -> None:
        self.btn_visibility.setText("◉" if visible else "○")
        self.btn_visibility.setToolTip(
            self._hide_tooltip if visible else self._show_tooltip
        )


class NeatTreeWidget(QTreeWidget):
    """Clean tree widget with compact indentation, guide lines, and drop handling."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_nif_dropped: Optional[Callable[[str], None]] = None
        self.setIndentation(14)
        self.setHeaderHidden(True)
        self.setAnimated(True)

    def startDrag(self, _supported_actions) -> None:
        item = self.currentItem()
        nif_path = item.data(0, ROLE_NIF_PATH) if item is not None else None
        if nif_path:
            _start_nif_drag(self, str(nif_path))

    def dragEnterEvent(self, event) -> None:
        if _nif_path_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        is_drop_target = item is not None and bool(
            item.data(0, ROLE_FOLDER_PATH)
            or item.data(0, ROLE_MOD_ROOT_PATH)
            or item.data(0, ROLE_OVERWRITE_FALLBACK)
        )
        if is_drop_target and _nif_path_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event) -> None:
        item = self.itemAt(event.position().toPoint())
        nif_path = _nif_path_from_mime_data(event.mimeData())
        is_drop_target = item is not None and bool(
            item.data(0, ROLE_FOLDER_PATH)
            or item.data(0, ROLE_MOD_ROOT_PATH)
            or item.data(0, ROLE_OVERWRITE_FALLBACK)
        )
        if is_drop_target and nif_path and self.on_nif_dropped is not None:
            self.on_nif_dropped(nif_path)
            event.acceptProposedAction()
            return
        event.ignore()

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


class PresetNifListWidget(QListWidget):
    """NIF list that accepts drops from the mod tree or another outfit list."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.on_nif_dropped: Optional[Callable[[str], None]] = None
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)

    def dragEnterEvent(self, event) -> None:
        if _nif_path_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        if _nif_path_from_mime_data(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event) -> None:
        nif_path = _nif_path_from_mime_data(event.mimeData())
        if nif_path and self.on_nif_dropped is not None:
            self.on_nif_dropped(nif_path)
            event.acceptProposedAction()
            return
        event.ignore()


class AnimationPlaybackToolbar(QFrame):
    """
    Docked bottom playback toolbar for `.hkx` animations.

    Displays metadata (`[Mod Name (Parent Directory)] - [Filename.hkx]`), a formatted
    timestamp counter (`MM:SS.CC / MM:SS.CC`), an interactive timeline scrubber, and
    Play/Pause and Close/Reset controls.
    """

    SCRUB_STEPS = 1000

    def __init__(
        self,
        on_toggle_play,
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
        self._updating_slider = False

        self.setObjectName("animPlaybackToolbar")
        self.setFixedHeight(44)
        self.setStyleSheet(
            """
            QFrame#animPlaybackToolbar {
                background-color: #161820;
                border: 1px solid #363c50;
                border-radius: 4px;
            }
            """
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self.btn_play = QPushButton(self._play_text)
        self.btn_play.setFixedSize(28, 28)
        self.btn_play.setToolTip(self._play_tooltip)
        self.btn_play.setStyleSheet(
            """
            QPushButton {
                background-color: #242834;
                border: 1px solid #454d68;
                border-radius: 4px;
                color: #e4e7ef;
                font-size: 13px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #313747;
                color: #f0d59c;
            }
            """
        )
        self.btn_play.clicked.connect(on_toggle_play)
        layout.addWidget(self.btn_play)

        self.btn_close = QPushButton("×")
        self.btn_close.setFixedSize(28, 28)
        self.btn_close.setToolTip("Close animation")
        self.btn_close.setStyleSheet(
            """
            QPushButton {
                background-color: #242834;
                border: 1px solid #454d68;
                border-radius: 4px;
                color: #e4e7ef;
                font-size: 15px;
                font-weight: bold;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #442428;
                border-color: #943e49;
                color: #ff9da8;
            }
            """
        )
        self.btn_close.clicked.connect(on_close)
        layout.addWidget(self.btn_close)

        self.lbl_metadata = ElidedLabel("")
        self.lbl_metadata.setStyleSheet("color: #f0d59c; font-weight: 600; font-size: 12px;")
        layout.addWidget(self.lbl_metadata, stretch=0)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.SCRUB_STEPS)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider_value_changed)
        layout.addWidget(self.slider, stretch=1)

        self.lbl_time = QLabel("00:00.00 / 00:00.00")
        self.lbl_time.setFixedWidth(135)
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_time.setStyleSheet("color: #aeb8cc; font-family: monospace; font-size: 11px;")
        layout.addWidget(self.lbl_time)

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        s = max(0.0, float(seconds))
        minutes = int(s // 60)
        remaining = s % 60
        sec = int(remaining)
        centisec = int((remaining - sec) * 100)
        return f"{minutes:02d}:{sec:02d}.{centisec:02d}"

    def set_translations(
        self,
        play_text: str,
        pause_text: str,
        close_text: str,
        close_tooltip: Optional[str] = None,
    ) -> None:
        self._play_tooltip = play_text
        self._pause_tooltip = pause_text
        self.btn_play.setToolTip(
            self._pause_tooltip if self._is_playing else self._play_tooltip
        )
        self.btn_close.setText("×")
        self.btn_close.setToolTip(close_tooltip or close_text)

    def set_metadata(self, label_text: str, duration: float) -> None:
        self._duration = max(0.0, float(duration))
        self.lbl_metadata.setText(label_text)
        self.lbl_metadata.setToolTip(label_text)
        self.update_time(0.0, self._duration)

    def update_state(self, is_playing: bool) -> None:
        self._is_playing = bool(is_playing)
        self.btn_play.setText(self._pause_text if is_playing else self._play_text)
        self.btn_play.setToolTip(
            self._pause_tooltip if is_playing else self._play_tooltip
        )

    def update_time(self, current_time: float, duration: float) -> None:
        if duration > 0.0:
            self._duration = float(duration)
        cur_str = self.format_timestamp(current_time)
        dur_str = self.format_timestamp(self._duration)
        self.lbl_time.setText(f"{cur_str} / {dur_str}")

        if not self.slider.isSliderDown() and self._duration > 0.0:
            self._updating_slider = True
            norm = max(0.0, min(1.0, float(current_time) / self._duration))
            self.slider.setValue(int(round(norm * self.SCRUB_STEPS)))
            self._updating_slider = False

    def _on_slider_value_changed(self, value: int) -> None:
        if self._updating_slider or not self.slider.isSliderDown():
            return
        if self._duration <= 0.0:
            return
        target_time = (float(value) / float(self.SCRUB_STEPS)) * self._duration
        self._on_seek(target_time)
