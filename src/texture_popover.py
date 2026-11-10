"""
Texture popover UI and diffuse texture switching helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QPoint, QSignalBlocker, Qt
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .nif_loader import LoadedNifData, ShapeMeshData
from .ui_widgets import ElidedLabel, TEXTURE_PATH_STYLE

log = logging.getLogger("dibellas_wardrobe.texture_popover")

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


def resolve_shape_texture_path(
    shape: ShapeMeshData,
    slot: str,
    default_path: Path,
    target_dir: Path,
) -> Path:
    """Find a sibling texture in target_dir matching the shape's original texture name."""
    cur_res = shape.shader.resolved_textures.get(slot)
    cur_name = ""
    if cur_res is not None and cur_res.resolved_path is not None:
        cur_name = cur_res.resolved_path.name
    elif cur_res is not None and cur_res.raw_nif_path:
        cur_name = Path(cur_res.raw_nif_path.replace("\\", "/")).name
    if cur_name:
        sibling = target_dir / cur_name
        if sibling.is_file():
            return sibling
    return default_path


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
        self.lbl_current_tex.setStyleSheet(TEXTURE_PATH_STYLE)
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

            default_shape = self.ALL_OUTFIT_SHAPES if len(outfit_indices) >= 2 else 0
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
