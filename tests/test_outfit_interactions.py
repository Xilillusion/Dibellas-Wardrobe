import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QHelpEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QToolTip

from src.main_window import DibellasWardrobeWindow
from src.ui_widgets import NIF_DRAG_MIME_TYPE


class OutfitInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.files = [root / f"outfit{i}.nif" for i in range(1, 5)]
        for path in self.files:
            path.touch()
        self.keys = [str(path.resolve()) for path in self.files]
        config = root / "config.json"
        config.write_text(json.dumps({
            "locked_presets": {"1": [self.keys[0]], "2": [self.keys[1]], "3": [self.keys[2]]},
        }), encoding="utf-8")
        self.load_patch = patch("src.main_window.NifLoader.load_nif", return_value=object())
        self.add_patch = patch("src.viewport.OutfitViewportWidget.add_nif")
        self.remove_patch = patch("src.viewport.OutfitViewportWidget.remove_nif")
        self.load_patch.start()
        self.add = self.add_patch.start()
        self.remove = self.remove_patch.start()
        self.window = DibellasWardrobeWindow(initial_root_dir=root, config_path=config)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()
        self.remove_patch.stop()
        self.add_patch.stop()
        self.load_patch.stop()
        self.temp.cleanup()

    def drop_nif(self, target, key):
        mime = QMimeData()
        mime.setData(NIF_DRAG_MIME_TYPE, key.encode("utf-8"))
        viewport = target.viewport()
        point = QPoint(5, 5)
        enter = QDragEnterEvent(point, Qt.DropAction.MoveAction, mime,
                                Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(viewport, enter)
        move = QDragMoveEvent(point, Qt.DropAction.MoveAction, mime,
                              Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(viewport, move)
        drop = QDropEvent(QPointF(point), Qt.DropAction.MoveAction, mime,
                          Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(viewport, drop)
        self.assertTrue(drop.isAccepted())

    def test_animation_tooltip_shows_full_path_for_each_clip(self):
        root = Path(self.temp.name)
        paths = (root / "mod one" / "animations" / "idle.hkx",
                 root / "mod two" / "animations" / "idle.hkx")
        for path in paths:
            loaded = SimpleNamespace(filepath=path, duration=2.0)
            with patch.object(self.window.viewport, "load_animation", return_value=loaded):
                self.assertTrue(self.window._load_and_play_animation(str(path)))
            label = self.window.anim_toolbar.lbl_metadata
            self.assertEqual(label.toolTip(), str(path.resolve()))
            self.assertEqual(label.text(), self.window._format_animation_metadata(path))

        def tooltip_style(widget):
            QToolTip.hideText()
            QTest.qWait(350)
            self.app.sendEvent(widget, QHelpEvent(
                QHelpEvent.Type.ToolTip, widget.rect().center(), widget.mapToGlobal(widget.rect().center())
            ))
            self.app.processEvents()
            tip = next(w for w in self.app.topLevelWidgets()
                       if w.isVisible() and w.windowType() == Qt.WindowType.ToolTip)
            self.assertEqual(tip.text(), widget.toolTip())
            return (tip.font(), tip.palette().color(tip.foregroundRole()),
                    tip.palette().color(tip.backgroundRole()))

        texture_label = self.window.texture_popover.lbl_current_tex
        texture_label.setToolTip(str(paths[0]))
        self.assertEqual(tooltip_style(label), tooltip_style(texture_label))
        QToolTip.hideText()

    def test_cbpc_softness_sliders_persist_and_restore(self):
        viewport = self.window.viewport
        viewport.cbpc.set_mesh_bones("body", ["L Breast01", "NPC L Butt", "NPC Belly"])
        viewport._sync_cbpc_overlay()
        for group, value in (("breasts", 40), ("butt", 10), ("belly", 70)):
            viewport.cbpc_overlay.softness_sliders[group].setValue(value)
        expected = {"breasts": 2.0, "butt": 0.5, "belly": 3.5, "other": 1.0}
        self.assertEqual(viewport.cbpc.softness, expected)
        config_path = self.window.config_path
        root = self.window.current_root_dir
        self.window.close()
        self.assertEqual(json.loads(Path(config_path).read_text(encoding="utf-8"))["cbpc"]["softness"], expected)
        self.window = DibellasWardrobeWindow(initial_root_dir=root, config_path=config_path)
        self.assertEqual(self.window.viewport.cbpc.softness, expected)
        sliders = self.window.viewport.cbpc_overlay.softness_sliders
        self.assertEqual([sliders[g].value() for g in ("breasts", "butt", "belly")], [40, 10, 70])

    def test_cbpc_controls_follow_language_on_startup_and_change(self):
        self.window.language_combo.setCurrentIndex(self.window.language_combo.findData("zh"))
        config_path = self.window.config_path
        root = self.window.current_root_dir
        self.window.close()
        self.window = DibellasWardrobeWindow(initial_root_dir=root, config_path=config_path)
        overlay = self.window.viewport.cbpc_overlay
        expected = {
            "zh": ("启用 CBPC 物理", ("重力", "胸部柔软度", "臀部柔软度", "腹部柔软度")),
            "en": ("Enable CBPC Physics", ("Gravity", "Breasts softness", "Butt softness", "Belly softness")),
        }
        keys = ("gravity", "breasts", "butt", "belly")
        changes = []
        overlay.gravityChanged.connect(lambda value: changes.append(value))
        overlay.softnessChanged.connect(lambda group, value: changes.append((group, value)))
        for language in ("zh", "en", "zh"):
            self.window.language_combo.setCurrentIndex(self.window.language_combo.findData(language))
            self.assertEqual(self.window.language, language)
            self.assertEqual(overlay.checkbox.text(), expected[language][0])
            self.assertEqual(tuple(overlay.captions[key].text() for key in keys), expected[language][1])
            sliders = (overlay.slider, *(overlay.softness_sliders[key] for key in keys[1:]))
            self.assertEqual(tuple(slider.accessibleName() for slider in sliders), expected[language][1])
            self.assertEqual(overlay.x() + overlay.width(), self.window.viewport.width() - overlay.MARGIN)
            self.assertEqual(overlay.y() + overlay.height(), self.window.viewport.height() - overlay.MARGIN)
        self.assertEqual(changes, [])

    @patch("src.smp_bridge.SmpBridgeSolver.available", new_callable=PropertyMock, return_value=True)
    def test_smp_toggle_above_cbpc_drives_viewport_and_persists(self, _available):
        viewport = self.window.viewport
        overlay = viewport.cbpc_overlay
        self.assertIs(overlay.layout().itemAt(0).widget(), overlay.smp_checkbox)
        self.assertIs(overlay.layout().itemAt(1).widget(), overlay.checkbox)
        self.assertEqual(overlay.smp_checkbox.styleSheet(), overlay.checkbox.styleSheet())
        self.assertEqual(overlay.smp_checkbox.text(), "Enable SMP Physics")
        self.assertFalse(self.window.anim_toolbar.isVisible())
        self.assertTrue(overlay.smp_checkbox.isVisible())
        self.assertTrue(overlay.smp_checkbox.isChecked())
        self.assertTrue(viewport.smp.enabled)
        overlay.smp_checkbox.setChecked(False)
        self.assertFalse(viewport.smp.enabled)
        config_path = self.window.config_path
        root = self.window.current_root_dir
        self.window.close()
        self.assertFalse(json.loads(Path(config_path).read_text(encoding="utf-8"))["smp"]["enabled"])
        self.window = DibellasWardrobeWindow(initial_root_dir=root, config_path=config_path)
        self.assertFalse(self.window.viewport.smp.enabled)
        overlay = self.window.viewport.cbpc_overlay
        self.assertFalse(overlay.smp_checkbox.isChecked())
        overlay.smp_checkbox.setChecked(True)
        self.assertTrue(self.window.viewport.smp.enabled)
        self.window.language_combo.setCurrentIndex(self.window.language_combo.findData("zh"))
        self.assertEqual(overlay.smp_checkbox.text(), "启用 SMP 物理")
        self.assertEqual(overlay.checkbox.text(), "启用 CBPC 物理")

    @patch("src.smp_bridge.SmpBridgeSolver.available", new_callable=PropertyMock, return_value=True)
    def test_smp_unavailable_state_and_tooltip_follow_language(self, _available):
        overlay = self.window.viewport.cbpc_overlay
        changes = []
        overlay.smpEnabledChanged.connect(changes.append)
        with patch.object(type(self.window.viewport.smp), "available", new_callable=PropertyMock, return_value=False):
            self.window._apply_smp_texts()
            self.assertFalse(overlay.smp_checkbox.isEnabled())
            self.assertEqual(overlay.smp_checkbox.toolTip(), self.window._t("smp_unavailable"))
            self.window.language_combo.setCurrentIndex(self.window.language_combo.findData("zh"))
            self.assertEqual(overlay.smp_checkbox.toolTip(), self.window._t("smp_unavailable"))
        self.window._apply_smp_texts()
        self.assertEqual(overlay.smp_checkbox.toolTip(), self.window._t("smp_tip"))
        self.assertEqual(changes, [])

    def test_breast_gravity_slider_refreshes_static_and_paused_poses(self):
        viewport = self.window.viewport
        viewport.cbpc.set_mesh_bones("body", ["L Breast01", "NPC L Butt"])
        viewport.set_cbpc_softness("breasts", 0.0)
        viewport._sync_cbpc_overlay()
        for paused in (False, True):
            viewport.anim_engine.active_animation = SimpleNamespace() if paused else None
            pose = viewport.anim_engine.get_bind_pose()
            with patch.object(viewport.anim_engine, "evaluate_pose", return_value=pose):
                viewport.cbpc_overlay.slider.setValue(0)
                baseline = viewport._current_pose_matrices["L Breast01"].copy()
                viewport.cbpc_overlay.slider.setValue(90)
                heavy = viewport._current_pose_matrices["L Breast01"]
            self.assertLess(float(heavy[2, 3]), float(baseline[2, 3]) - 0.1)
            self.assertTrue(np.allclose(heavy[:2, 3], baseline[:2, 3]))
        viewport.anim_engine.active_animation = None

    def test_only_first_preset_is_loaded_and_switching_preserves_wardrobe(self):
        self.assertEqual(self.window._checked_nif_paths, {self.keys[0]})
        self.assertEqual(self.window.locked_preset_combo.count(), 3)
        self.window._on_nif_dropped_on_wardrobe(self.keys[3])
        self.window.locked_preset_combo.setCurrentIndex(
            self.window.locked_preset_combo.findData(2)
        )
        self.assertEqual(self.window._checked_nif_paths, {self.keys[1], self.keys[3]})
        self.assertEqual(self.window._locked_presets[1], {self.keys[0]})
        self.assertEqual(self.window._locked_presets[2], {self.keys[1]})
        self.window.locked_preset_combo.setCurrentIndex(
            self.window.locked_preset_combo.findData(1)
        )
        self.assertEqual(self.window._checked_nif_paths, {self.keys[0], self.keys[3]})

    def test_tree_drag_and_empty_preset_drop(self):
        item = self.window.file_tree.topLevelItem(3)
        self.window.file_tree.setCurrentItem(item)
        with patch("src.ui_widgets._start_nif_drag") as drag:
            self.window.file_tree.startDrag(Qt.DropAction.MoveAction)
            drag.assert_called_once_with(self.window.file_tree, self.keys[3])
        self.window._add_locked_preset()
        self.app.processEvents()
        self.assertFalse(self.window.locked_nifs_list.isHidden())
        self.drop_nif(self.window.locked_nifs_list, self.keys[3])
        self.assertIn(self.keys[3], self.window._locked_presets[4])
        self.assertIn(self.keys[3], self.window._checked_nif_paths)

    def test_wardrobe_drop_right_click_and_double_click(self):
        self.drop_nif(self.window.added_nifs_list, self.keys[3])
        item = self.window._added_nif_items[self.keys[3]]
        row = self.window.added_nifs_list.itemWidget(item)
        QTest.mouseClick(row, Qt.MouseButton.RightButton)
        self.assertIn(self.keys[3], self.window._locked_presets[1])
        self.window._on_nif_dropped_on_wardrobe(self.keys[3])
        row = self.window.added_nifs_list.itemWidget(
            self.window._added_nif_items[self.keys[3]]
        )
        QTest.mouseDClick(row, Qt.MouseButton.LeftButton)
        self.assertNotIn(self.keys[3], self.window._checked_nif_paths)

    def test_removing_wardrobe_item_preserves_inactive_preset(self):
        self.window._on_nif_dropped_on_wardrobe(self.keys[1])
        row = self.window.added_nifs_list.itemWidget(
            self.window._added_nif_items[self.keys[1]]
        )
        QTest.mouseDClick(row, Qt.MouseButton.LeftButton)
        self.assertNotIn(self.keys[1], self.window._checked_nif_paths)
        self.assertIn(self.keys[1], self.window._locked_presets[2])

    def test_unlock_last_preset_outfit_loads_next_preset(self):
        self.window._on_nif_dropped_on_wardrobe(self.keys[0])
        self.assertEqual(self.window._current_locked_preset, 2)
        self.assertEqual(self.window._checked_nif_paths, {self.keys[0], self.keys[1]})
        self.assertIn(self.keys[0], self.window._added_nif_items)
        self.assertIn(self.keys[1], self.window._locked_nif_items)

    def test_remove_last_preset_outfit_loads_next_preset(self):
        self.window._on_nif_dropped_on_mods(self.keys[0])
        self.assertEqual(self.window._current_locked_preset, 2)
        self.assertEqual(self.window._checked_nif_paths, {self.keys[1]})
        self.assertIn(self.keys[1], self.window._locked_nif_items)

    def test_same_outfit_can_belong_to_multiple_presets(self):
        self.window._on_nif_dropped_on_preset(self.keys[1])
        self.assertIn(self.keys[1], self.window._locked_presets[1])
        self.assertIn(self.keys[1], self.window._locked_presets[2])
        self.window.locked_preset_combo.setCurrentIndex(
            self.window.locked_preset_combo.findData(2)
        )
        self.assertEqual(self.window._checked_nif_paths, {self.keys[1]})
        self.assertIn(self.keys[1], self.window._locked_nif_items)


if __name__ == "__main__":
    unittest.main()
