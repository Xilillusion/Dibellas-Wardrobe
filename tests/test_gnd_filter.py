import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.asset_scanner import is_filtered_nif_filename
from src.main_window import DibellasWardrobeWindow
from src.ui_widgets import ROLE_NIF_PATH


class GndFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "data"
        mod = self.root / "example_mod"
        mod.mkdir(parents=True)
        for directory in (self.root, mod):
            for name in ("armor_GND.NIF", "armor_1.nif", "armor_0.nif"):
                (directory / name).touch()
        gnd = mod / "gnd"
        gnd.mkdir()
        (gnd / "ground.nif").touch()
        self.window = DibellasWardrobeWindow(
            initial_root_dir=self.root, config_path=Path(self.temp.name) / "config.json"
        )

    def tearDown(self):
        if self.window._search_thread and self.window._search_thread.isRunning():
            self.window._search_thread.cancel()
            self.window._search_thread.wait()
        self.app.processEvents()
        self.window.close()
        self.temp.cleanup()

    def tree_nifs(self):
        paths = []
        def visit(item):
            path = item.data(0, ROLE_NIF_PATH)
            if path:
                paths.append(Path(path))
            for index in range(item.childCount()):
                visit(item.child(index))
        for index in range(self.window.file_tree.topLevelItemCount()):
            visit(self.window.file_tree.topLevelItem(index))
        return paths

    def test_filename_filter_respects_independent_flags(self):
        self.assertTrue(is_filtered_nif_filename("Armor_GND.NIF", False, True))
        self.assertFalse(is_filtered_nif_filename("Armor_GND.NIF", False, False))
        self.assertTrue(is_filtered_nif_filename("armor_0.nif", True, False))
        self.assertFalse(is_filtered_nif_filename("armor_0.nif", False, True))
        self.assertFalse(is_filtered_nif_filename("armor_gnd_extra.nif", False, True))

    def test_browsing_filters_root_and_nested_files(self):
        for enabled in (True, False):
            self.window.chk_hide_gnd_folder.setChecked(enabled)
            self.window.populate_file_tree()
            mod = next(
                self.window.file_tree.topLevelItem(i)
                for i in range(self.window.file_tree.topLevelItemCount())
                if self.window.file_tree.topLevelItem(i).text(0) == "example_mod"
            )
            mod.setExpanded(True)
            paths = self.tree_nifs()
            for parent in (self.root, self.root / "example_mod"):
                self.assertEqual(parent / "armor_GND.NIF" in paths, not enabled)
                self.assertIn(parent / "armor_1.nif", paths)
                self.assertNotIn(parent / "armor_0.nif", paths)
            self.assertEqual(
                any(mod.child(i).text(0) == "gnd" for i in range(mod.childCount())),
                not enabled,
            )

    def test_search_respects_filter(self):
        for enabled in (True, False):
            self.window.chk_hide_gnd_folder.setChecked(enabled)
            self.window.search_edit.setText("example_mod")
            self.window._search_timer.stop()
            self.window._start_search_worker()
            self.assertTrue(self.window._search_thread.wait(10000))
            self.app.processEvents()
            paths = self.tree_nifs()
            self.assertEqual(self.root / "example_mod" / "armor_GND.NIF" in paths, not enabled)
            self.assertIn(self.root / "example_mod" / "armor_1.nif", paths)
            self.assertNotIn(self.root / "example_mod" / "armor_0.nif", paths)
            self.assertEqual(
                self.root / "example_mod" / "gnd" / "ground.nif" in paths,
                not enabled,
            )
            self.window.search_edit.clear()


if __name__ == "__main__":
    unittest.main()
