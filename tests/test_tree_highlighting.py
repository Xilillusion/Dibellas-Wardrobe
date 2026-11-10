import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.main_window import DibellasWardrobeWindow
from src.ui_widgets import ROLE_NIF_PATH


class TreeHighlightingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name) / "data"
        folder = root / "example_mod" / "meshes"
        folder.mkdir(parents=True)
        self.first = folder / "first.nif"
        self.second = folder / "second.nif"
        self.first.touch()
        self.second.touch()
        self.window = DibellasWardrobeWindow(
            initial_root_dir=root, config_path=Path(self.temp.name) / "config.json"
        )

    def tearDown(self):
        if self.window._search_thread and self.window._search_thread.isRunning():
            self.window._search_thread.cancel()
            self.window._search_thread.wait()
        self.app.processEvents()
        self.window.close()
        self.temp.cleanup()

    def assert_selected(self, path):
        current = self.window.file_tree.currentItem()
        self.assertIsNotNone(current)
        self.assertEqual(Path(current.data(0, ROLE_NIF_PATH)), path)
        self.assertEqual(current.background(0).color().name(), "#5b3b24")

    def test_select_after_refresh_and_switch_selection(self):
        self.window._select_nif_in_tree(str(self.first))
        self.assert_selected(self.first)
        self.window.populate_file_tree()
        self.assertFalse(self.window._highlighted_tree_items)
        self.window._select_nif_in_tree(str(self.first))
        previous = self.window.file_tree.currentItem()
        self.window._select_nif_in_tree(str(self.second))
        self.assert_selected(self.second)
        self.assertEqual(previous.background(0).style().value, 0)

    def test_select_after_search_rebuild(self):
        self.window._select_nif_in_tree(str(self.first))
        self.window.search_edit.setText("example_mod")
        self.window._search_timer.stop()
        self.window._start_search_worker()
        self.assertFalse(self.window._highlighted_tree_items)
        self.assertTrue(self.window._search_thread.wait(10000))
        self.app.processEvents()
        self.window._select_nif_in_tree(str(self.second))
        self.assert_selected(self.second)

    def test_search_expands_single_folder_chains_after_branch(self):
        root = self.first.parent.parent
        for branch in ("armor", "clothes"):
            directory = root / branch / "nested" / "deeper"
            directory.mkdir(parents=True)
            (directory / "outfit.nif").touch()

        self.window.search_edit.setText("example_mod")
        self.window._search_timer.stop()
        self.window._start_search_worker()
        self.assertTrue(self.window._search_thread.wait(10000))
        self.app.processEvents()

        group = self.window.file_tree.topLevelItem(0)
        self.assertEqual(group.childCount(), 3)
        group.setExpanded(True)
        branch = next(
            group.child(i)
            for i in range(group.childCount())
            if group.child(i).text(0) == "armor"
        )
        self.assertFalse(branch.isExpanded())
        branch.setExpanded(True)
        nested = branch.child(0)
        deeper = nested.child(0)
        self.assertTrue(nested.isExpanded())
        self.assertTrue(deeper.isExpanded())
        self.assertEqual(deeper.child(0).text(0), "outfit.nif")

    def test_search_does_not_expand_past_files_or_folder_branches(self):
        root = self.first.parent.parent
        branch = root / "armor"
        nested = branch / "nested"
        nested.mkdir(parents=True)
        (branch / "outfit.nif").touch()
        (nested / "outfit.nif").touch()

        self.window.search_edit.setText("example_mod")
        self.window._search_timer.stop()
        self.window._start_search_worker()
        self.assertTrue(self.window._search_thread.wait(10000))
        self.app.processEvents()

        group = self.window.file_tree.topLevelItem(0)
        group.setExpanded(True)
        branch_item = next(
            group.child(i)
            for i in range(group.childCount())
            if group.child(i).text(0) == "armor"
        )
        branch_item.setExpanded(True)
        nested_item = next(
            branch_item.child(i)
            for i in range(branch_item.childCount())
            if branch_item.child(i).text(0) == "nested"
        )
        self.assertFalse(nested_item.isExpanded())

    def test_search_auto_expands_root_chain_on_completion(self):
        self.window.search_edit.setText("example_mod")
        self.window._search_timer.stop()
        self.window._start_search_worker()
        self.assertTrue(self.window._search_thread.wait(10000))
        self.app.processEvents()

        group = self.window.file_tree.topLevelItem(0)
        self.assertEqual(group.childCount(), 1)
        self.assertTrue(group.child(0).isExpanded())


if __name__ == "__main__":
    unittest.main()
