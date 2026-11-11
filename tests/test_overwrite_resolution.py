import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.asset_scanner import NifSearchThread, get_mod_name_for_nif, resolve_mod_root_for_nif
from src.main_window import DibellasWardrobeWindow
from src.texture_resolver import TextureResolver
from src.ui_widgets import ROLE_NIF_PATH


class OverwriteResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mods = self.root / "mods"
        self.overwrite = self.root / "overwrite"
        self.original = self.make_file("mods/Example/meshes/armor/dress.nif")
        self.override = self.make_file("overwrite/meshes/armor/dress.nif")
        self.orphan = self.make_file("overwrite/meshes/other/orphan.nif")
        (self.mods / "Empty" / "meshes" / "empty").mkdir(parents=True)
        self.overlay_only = self.make_file("overwrite/meshes/empty/added.nif")
        self.window = DibellasWardrobeWindow(
            initial_root_dir=self.root, config_path=self.root / "config.json"
        )

    def tearDown(self):
        if self.window._search_thread and self.window._search_thread.isRunning():
            self.window._search_thread.cancel()
            self.window._search_thread.wait()
        self.window.close()
        self.app.processEvents()
        self.temp.cleanup()

    def make_file(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path.resolve()

    def search(self, query):
        found = []
        worker = NifSearchThread(
            self.mods, query, True, True, True, self.window.mod_scanner
        )
        worker.matchFound.connect(lambda *args: found.append(args))
        worker.run()
        return found

    def find_tree_paths(self):
        found = []

        def visit(item):
            path = item.data(0, ROLE_NIF_PATH)
            if path:
                found.append(Path(path))
            if item.childCount() and item.data(0, ROLE_NIF_PATH) is None:
                item.setExpanded(True)
                self.window._on_tree_item_expanded(item)
            for i in range(item.childCount()):
                visit(item.child(i))

        for i in range(self.window.file_tree.topLevelItemCount()):
            visit(self.window.file_tree.topLevelItem(i))
        return found

    def test_search_includes_replacements_and_unmatched_files(self):
        result = self.search("Example")
        self.assertEqual([(group, name, Path(path)) for group, _, name, path in result],
                         [("Example", "[dress.nif]", self.override)])
        self.assertEqual([Path(item[3]) for item in self.search("Empty")],
                         [self.overlay_only])
        self.assertEqual([Path(item[3]) for item in self.search("Overwrite")],
                         [self.orphan])

    def test_tree_includes_overlay_only_mod_and_fallback(self):
        paths = self.find_tree_paths()
        self.assertIn(self.override, paths)
        self.assertNotIn(self.original, paths)
        self.assertIn(self.overlay_only, paths)
        self.assertIn(self.orphan, paths)

    def test_conflicting_mods_show_overwrite_in_each_tree_and_search(self):
        other = self.make_file("mods/Other/meshes/armor/dress.nif")
        self.window._refresh_file_tree()
        scanner = self.window.mod_scanner
        self.assertIsNone(scanner.owner_for_nif(self.override))
        self.assertTrue(scanner.is_overwrite_nif(self.override))
        self.assertEqual(get_mod_name_for_nif(str(self.override), scanner, self.mods), "[Overwrite]")
        self.assertEqual(resolve_mod_root_for_nif(str(self.override), scanner, self.mods),
                         self.overwrite)
        paths = self.find_tree_paths()
        self.assertEqual(paths.count(self.override), 2)
        self.assertNotIn(self.original, paths)
        self.assertNotIn(other, paths)
        for group in ("Example", "Other"):
            self.assertEqual([Path(item[3]) for item in self.search(group)],
                             [self.override])

    def test_refresh_reindexes_tree_search_and_textures(self):
        resolver = self.window.resolver
        texture_path = "textures/armor/dress.dds"
        self.assertIsNone(resolver.resolve_path(texture_path, self.original))
        added = self.make_file("overwrite/meshes/armor/new.nif")
        texture = self.make_file("overwrite/textures/armor/dress.dds")
        self.assertFalse(self.window.mod_scanner.is_overwrite_nif(added))
        self.window.search_edit.setText("Example")
        self.window._search_timer.stop()
        self.window.btn_refresh_tree.click()
        self.assertTrue(self.window._search_thread.wait(10000))
        self.app.processEvents()
        self.assertIn(added, self.find_tree_paths())
        self.assertEqual(resolver.resolve_path(texture_path, self.original), texture)

    def test_overwrite_search_respects_filters_and_includes_animations(self):
        hidden_weight = self.make_file("overwrite/meshes/armor/dress_0.nif")
        hidden_ground = self.make_file("overwrite/meshes/armor/gnd/ground.nif")
        anim = self.make_file("overwrite/meshes/actors/animations/walk.hkx")
        self.window._refresh_file_tree()
        results = [Path(item[3]) for item in self.search("Overwrite")]
        self.assertNotIn(hidden_ground, results)
        self.assertIn(anim, results)
        self.assertNotIn(hidden_weight, [Path(item[3]) for item in self.search("Example")])

    def test_chargen_files_with_same_name_are_not_omitted(self):
        first = self.make_file("overwrite/SKSE/Plugins/CharGen/first/head.nif")
        second = self.make_file("overwrite/SKSE/Plugins/CharGen/second/head.nif")
        self.window._refresh_file_tree()
        paths = self.find_tree_paths()
        self.assertIn(first, paths)
        self.assertIn(second, paths)
        results = [Path(item[3]) for item in self.search("Overwrite")]
        self.assertIn(first, results)
        self.assertIn(second, results)

    def test_texture_overwrite_precedes_mod_texture(self):
        mod_texture = self.make_file("mods/Example/textures/armor/dress.dds")
        overwrite_texture = self.make_file("overwrite/textures/armor/dress.dds")
        resolver = TextureResolver([self.mods], self.overwrite)
        self.assertNotEqual(mod_texture, overwrite_texture)
        self.assertEqual(
            resolver.resolve_path("textures/armor/dress.dds", self.original),
            overwrite_texture,
        )


if __name__ == "__main__":
    unittest.main()