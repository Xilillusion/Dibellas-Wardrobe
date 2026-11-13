"""
Dibella's Wardrobe â€” Standalone Skyrim SE/AE Outfit Previewer

Usage:
    D:/Python/python.exe main.py [--root <path_to_mods_folder>] [--body <path_to_femalebody_1.nif>]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from src.main_window import DibellasWardrobeWindow


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Dibella's Wardrobe - No CTD Outfit Previewer"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="",
        help="Initial root directory to scan for Skyrim .nif meshes.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional path to config.json file.",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Dibella's Wardrobe")
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    icon = QIcon(str(base_dir / "dibellas-wardrobe.ico"))
    app.setWindowIcon(icon)

    window = DibellasWardrobeWindow(
        initial_root_dir=Path(args.root) if args.root else None,
        config_path=Path(args.config) if args.config else None,
    )
    window.showMaximized()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
