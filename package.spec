# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve()
NIFLY_ROOT = ROOT / "io_scene_nifly"
sys.path.insert(0, str(NIFLY_ROOT))

analysis = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT), str(NIFLY_ROOT)],
    binaries=[
        (str(NIFLY_ROOT / "NiflyDLL.dll"), "."),
    ],
    datas=[
        (str(ROOT / "shaders"), "shaders"),
        (str(ROOT / "locales.json"), "."),
        (str(ROOT / "config.json"), "."),
    ],
    hiddenimports=collect_submodules("pyn"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="DibellaWardrobe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

distribution = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="DibellaWardrobe",
)
