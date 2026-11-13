# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve()
NIFLY_ROOT = ROOT / "io_scene_nifly"
HKX_ROOT = NIFLY_ROOT / "hkx"
SMP_PUBLISH = ROOT / "smp_bridge" / "SmpBridge" / "bin" / "Release" / "net8.0" / "win-x64" / "publish"
SMP_LIBRARIES = ("SmpBridge.dll", "libbulletc.dll")
for library in SMP_LIBRARIES:
    if not (SMP_PUBLISH / library).is_file():
        raise SystemExit(
            f"{library} is missing from {SMP_PUBLISH}. "
            "Build it first: dotnet publish smp_bridge/SmpBridge -c Release -r win-x64"
        )
sys.path.insert(0, str(NIFLY_ROOT))
sys.path.insert(0, str(HKX_ROOT))

analysis = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT), str(NIFLY_ROOT), str(HKX_ROOT)],
    binaries=[
        (str(NIFLY_ROOT / "NiflyDLL.dll"), "."),
        *[(str(SMP_PUBLISH / library), "smp_bridge") for library in SMP_LIBRARIES],
    ],
    datas=[
        (str(ROOT / "shaders"), "shaders"),
        (
            str(ROOT / "assets" / "skeleton_female.nif"),
            "assets",
        ),
        (str(ROOT / "assets" / "CBPConfig.json"), "assets"),
        (str(ROOT / "assets" / "CITATION.md"), "assets"),
        (str(ROOT / "locales.json"), "."),
        (str(ROOT / "icon.png"), "."),
    ],
    hiddenimports=collect_submodules("pyn") + ["anim_fo4", "anim_skyrim"],
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
    upx_exclude=["SmpBridge.dll", "libbulletc.dll"],
    name="DibellaWardrobe",
)
