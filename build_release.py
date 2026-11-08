"""Build and validate a portable Windows release of Dibella's Wardrobe."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence


ROOT = Path(__file__).resolve().parent
APP_NAME = "DibellaWardrobe"
SPEC_PATH = ROOT / "package.spec"
DEFAULT_VERSION = "0.1.0"
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


class BuildError(RuntimeError):
    """Raised when a release prerequisite, build, or validation fails."""


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and package a portable Windows release."
    )
    parser.add_argument(
        "--skip-zip",
        action="store_true",
        help="Build and validate the application without creating a ZIP archive.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep existing build/ and dist/ directories.",
    )
    parser.add_argument(
        "--version",
        help="Override the release version (must be semantic versioning).",
    )
    args = parser.parse_args(argv)
    if args.version and not SEMVER_PATTERN.fullmatch(args.version):
        parser.error("--version must be a semantic version (for example 1.2.3)")
    return args


def _validate_version(version: str) -> bool:
    return SEMVER_PATTERN.fullmatch(version) is not None


def read_version() -> str:
    for version_file in (ROOT / "VERSION", ROOT / "version.txt"):
        if not version_file.is_file():
            continue
        version = version_file.read_text(encoding="utf-8").strip()
        if _validate_version(version):
            return version
        log(f"Ignoring invalid semantic version in {version_file}")

    config_path = ROOT / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log(f"Could not read version from {config_path}: {exc}")
        else:
            version = config.get("version") if isinstance(config, dict) else None
            if isinstance(version, str) and _validate_version(version):
                return version
            if version is not None:
                log(f"Ignoring invalid semantic version in {config_path}")

    return DEFAULT_VERSION


def validate_inputs() -> None:
    required_files = (
        ROOT / "main.py",
        ROOT / "io_scene_nifly" / "NiflyDLL.dll",
        ROOT / "locales.json",
        ROOT / "config.json",
    )
    missing = [path for path in required_files if not path.is_file()]
    if not (ROOT / "shaders").is_dir():
        missing.append(ROOT / "shaders")
    if not any(path.is_file() for path in (ROOT / "main.py", SPEC_PATH)):
        missing.append(ROOT / "main.py or package.spec")
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise BuildError(f"Required build inputs are missing:\n{formatted}")

    required_shaders = (
        "default.vert",
        "default.frag",
        "postprocess.vert",
        "postprocess.frag",
    )
    missing_shaders = [
        ROOT / "shaders" / name
        for name in required_shaders
        if not (ROOT / "shaders" / name).is_file()
    ]
    if missing_shaders:
        formatted = "\n".join(f"  - {path}" for path in missing_shaders)
        raise BuildError(f"Required shader assets are missing:\n{formatted}")

    if not SPEC_PATH.is_file():
        raise BuildError(f"PyInstaller spec file is missing: {SPEC_PATH}")


def _remove_readonly_and_retry(
    function, path: str, _error_info: object
) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        function(path)
    except OSError as exc:
        raise BuildError(f"Could not remove {path}: {exc}") from exc


def clean_build_directories() -> None:
    for name in ("build", "dist"):
        target = ROOT / name
        if target.is_symlink():
            raise BuildError(f"Refusing to clean symlinked build path: {target}")
        if not target.exists():
            continue
        if target.parent != ROOT or target.name not in {"build", "dist"}:
            raise BuildError(f"Refusing to clean unexpected path: {target}")
        log(f"Removing {target}")
        try:
            shutil.rmtree(target, onerror=_remove_readonly_and_retry)
        except OSError as exc:
            raise BuildError(f"Could not remove {target}: {exc}") from exc


def ensure_pyinstaller_available() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--version"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise BuildError(
            "PyInstaller is not installed in the active Python environment.\n"
            f"Install it with:\n  \"{sys.executable}\" -m pip install pyinstaller"
        )
    log(f"Using PyInstaller {result.stdout.strip()}")


def run_pyinstaller() -> None:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--distpath",
        str(ROOT / "dist"),
        "--workpath",
        str(ROOT / "build"),
        str(SPEC_PATH),
    ]
    log("Running PyInstaller")
    log("Command: " + subprocess.list2cmdline(command))
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise BuildError(f"Could not start PyInstaller: {exc}") from exc

    assert process.stdout is not None
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise BuildError("Build interrupted by user.") from None
    if return_code != 0:
        raise BuildError(f"PyInstaller failed with exit code {return_code}.")


def validate_output() -> Path:
    distribution = ROOT / "dist" / APP_NAME
    executable = distribution / f"{APP_NAME}.exe"
    if not executable.is_file():
        raise BuildError(f"Expected executable was not created: {executable}")

    expected_shaders = {
        "default.vert",
        "default.frag",
        "postprocess.vert",
        "postprocess.frag",
    }
    found_shader_sets = [
        {
            path.name
            for path in directory.iterdir()
            if path.is_file()
        }
        for shader_file in distribution.rglob("default.vert")
        for directory in (shader_file.parent,)
        if (directory / "default.frag").is_file()
    ]
    if not any(expected_shaders.issubset(files) for files in found_shader_sets):
        raise BuildError(
            f"One or more required shaders are missing from {distribution}."
        )

    if not any(
        path.name.casefold() == "niflydll.dll"
        for path in distribution.rglob("*")
        if path.is_file()
    ):
        raise BuildError(f"NiflyDLL.dll is missing from {distribution}.")

    if not any(
        path.name.casefold() == "qwindows.dll"
        for path in distribution.rglob("*")
        if path.is_file()
    ):
        raise BuildError(f"The Qt Windows platform plugin is missing from {distribution}.")

    log(f"Validated executable and runtime dependencies in {distribution}")
    return distribution


def create_portable_zip(distribution: Path, version: str) -> Path:
    archive_path = ROOT / "dist" / (
        f"{APP_NAME}-v{version}-Windows-x64-Portable.zip"
    )
    log(f"Creating portable archive {archive_path}")
    try:
        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for path in sorted(distribution.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        Path(distribution.name) / path.relative_to(distribution),
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        raise BuildError(f"Could not create portable archive: {exc}") from exc

    with zipfile.ZipFile(archive_path, "r") as archive:
        expected_root = distribution.name + "/"
        if not archive.namelist() or not all(
            name.startswith(expected_root) for name in archive.namelist()
        ):
            raise BuildError(f"Archive does not contain a single root folder: {archive_path}")

    return archive_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    started = time.monotonic()
    args = parse_args(argv)
    version = args.version or read_version()
    log(f"Starting portable release build v{version}")
    try:
        validate_inputs()
        ensure_pyinstaller_available()
        if not args.no_clean:
            clean_build_directories()
        else:
            log("Keeping existing build/ and dist/ directories")
        run_pyinstaller()
        distribution = validate_output()
        if args.skip_zip:
            log(f"Build completed; portable ZIP skipped. Distribution: {distribution}")
        else:
            archive = create_portable_zip(distribution, version)
            log(f"Portable package: {archive.resolve()}")
        log(f"Completed in {time.monotonic() - started:.1f} seconds")
        return 0
    except BuildError as exc:
        log(f"ERROR: {exc}")
        log(f"Failed after {time.monotonic() - started:.1f} seconds")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
