# Dibella's Wardrobe

Dibella's Wardrobe is a desktop previewer for Skyrim Special Edition and Anniversary Edition outfit meshes. Browse `.nif` files in a mod directory, add them to a live 3D preview, and adjust lighting and display options without launching the game.

## Features

- Preview multiple Skyrim `.nif` meshes together in an interactive OpenGL viewport.
- Browse a mod directory or Mod Organizer 2 (MO2) `mods` folder, with filters for low-weight meshes and `gnd` folders.
- Resolve material textures from the mesh's mod, nearby folders, registered roots, and MO2 `overwrite`.
- Merge matching MO2 `overwrite/meshes` NIFs into their owning mod in the tree. Overwritten files are shown in brackets; unmatched overwrite meshes appear under `[Overwrite]`.
- Show CharGen-exported NIFs from `overwrite/SKSE/Plugins/CharGen` in the fallback list. Other overwrite folders are not included in the NIF tree.
- Keep selected outfits in the wardrobe, lock outfits to reload them at startup, and hide or show individual outfits.
- Adjust lighting, shader display options, and language in Settings. Translation choices are loaded from `locales.json`.
- Tune screen-space reflections (SSR), ambient occlusion (GTAO), and screen-space subsurface scattering (SSSS) in Settings.

## Advanced viewport effects

The Settings panel exposes independent enable switches and quality controls for SSR (step size, maximum ray steps, and thickness), ambient occlusion (radius, intensity, and bias), and SSSS (radius and falloff color). All three effects are disabled by default.

These are screen-space approximations evaluated from the rendered meshes' color and depth buffers. Background pixels are excluded, and the optional reference grid is composited separately so it does not contribute to the effects. Screen-space reflections can only reflect visible on-screen geometry; they do not represent off-screen objects or a room environment.

## Requirements

- Windows (the bundled NIF parser includes `io_scene_nifly/NiflyDLL.dll`).
- Python 3.10 or newer.
- A graphics driver that supports OpenGL 3.3.

## Install

From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, run the Python executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

Start the application and choose a directory using the folder button:

```powershell
python main.py
```

Or provide an initial directory from the command line:

```powershell
python main.py --root "E:\Skyrim\MO2\mods"
```

You can provide either an MO2 `mods` folder, the MO2 instance root containing both `mods` and `overwrite`, or another directory containing Skyrim meshes. When an MO2 instance root is supplied, the app uses its `mods` directory and recognizes its `overwrite` directory.

To use a configuration file at a different location:

```powershell
python main.py --config "C:\path\to\config.json"
```

## Build a portable release

Install PyInstaller into the Python environment used to build the app:

```powershell
python -m pip install pyinstaller
```

Then run the release script from the project directory:

```powershell
python build_release.py
```

The script cleans `build/` and `dist/`, builds with `package.spec`, validates the executable and bundled runtime assets, and writes a portable ZIP under `dist/`. Version is read from `VERSION`, `version.txt`, or the optional `version` field in `config.json`, in that order; if none is present, it defaults to `0.1.0`.

Options:

```text
--version 1.2.3   Override the release version
--skip-zip        Build and validate without creating the ZIP
--no-clean        Keep existing build/ and dist/ directories
```

## Using the preview

1. Select a directory using **Browse Folder** (or start with `--root`).
2. Expand folders in the Outfit Meshes tree. Click a `.nif` to add it to or remove it from the preview.
3. Use the Wardrobe list to select an outfit in the tree, toggle its visibility, or clear unlocked outfits. Double-click an outfit to lock or unlock it.
4. Adjust camera and rendering settings as needed:
   - Left-drag: orbit
   - Right-drag, or Shift + left-drag: pan
   - Mouse wheel: zoom
   - `R`: reset camera
   - `Ctrl+O`: choose a mod directory
   - `Ctrl+F`: toggle the mod-directory filter

Locked outfits are saved in the configuration and loaded when the application starts. Unavailable locked mesh paths are removed from the configuration.

## Configuration and languages

The default configuration file is `config.json` in the project directory. It stores the last-used folder, language, lighting and display settings, and locked NIF paths. Use `--config` to select another file.

Available languages are read from `locales.json`. Each locale is an object keyed by language code and must include a `language_name` string; translated UI text is defined by the other entries in that object.

## Project files

- `main.py` - application entry point.
- `src/` - user interface, NIF loader, mod scanner, texture resolver, and viewport.
- `io_scene_nifly/` - bundled NIF parsing library and native DLL.
- `shaders/` - mesh and post-processing GLSL shaders used by the viewport.
- `requirements.txt` - Python dependencies.
- `CITATION.md` - acknowledgements and third-party dependency information.
