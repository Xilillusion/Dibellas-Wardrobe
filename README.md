# Dibella's Wardrobe

Dibella's Wardrobe is a desktop previewer for Skyrim Special Edition and Anniversary Edition outfit meshes. Browse `.nif` files in a mod directory, add them to a live 3D preview, and adjust lighting and display options without launching the game.

## Features

- Preview multiple Skyrim `.nif` meshes together in an interactive OpenGL viewport.
- Play Skyrim `.hkx` animations (`meshes/actors/character/animations/...`) with real-time GPU Linear Blend Skinning driven by `skeleton_female.nif` and a bottom playback toolbar (Play/Pause, Loop, Timeline Scrubber, and Stop/Reset to bind pose).
- Browse a mod directory or Mod Organizer 2 (MO2) `mods` folder, with filters for low-weight meshes, `gnd` folders, and `_gnd.nif` files.
- Resolve material textures from the mesh's mod, nearby folders, registered roots, and MO2 `overwrite`.
- Merge matching MO2 `overwrite/meshes` NIFs into their owning mod in the tree. Overwritten files are shown in brackets; unmatched overwrite meshes appear under `[Overwrite]`.
- Show CharGen-exported NIFs from `overwrite/SKSE/Plugins/CharGen` in the fallback list. Other overwrite folders are not included in the NIF tree.
- Keep selected outfits in the wardrobe, lock outfits to reload them at startup, and hide or show individual outfits.
- Adjust lighting, shader display options, and language in Settings. Translation choices are loaded from `locales.json`.
- Tune screen-space reflections (SSR), ambient occlusion (GTAO), and screen-space subsurface scattering (SSSS) in Settings.
- Adjust CBPC body physics using the viewport's breast-gravity and per-area softness controls.
- Simulate HDT-SMP cloth, hair and capes for outfits that ship an SMP physics XML, using a native Bullet engine. Toggle it with **Enable SMP Physics** above the CBPC controls in the viewport.

## Advanced viewport effects

The Settings panel exposes independent enable switches and quality controls for SSR (step size, maximum ray steps, and thickness), ambient occlusion (radius, intensity, and bias), and SSSS (radius and falloff color). All three effects are disabled by default.

These are screen-space approximations evaluated from the rendered meshes' color and depth buffers. Background pixels are excluded, and the optional reference grid is composited separately so it does not contribute to the effects. Screen-space reflections can only reflect visible on-screen geometry; they do not represent off-screen objects or a room environment.

## Requirements

- Windows (the bundled NIF parser includes `io_scene_nifly/NiflyDLL.dll`).
- Python 3.10 or newer.
- A graphics driver that supports OpenGL 3.3.
- For HDT-SMP cloth physics: `SmpBridge.dll` and `libbulletc.dll`, built from `smp_bridge/` (see [Build the SMP physics library](#build-the-smp-physics-library)). Portable releases already include them. Without them the app still runs; cloth then follows its skeleton parent and the SMP toggle is disabled.

## Install

From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The commands above use Python 3.12; Python 3.10 and newer are supported.
Package managers select releases compatible with the active Python version.

If PowerShell prevents virtual-environment activation, run the Python executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Build the SMP physics library

HDT-SMP cloth physics runs in `smp_bridge/SmpBridge`, a C# project (BulletSharp) compiled with .NET Native AOT into a plain native `SmpBridge.dll` that Python loads with `ctypes`. No .NET runtime is needed to run the app, but building the library needs:

- the .NET 8 SDK, and
- the Visual Studio "Desktop development with C++" build tools (the Native AOT linker).

```powershell
dotnet publish smp_bridge/SmpBridge -c Release -r win-x64
```

The output in `smp_bridge/SmpBridge/bin/Release/net8.0/win-x64/publish/` must contain `SmpBridge.dll` and `libbulletc.dll`; the app finds them there automatically. Set the `SMP_BRIDGE_DIR` environment variable to load them from another folder.

### How SMP physics is applied

For each visible outfit the app reads the XML path stored in the NIF's `HDT Skinned Mesh Physics Object` string extra data (relative to the mod root), falling back to a `[nif_name].xml` next to the NIF. Additional collision XMLs can be listed under `"smp": { "collision_xmls": ["C:\\path\\to\\Collision.xml"] }` in the configuration; they are loaded into the same world before outfit XMLs. Set `"show_collisions": true` in the same section to draw the loaded XML primitives.

Mass-zero and keyframed bodies follow the animated skeleton; positive-mass dynamic bodies use `<generic-constraint>` springs. Explicit sphere, capsule, and box shapes are used for collisions. Capsule `height` denotes total end-to-end height; alternatively `p0` and `p1` define the centers of its caps. Box shapes require explicit `half-extents`. Shapeless bodies have no collision geometry. CBPC does not supply SMP colliders. Shared bones remain active in SMP; when CBPC is active, its result takes precedence for visual deformation.

`<per-vertex-shape>` and `<per-triangle-shape>` resolve skinned NIF shapes by name. Their vertices and triangle plates follow the animated or simulated bones, and contacts are submitted to Bullet using the XML's margins, penetration, weight thresholds and collision filters. An XML loaded as an additional collision XML can resolve shapes from any visible NIF; an outfit XML resolves shapes from its own NIF. Missing meshes are reported in the SMP diagnostics. `VirtualGround` requires a matching NIF shape; it is not an implicit floor. `stiffspring` and `conetwist` constraints and `<pattern>` macros are not simulated.

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

The script publishes the SMP bridge (`dotnet publish`), cleans `build/` and `dist/`, builds with `package.spec`, validates the executable and bundled runtime assets (including `SmpBridge.dll` and `libbulletc.dll`), and writes a portable ZIP under `dist/`. Version is read from `VERSION`, `version.txt`, or the optional `version` field in `config.json`, in that order; if none is present, it defaults to `0.1.0`.

The release starts with default settings rather than including the builder's `config.json`. Settings are saved in the portable distribution when the app closes.

Options:

```text
--version 1.2.3       Override the release version
--skip-zip            Build and validate without creating the ZIP
--no-clean            Keep existing build/ and dist/ directories
--skip-native-build   Reuse the existing SmpBridge.dll instead of running dotnet publish
```

## Using the preview

1. Select a directory using **Browse Folder** (or start with `--root`).
2. Expand folders in the Outfit Meshes tree. Click a `.nif` to add it to or remove it from the preview.
3. Drag outfits from Mods to Wardrobe or the current preset. In Wardrobe, double-click an outfit to remove it or right-click it to add it to the current preset. Drag preset outfits to Wardrobe to unlock them, or drag outfits to Mods to remove them.
4. Adjust camera and rendering settings as needed:
   - Left-drag: orbit
   - Right-drag, or Shift + left-drag: pan
   - Mouse wheel: zoom
   - `R`: reset camera
   - `Ctrl+O`: choose a mod directory
   - `Ctrl+F`: toggle the mod-directory filter

Presets are saved in the configuration. The first preset is loaded at startup, and selecting another preset switches its outfits while retaining other wardrobe outfits. Unavailable preset mesh paths are removed from the configuration.

The CBPC overlay controls breast gravity and separate softness values for breasts, butt and belly.
Each softness slider ranges from 0× (motion off for that area) to 5× (softer, slower response);
1× is the baseline. Breast gravity lowers the breast rest position without changing spring
stiffness, damping, or the motion response; it remains effective at 0× breast softness.
Breast chains use progressively larger rest drops toward the tip and follow the base during
motion. Torso collision does not directly push the tip; external collider contacts take
precedence over chain following.
Higher softness reduces stiffness and damping and expands displacement limits, producing
a slower, less damped response. For non-breast bones, softness also changes gravity.
At 1× the non-breast gravity and displacement-limit multipliers are 1.2. Other physics bones retain their saved
softness independently of these sliders (1× by default). CBPC settings are saved in the
configuration; an existing single softness value is applied to all areas and other bones.
Legacy bounce settings are mapped to softness, but the new scaling does not reproduce their
previous dynamics exactly.

## Configuration and languages

The default configuration file is `config.json` in the project directory. It stores the last-used folder, language, lighting and display settings, CBPC and SMP physics settings, and locked NIF paths. Use `--config` to select another file.

Available languages are read from `locales.json`. Each locale is an object keyed by language code and must include a `language_name` string; translated UI text is defined by the other entries in that object.

## Project files

- `main.py` - application entry point.
- `src/` - user interface, NIF loader, mod scanner, texture resolver, viewport, and the CBPC and SMP physics layers (`smp_bridge.py` is the `ctypes` binding to the native SMP library).
- `smp_bridge/` - C# (.NET 8, BulletSharp, Native AOT) source of the HDT-SMP physics engine, built into `SmpBridge.dll`.
- `io_scene_nifly/` - bundled headless NIF and HKX parsing modules and native DLL.
- `shaders/` - mesh and post-processing GLSL shaders used by the viewport.
- `requirements.txt` - Python dependencies.
- `CITATION.md` - acknowledgements and third-party dependency information.
