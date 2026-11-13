# Dibella's Wardrobe

Dibella's Wardrobe is a desktop previewer for Skyrim Special Edition and Anniversary Edition outfit meshes. Browse `.nif` files in a mod directory, add them to a live 3D preview, and adjust lighting and display options without launching the game.

## Features

- Preview multiple Skyrim `.nif` meshes together in an interactive OpenGL viewport.
- Play Skyrim `.hkx` animations (`meshes/actors/character/animations/...`) with real-time GPU Linear Blend Skinning driven by `skeleton_female.nif` and a bottom playback toolbar (Play/Pause, Loop, Timeline Scrubber, and Stop/Reset to bind pose).
- Browse a mod directory or Mod Organizer 2 (MO2) `mods` folder, with filters for low-weight meshes, `gnd` folders, and `_gnd.nif` files.
- Resolve material textures from the mesh's mod, nearby folders, registered roots, and MO2 `overwrite`.
- Hide mesh shapes whose diffuse texture is missing or unreadable; assigning a valid diffuse texture makes them visible.
- Merge matching MO2 `overwrite/meshes` NIFs and animations into the tree and mod-name search. Overwritten files are shown in brackets; unmatched files appear under `[Overwrite]`. If multiple mods contain the same mesh path, the overwrite is shown in each without choosing a single owner.
- Show CharGen-exported NIFs from `overwrite/SKSE/Plugins/CharGen` in the fallback list. Other overwrite folders are not included in the NIF tree.
- Use the mesh tree refresh button to re-index mods, overwrite assets, and texture paths after changing files.
- Keep selected outfits in the wardrobe, lock outfits to reload them at startup, and hide or show individual outfits.
- Adjust lighting, shader display options, and language in Settings. Translation choices are loaded from `locales.json`.
- Tune SMAA anti-aliasing in Settings; outfit finishes enable ambient occlusion (GTAO) and screen-space subsurface scattering (SSSS) automatically.
- Adjust physics using the settings sidebar's SMP and CBPC controls.
- Simulate HDT-SMP cloth, hair and capes for outfits that ship an SMP physics XML, using a native Bullet engine. Toggle it with **Enable SMP Physics** above the CBPC controls in the viewport.

## Advanced viewport effects

The Settings panel exposes an enable switch and quality control (Low, Medium, High, or Ultra) for SMAA anti-aliasing, which is enabled at High quality by default. Ambient occlusion (GTAO) and SSSS have no manual controls: the selected **Outfit Finish** decides whether they run and how strong they are. **Realistic** and **Standard** enable both with balanced settings, **Glossy** enables both with strong SSSS on skin that runs after its own shading so its highlights stay sharp, **Vivid** enables both with stronger, pinker SSSS on skin after its shading, **Anime** uses light GTAO without SSSS, and **Default** runs neither. A disabled effect's pass is skipped entirely.

GTAO and SSSS are screen-space approximations evaluated from the rendered meshes' color and depth buffers. Background pixels are excluded, and the optional reference grid is composited separately so it does not contribute to them.

SMAA runs last, after any outfit shader preset, and smooths jagged edges on the meshes and the reference grid. It uses the reference [SMAA](https://github.com/iryoku/smaa) implementation by Jimenez et al., included in `shaders/smaa` under its MIT license.

## Outfit shader presets

Choose **Default** (the viewport's own shading) or a discovered finish from **Outfit Finish** under Lighting Controls. **Realistic** relights the outfit from the scene's key light (it follows the Lighting Controls): a warm key with wrapped falloff over a cool sky and warm ground ambient, with skin wrapping further for a warm shadow edge. With lighting disabled it keeps the viewport's shading. It also adds depth-aware contact shading, subtle normal-map cavity shading, normal-aware surface detail, and a slight brightness lift (10%) with a neutral contrast adjustment. When a mesh has them, it also uses its environment mask (`_m.dds`: red reflection, plus Complex Material glossiness/metalness) or tangent normal-map specular alpha for glossy highlights and Fresnel sheen (highlights are broken up by the surface texture and rough materials get only faint edge reflections, so surfaces do not look plastic), and its subsurface map (`_sk.dds`) for soft light diffusion with tinted shadows; meshes without these maps look as before. It enables GTAO (with reduced intensity, since it also adds its own contact shading) and a light SSSS pass, and SMAA continues to apply on top. This is a screen-space finish: it cannot recover material properties or lighting outside the camera view.

**Standard** is a simplified Realistic in about half the code. It keeps the same key-light relighting with warm skin wrap, contact shading, skin subsurface blur, texture-broken highlights with Fresnel sheen, and brightness and contrast curve, but drops the micro-detail passes (normal-map cavity shading, local detail sharpening and highlight widening on noisy normals). It is a good starting point for writing new presets.

**Simple** keeps the viewport's existing material and lighting, applies a subtle light-dependent color adjustment, and uses Vivid's saturation, exposure and contrast grade. It uses exactly Realistic's GTAO and SSSS profile before the finish, so contact shading and screen-space scattering do the bulk of the work. As with Realistic, this SSSS pass affects all surfaces, not just skin.

**Glossy** is **Vivid** plus oil: the classic Skyrim "Korean-style" oiled skin look. Its diffuse is Vivid's (soft two-tone key, smoothed and pink-flushed skin, contact shading, saturated grade) under a wet film, and it keeps Vivid's rim light; Vivid's skin gloss is replaced by oil. Skin (meshes with a `_sk.dds` subsurface map) and glossy materials get dual-lobe GGX specular from the key light: a strong broad oil sheen (roughness 0.40) plus a sharp, over-bright wet highlight (roughness 0.11, 4x). A strong grazing Fresnel term (`pow(1 - N.V, 3.5)`) brightens the wet highlight and reflects the studio cubemap along limbs, shoulders and facial contours, with a faint reflection even head-on. The oil mask is the environment mask red channel (`_m.dds`), else the tangent normal map's alpha. On skin it is floored at 60%, and assumed to be 50% when neither map exists (model-space bodies), so skin always keeps a continuous sheen. Cloth without either map stays matte. Skin keeps 40% of its normal-map detail, as in Vivid, so pores do not break up the highlights. The wet film warms the diffuse by (1.04, 0.99, 0.96), fully on skin and half elsewhere, then darkens and saturates it in proportion to the oil mask. The oil is partly broken up by the surface texture and dimmed inside creases. Glossy writes diffuse and oil separately: the built-in SSSS pass runs after it on skin only (mask 0.85-1.0), and the oil and rim are added afterwards, so subsurface scattering never blurs the wet highlights. `u_gloss_intensity` (oil) and `u_rim_intensity` are initialized in `shaders/glossy.frag`. GTAO runs before the preset. Because it works in screen space, the viewport's own built-in specular is still part of the diffuse it receives.

**Vivid** is **Standard** restyled after the glossy 3D adult renders of Koikatsu, Honey Select and Blender/SFM. Its key light is partly pulled into a soft two-tone ramp, so the terminator reads crisper. Skin keeps only 40% of its normal-map detail for a smooth, doll-like finish, and gets a pink flush, deepest along the terminator. Skin also gets a glossy highlight: a tight white pinpoint over a soft sheen. Lit contours get a bright, slightly pink Fresnel rim light, stronger on skin than on cloth, that separates the body from the background, and colours are a little more saturated and contrasty than Standard. Gloss and rim go to the split specular output, so SSSS does not blur them. `u_gloss_intensity` and `u_rim_intensity` are initialized in `shaders/vivid.frag`.

**Anime** uses a soft two-band cel ramp, cool outfit shadows, rosy shadows on surfaces with subsurface maps, and a luminous lit tone. A gamut-limited saturation boost keeps colors vivid without clipping the albedo; highlights are tight and a cyan-white rim and restrained colored contour separate shapes. Depth-derived geometric normals reduce normal-map detail where available, while light GTAO adds contact shading without blurring the cel edge. The preset's shadow boundary, rim, normal detail, saturation, and gloss are initialized in `shaders/anime.frag`. These are screen-space approximations: the pass cannot identify hair or recover true albedo or vertex normals.

**Copilot** is a luminous studio-illustration finish with a warm key, cool shadow fill, subtle color saturation, soft skin and a cool rim. Glossy materials use their environment mask or tangent-normal alpha, and skin gets a gentle highlight; metallic environment masks tint the highlight. Light GTAO runs before the finish, while skin-masked SSSS runs on its diffuse output and highlights are composited afterward. Like the other screen-space finishes, it estimates albedo and surface shape from the rendered scene rather than recovering the original material or mesh normals.

Presets are discovered from `shaders/*.frag` at startup, excluding `default.frag` and `postprocess.frag`. To add one, create a fragment shader that uses the shared `postprocess.vert` interface (`v_uv`) and the `u_scene_color`, `u_scene_depth`, `u_viewport_size`, and `u_inverse_projection` uniforms. Its filename stem becomes the selector label. An optional `vec3 u_key_light` gives the view-space direction towards the key light, or zero while lighting is disabled. An optional `samplerCube u_studio_cubemap` is the viewport's studio environment, sampled with world-space directions; an optional `mat3 u_view_to_world` rotates view-space directions into that space. An optional `sampler2D u_scene_normal` provides the mesh shader's final view-space shading normal, including normal maps: decode RGB with `rgb * 2.0 - 1.0` and check alpha greater than 0.5 before normalizing. Missing normals have zero alpha; a depth-derived normal can be used as a fallback. Normal and depth buffers represent the depth-writing surface, so translucent layers without depth writes are not represented.

A preset can turn on GTAO and SSSS with an optional sidecar `shaders/<name>.json` next to its `.frag`; without one, neither effect runs. Effects are applied to the scene before the preset shader runs, unless `ssss_after_preset` moves SSSS after it (see below). Omitted fields use the defaults shown here, and values are clamped to the shown ranges:

```json
{
  "postprocess": {
    "enable_gtao": false,
    "gtao_radius": 8.0,
    "gtao_intensity": 1.0,
    "gtao_bias": 0.02,
    "enable_ssss": false,
    "sss_radius": 5.0,
    "sss_strength": 0.35,
    "sss_falloff_color": [1.0, 0.45, 0.35],
    "ssss_after_preset": false
  }
}
```

`gtao_radius` (1?40) is the occlusion search radius in view units, `gtao_intensity` (0?3) scales darkening, and `gtao_bias` (0?0.1) ignores shallow angles. `sss_radius` (1?24) is the blur radius in pixels, `sss_strength` (0?1) is how much of the blurred, `sss_falloff_color`-tinted (0?1 per channel) result is blended in. SSSS is depth-based and applies to all surfaces, not only skin, unless `ssss_after_preset` is set. If the file is malformed or has unknown keys, a warning is logged and the preset runs without either effect.

With `"ssss_after_preset": true`, the preset renders a split output instead of a final color. It writes `layout(location = 0)` diffuse (alpha is kept) and `layout(location = 1)` specular RGB with an SSSS mask in alpha; the specular target is half-float, so values above 1 survive until the final composite. GTAO still runs before the preset. Afterwards, SSSS blurs only masked diffuse, drawing only from masked neighbours, scaled by the mask. Then specular is added unblurred. With `enable_ssss` false, the split output is simply summed. A split preset that does not write location 1 adds no specular and gets no SSSS.

The sidecar configures post-processing only. Preset labels come from the fragment filename, and any preset-specific uniform defaults belong in the fragment shader.

Presets may also declare any of these optional screen-space `sampler2D` inputs: `u_material_normal` (raw `_n.dds` or `_msn.dds` RGBA), `u_material_env_mask` (`_m.dds` RGBA), `u_material_subsurface` (`_sk.dds` RGBA), and `u_material_flags`. The map samples use each mesh's UV coordinates and retain all four texture channels; they are not view-space normals. In `u_material_flags`, R/G/B indicate whether the corresponding normal/environment/subsurface map is available, and A indicates a model-space normal map. Missing maps and background have zero flags and zero samples. Authored NIF texture slots take precedence; if absent, matching suffixes beside the NIF's diffuse texture are looked up using the normal texture search order (including MO2 overwrite). A diffuse `_d` suffix is also supported. As with the normal buffer, non-depth-writing translucent layers are not represented. These material inputs trigger an additional mesh pass only when a preset uses them; existing presets need not declare them.

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

For each visible outfit the app reads the XML path stored in the NIF's `HDT Skinned Mesh Physics Object` string extra data (relative to the mod root), falling back to a `[nif_name].xml` next to the NIF.

Mass-zero and keyframed bodies follow the animated skeleton; positive-mass dynamic bodies use `<generic-constraint>` springs. Explicit sphere, capsule, and box shapes are used for collisions. Capsule `height` denotes total end-to-end height; alternatively `p0` and `p1` define the centers of its caps. Box shapes require explicit `half-extents`. Shapeless bodies have no collision geometry. CBPC does not supply SMP colliders. Shared bones remain active in SMP; when CBPC is active, its result takes precedence for visual deformation.

`<per-vertex-shape>` and `<per-triangle-shape>` resolve skinned NIF shapes by name. Their vertices and triangle plates follow the animated or simulated bones, and contacts are submitted to Bullet using the XML's margins, penetration, weight thresholds and collision filters. An outfit XML resolves shapes from its own NIF. Missing meshes are reported in the SMP diagnostics, except for `VirtualGround`. `VirtualGround` requires a matching NIF shape; it is not an implicit floor. `stiffspring` and `conetwist` constraints and `<pattern>` macros are not simulated.

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
3. Drag outfits from Mods to Wardrobe or the current preset. In Wardrobe, double-click an outfit to remove it or right-click it to add it to the current preset. Drag preset outfits to Wardrobe to unlock them, or drop wardrobe and preset outfits anywhere on the Mods panel to remove them from the preview and presets without deleting files.
4. Adjust camera and rendering settings as needed:
   - Left-drag: orbit
   - Right-drag, or Shift + left-drag: pan
   - Mouse wheel: zoom
   - `R`: reset camera
   - `Ctrl+O`: choose a mod directory
   - `Ctrl+F`: toggle the mod-directory filter

Presets are saved in the configuration and remain available when empty. The first preset is loaded at startup, and selecting another preset switches its outfits while retaining other wardrobe outfits. Unavailable preset mesh paths are removed from the configuration.
Each preset also saves the resolved diffuse texture path for each mesh shape, including selected
replacements; other texture maps are not stored. Presets sharing a mesh can use different diffuse
textures. Missing saved textures are logged and the mesh's default diffuse texture is used.

Open the settings sidebar with the gear icon to access SMP and CBPC physics controls. The CBPC
controls include breast gravity and separate softness values for breasts, butt, belly and thighs.
Each softness slider ranges from 0× (motion off for that area) to 10× (softer, slower response) in 0.5× steps;
existing values between steps remain unchanged until the slider is adjusted.
1× is the baseline. Breast gravity lowers the breast rest position without changing spring
stiffness, damping, or the motion response; it remains effective at 0× breast softness.
Breast chains use progressively larger rest drops toward the tip and follow the base during
motion. Torso collision does not directly push the tip; external collider contacts take
precedence over chain following.
Higher softness reduces stiffness and damping and expands displacement limits, producing
a slower, less damped response with longer-lasting bounce. For non-breast bones, softness also changes gravity.
At 1× the non-breast gravity and displacement-limit multipliers are 1.2. Other physics bones retain their saved
softness independently of these sliders (1× by default). CBPC settings are saved in the
configuration; an existing single softness value is applied to all areas and other bones.
Existing configurations without a thigh value inherit their saved `other` softness for thighs.
Legacy bounce settings are mapped to softness, but the new scaling does not reproduce their
previous dynamics exactly.

## Configuration and languages

The default configuration file is `config.json` in the project directory. It stores the last-used folder, language, lighting and display settings, CBPC and SMP physics settings, and locked NIF presets. Older `locked_nifs` lists are migrated to a preset when the preset mapping is absent or invalid. Use `--config` to select another file.

Available languages are read from `locales.json`. Each locale is an object keyed by language code and must include a `language_name` string; translated UI text is defined by the other entries in that object.

## Project files

- `main.py` - application entry point.
- `src/` - user interface, NIF loader, mod scanner, texture resolver, viewport, and the CBPC and SMP physics layers (`smp_bridge.py` is the `ctypes` binding to the native SMP library).
- `smp_bridge/` - C# (.NET 8, BulletSharp, Native AOT) source of the HDT-SMP physics engine, built into `SmpBridge.dll`.
- `io_scene_nifly/` - bundled headless NIF and HKX parsing modules and native DLL.
- `shaders/` - mesh and post-processing GLSL shaders used by the viewport.
- `requirements.txt` - Python dependencies.
- `CITATION.md` - acknowledgements and third-party dependency information.
