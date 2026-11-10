## CBPConfig.json: Provenance & Compilation

`CBPConfig.json` is a compiled, pre-averaged form of the **CBPC (CBP Physics with Collision)** configuration files used by Skyrim's CBPC SKSE plugin and the 3BA/3BBB body setup. The app reads only this JSON at runtime; the original `.txt` files are not shipped or parsed.

### Source files

| Source file | Compiled into |
|---|---|
| `CBPConfig_3b.txt` (ExtraBreast1-3 L/R) | `bones` |
| `CBPConfig_butt.txt` (LButt / RButt) | `bones` |
| `CBPConfig_belly.txt` (Belly) | `bones` |
| `CBPConfig_leg.txt` (Front/RearThigh, RearCalf) | `bones` |
| `CBPConfig_Vagina.txt` (VaginaB, Clit, Labia) | `bones` |
| `CBPConfig_3b_MoreGravity.txt` | merged into `bones` (it is the gravity base) |
| `CBPConfig_3b_armor.txt`, `CBPConfig_3b_Gravity.txt` | not compiled; replaced by the gravity slider (rule 2) |
| `CBPCollisionConfig_Female.txt` | `collision` |

The original format is CBPC's: `Bone.attribute value_at_weight_0 value_at_weight_100`, with `[AffectedNodes]`, `[ColliderNodes]` and per-node collision sections for the collision file.

### Compilation rules

1. **Bones:** the five body files are merged attribute by attribute (later files override earlier ones) into one flat `bones` map. There is no `base` / `presets` split.
2. **More Gravity is the base:** the `More Gravity` patch (`gravityInvertedCorrection`) is merged into `bones`. `Armor` (`amplitude` / `pushup` 0.8) and `Gravity` (all zero) are not compiled. A runtime breast-gravity slider scales the compiled breast gravity values to determine their downward rest-position offsets; it does not change softness. Butt and belly gravity use a fixed factor of 1/3:

| Slider factor | Stop | Effective gravity |
|---|---|---|
| 0.00 | Armor (minimum) | `gravity` x 0 |
| 0.33 | Default | `gravity` x 1/3 |
| 0.67 | Gravity | `gravity` x 2/3 |
| 1.00 | More Gravity (the compiled config) | `gravity` x 1 |
| 2.00 | Even More Gravity | `gravity` x 2 |
| 5.00 | Maximum Gravity | `gravity` x 5 |

   The stop names and intermediate factors are app-defined, not direct CBPC preset values.
3. **Body weight:** each pair of values is linearly blended at weight **0.5** into a single number.
4. **Bone names:** CBPC config names are renamed to skeleton node names (`ExtraBreast1L` -> `NPC L Breast01`, `LButt` -> `NPC L Butt`, `Belly` -> `NPC Belly`, `LLabia` -> `NPC L Pussy02`, `Clit` -> `Clitoral1`, ...).
5. **Attributes reduced to solver scalars** (a bone only gets a key if its source sets the matching attributes):

| JSON key | Derived from |
|---|---|
| `stiffness` | `stiffness`, else mean of `stiffnessX/Y/Z` |
| `stiffness2` | `stiffness2`, else mean of `stiffness2X/Y/Z` |
| `damping` | `damping`, else mean of `dampingX/Y/Z` |
| `gravity` | breasts: `-gravityInvertedCorrection * 0.002` (More Gravity patch); butt and belly: app-defined `0.008` and `0.01`, because the sources define no gravity for them |
| `gravityInvertedCorrection` | same-named attribute (breasts only, kept as provenance; the solver ignores it) |
| `maxoffset` | `maxoffset`, else mean of `\|Xmaxoffset\|`, `\|Ymaxoffset\|`, `\|Zmaxoffset\|` |
| `collisionMaxoffset` | mean of `\|collisionX/Y/Zmaxoffset\|` |

   Attributes not listed (`amplitude`, `pushup`, rotational springs, `linear*`, `timeStep`, `timetick`, `collisionFriction`, etc.) are dropped; the solver does not model them. Values are rounded to 6 decimals.
6. **Collision:** `collision.colliders` holds the section shapes (spheres and capsules) in node-local space, blended at weight 0.5 and rounded to 4 decimals. `collision.collider_nodes` identifies nodes declared under `[ColliderNodes]`; the other shape sections describe affected nodes, not active colliders. `collision.affected` maps each affected bone to its `@`-prefixed self-collider exclusions. Unprefixed exclusions apply to other actors in CBPC and are not self-collider exclusions; this viewer simulates one actor. Node keys are normalised: lower-case, without the `NPC ` prefix, `@`, or the `[Tag]` suffix (`NPC L Thigh [LThg]` -> `l thigh`).

The solver transforms affected and collider shapes with their animated bones. It retains any
overlap present when collision tracking starts, but resolves additional overlap as colliders
move; reset or timeline scrubbing starts a new collision reference.

The conversion from these CBPC-native values to simulation units is done in `src/cbpc_solver.py` (`STIFFNESS_SCALE`, `DAMPING_SCALE`, `GRAVITY_SCALE`).

### Credits

CBPC is by **Fotogen** and contributors (CBP Physics with Collisions, a Skyrim SKSE plugin). The configuration values are derived from the CBPC / 3BA configuration files distributed with those mods; all rights remain with their original authors. Check the original mod pages for their license terms before redistributing the source `.txt` files.
