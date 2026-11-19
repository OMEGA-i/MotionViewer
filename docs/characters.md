# Adding a downloaded character

Historical import probes covered eighteen character packs. Importing and finding
standard MMD bone names does **not** establish natural motion, valid contacts, or
usable clothing on every character. The current native SOMA review covers Yoimiya,
Furina, Nilou, Yanqing, Liuying's Spring Letter outfit, Qingyi, and coatless Feixiao,
with two complete representative clips per exact asset version.

Body/finger retargeting is shared. Clothing selections, joint limits, collisions,
and presentation calibration require asset-specific verification. Reusable settings
live in `configs/characters/` and are guarded by the PMX SHA-256 and import scale;
see [character physics profiles](character-physics-profiles.md). Replacing a costume
or PMX requires revalidation rather than silently reusing its old profile.

Prefer native `--soma` for the generated dataset so its thirty mapped finger bones
are retained. Some rigs have one fewer torso segment: the mapper preserves the
intermediate rotation in the downstream chest orientation rather than failing to
import, but it cannot recreate the missing spatial bend distribution. These rigs
map 49 source joints; Yoimiya maps 50. Neither count proves equivalent contacts on
different body proportions.

## Unpacking

```bash
uv run python scripts/_extract_characters.py           # ~/Downloads -> assets/fbx/pmx
blender --background --python scripts/_probe_characters.py     # what is in each file
```

`assets/` is git-ignored and stays local; these models carry their own redistribution
terms.

### The encoding trap

These packs store entry names in **GBK**, and a GBK name misread as CP932 still decodes
— into half-width katakana. So a decoder that tries CP932 first *succeeds* and stops,
and `颜.png` becomes `ﾑﾕ.png`. The PMX then asks for the real name, finds nothing, and
the character renders magenta. Eight of nineteen did.

The fix is to prefer GBK and to reject any decode whose non-ASCII characters are
entirely half-width katakana, since that is the signature of this exact failure.
Model names come back too: `ﾄﾝﾂｶ.pmx` is `妮露.pmx`.

## What the probe reports

`REQUIRED_MMD_BONES` is the retarget's own declaration of what it cannot work without,
so the probe checks against that list rather than a second copy of it. A file missing
all seventeen is not a broken character — it is a weapon, which these packs ship as a
separate PMX with a handful of bones and no body.

## Weapons

```bash
blender --background --python scripts/render_mmd_compare.py -- \
    --asset assets/fbx/pmx/星穹铁道—彦卿/星穹铁道—彦卿/彦卿1.0.pmx \
    --motion <sword clip>.npz --weapon auto --toon --output outputs/x
```

`--weapon auto` looks the character up by its asset directory. A blade is aligned along
the **forearm measured off the rig**, not by fixed Euler angles: the angles that express
"pointing along the arm" depend on the rig's bone roll, and MMD rigs do not agree on
it. Hardcoding -95 degrees left a racket standing vertically while the forearm pointed
mostly down and to the side.

## Stages

Community MMD stages are PMX files, and the importer already reads them:

```bash
    --stage path/to/stage.pmx --stage-scale 1.0
```

A stage forces a static camera. A tracking shot slides a world-fixed set across the
frame, which is the single most obvious way to give away a composited environment.

## Deciding whether a character is usable

The criterion is whether a plain standing pose already buries the hands in the costume.
Measuring that took four attempts, and the failures are worth recording because each
looked plausible:

| attempt | result | why it was wrong |
| --- | --- | --- |
| ray-cast hand verts against other mesh objects | 0.0% for all 19 | every model is a **single** mesh with many materials, so there were no other objects to cast against |
| same, on the evaluated mesh | still 0.0% | the deformed-mesh fix was real but the loop still had nothing to test |
| BVH of non-hand faces of the same mesh | real numbers, but Yoimiya at 26.8% | wrists are enclosed by cuffs and gloves **by design** |
| fingers only | Yoimiya still 30.2% | glove vertices split weight between finger and wrist, land on the cloth side, and enclose their own fingers |
| **bind-pose baseline, report the rise** | see the table below | a costume encloses the fingers in the bind pose too, so it cancels; a skirt only closes over the hand once the arm comes down |

Run it with:

```bash
blender --background --python scripts/_check_hand_clipping.py
```

It writes `outputs/hand_clipping.json`.
