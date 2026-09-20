# Measurement records

Every number a design decision in this project rests on, as the probes wrote it. These
are the only copies: `outputs/` holds regenerable renders and is cleared periodically,
while re-running some of these needs a Blender session per character.

| file | what it settled |
| --- | --- |
| `character_probe.json` | the 19 usable characters — bone counts, heights, and which required MMD bones each file is missing (weapon PMX files miss all of them, which is how they are told apart from characters) |
| `hand_clipping.json` | whether a plain standing pose buries the hands. Reports the *rise* from bind pose to standing, because a glove encloses the fingers in both and cancels; the absolute figure flagged Yoimiya, which has never clipped |
| `skirt_physics.json` | how much of each character's mesh a dynamic rigid body drives. Read the caveat below before trusting the lower-body column |
| `mmd_validation*.json` | retarget error per bone; `_soma` is the SOMA-77 path, `_polished` has the polish pass on |
| `rig_dump*.json` | full bone trees, which is where the Japanese bone names in the mapping tables came from |
| `morphs_*.json` | available shape keys per character — the source for the expression presets |
| `materials_yoimiya.json` | the cel-shading inputs: ramp texture, sphere map, per-material edge colour |
| `motion_scores.json` | foot skate, jitter and ground penetration for the whole dataset |
| `diag_turn.json` | turn angular velocity, used to set the hard gate at the ground truth p90 |

## Caveat on `skirt_physics.json`

The `lower_share` column is not trustworthy. It selects "skirt vertices" by model-local
Z in (-0.5, 0.35), and these models do not put their origin at the waist — the band
actually lands on the shins and feet, which is visible in the dominant bones it reports
(`足首D`, `ひざD`). The `mesh_share` column is fine.

To fix it, select by the model's own skirt bones instead: Yoimiya has 96 bones named
`裙_*` and Liuying 112, so they are there to be used.
