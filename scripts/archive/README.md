# Archived scripts

Kept rather than deleted: each of these worked at some point and the measurements in
its docstring may still be the only record of how something was established. These
are historical references, not the normal entry points; see [the script index](../README.md).

| file | superseded by | why it was retired |
| --- | --- | --- |
| `render_motion.py` | `uv run motionviewer` | a two-line wrapper around `motionviewer.cli:app`, which `[project.scripts]` now installs directly |
| `check_addon.py` | `uv run motionviewer doctor` | same — a wrapper around the packaged addon check |
| `_dump_mmd_materials.py` | — | one-off inspection of the material graph `mmd_tools` builds. Nothing references it, but it is how the cel-shading inputs were first read |
| `_render_hand_closeup.py` | `_render_grip.py` | aimed at the wrist bone, which is not where a held weapon sits; the replacement aims at the fist centre and shoots two orthogonal views |

`_extract_pmx_zip.py` was restored to [`scripts/_extract_pmx_zip.py`](../_extract_pmx_zip.py)
on 2026-10-22: `onboard_character.py --archive` still calls that exact path.
`_extract_characters.py` is a separate batch importer, not a compatible replacement
for that single-archive command.

Before reviving any other archived script, check its repository-root calculation,
imports and callers. Several were written to run one directory higher.
