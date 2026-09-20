# Archived scripts

Kept rather than deleted: each of these worked at some point and the measurements in
its docstring may still be the only record of how something was established. Nothing
here is on the current render path, and nothing imports from this directory.

| file | superseded by | why it was retired |
| --- | --- | --- |
| `render_motion.py` | `uv run motionviewer` | a two-line wrapper around `motionviewer.cli:app`, which `[project.scripts]` now installs directly |
| `check_addon.py` | `uv run motionviewer doctor` | same — a wrapper around the packaged addon check |
| `_extract_pmx_zip.py` | `_extract_characters.py` | single-archive version. Its encoding notes are the fuller ones: Shift-JIS from Japanese tools, GBK from Chinese ones, and archives that mix flagged UTF-8 with unflagged Shift-JIS |
| `_dump_mmd_materials.py` | — | one-off inspection of the material graph `mmd_tools` builds. Nothing references it, but it is how the cel-shading inputs were first read |
| `_render_hand_closeup.py` | `_render_grip.py` | aimed at the wrist bone, which is not where a held weapon sits; the replacement aims at the fist centre and shoots two orthogonal views |

To bring one back, move it up a directory — none of them depend on being here.
