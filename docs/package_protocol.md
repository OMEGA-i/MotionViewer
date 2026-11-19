# Sample Packing Protocol v2.0

Protocol for packaging generated motion samples for downstream Blender
visualization. **Single bundle per eval run** — all provenances and all task
types (t2m, pred, recon) coexist in one self-describing tar.gz.

MotionViewer is a **consumer only**. Upstream produces packages via
`package_samples.py`; this repo validates and renders through
`motionviewer.packages` (`validate_package`, `inspect_package`,
`import_package`, `build_render_job_from_package`).

## Quick Start (downstream)

```python
import json
from pathlib import Path

import numpy as np

with open("manifest.json") as f:
    manifest = json.load(f)

print(f"Track: {manifest['track']}, Split: {manifest['split']}")
print(f"Tasks: {list(manifest['tasks'])}")

for task_name, task_section in manifest["tasks"].items():
    for clip_dir in sorted(Path(f"clips/{task_name}").iterdir()):
        meta = json.loads((clip_dir / "meta.json").read_text())
        gt = np.load(clip_dir / meta["gt"]["joints22"])  # [T,22,3]
        for sid, s in meta["sources"].items():
            j22 = np.load(clip_dir / s["joints22"])
            smplx = np.load(clip_dir / s["smplx_params"])
            # render gt + j22 overlay, or drive SMPL-X mesh
```

## Bundle naming

```
{track}_{split}.tar.gz
```

| Part | Values | Example |
|------|--------|---------|
| `track` | `soma_tmr`, `smpl_hml` | `soma_tmr` |
| `split` | `test`, `val` | `test` |

Example: `soma_tmr_test.tar.gz`

## Bundle directory layout

```
{track}_{split}/
├── manifest.json                   # v2.0 run-level manifest
├── README.md
│
└── clips/
    └── {task}/                     # task namespace
        └── {clip_id}/
            ├── meta.json           # per-clip metadata
            ├── gt/                 # ground truth
            │   ├── joints22.npy
            │   ├── joints77.npy    # soma_tmr only
            │   └── smplx_params.npz
            ├── {source}/           # model or reconstruction level
            │   ├── joints22.npy
            │   ├── smplx_params.npz
            │   └── source_meta.json
            └── levels/             # recon only
                └── q{NN}.json
```

## manifest.json (v2.0)

```jsonc
{
  "protocol_version": "2.0",
  "track": "soma_tmr",
  "split": "test",
  "fps": 30.0,
  "num_clips": 35,

  "tasks": {
    "t2m": {
      "task": "t2m",
      "num_clips": 32,
      "provenances": ["HumanML3D", "mm_MotionGV"],
      "sources": [
        {"source_id": "omegamotiongpt", "kind": "model", "role": "ours", "native_rep": "motion_codes"},
        {"source_id": "kimodo", "kind": "model", "role": "baseline", "native_rep": "soma77"}
      ],
      "tokenizer": {"num_codebooks": 16, "codebook_size": 1024, "temporal_stride": 2},
      "clip_stats": {"min_T": 60, "max_T": 180, "median_T": 120}
    },
    "recon": {
      "task": "recon",
      "num_clips": 3,
      "provenances": ["HumanML3D"],
      "sources": [
        {"source_id": "q04", "kind": "reconstruction_level", "num_codebooks_used": 4},
        {"source_id": "q10", "kind": "reconstruction_level", "num_codebooks_used": 10}
      ],
      "tokenizer": {"num_codebooks": 16, "codebook_size": 1024, "temporal_stride": 2},
      "reconstruction_levels": [4, 10, 16],
      "source_clip_id": "test:rec_01KX..."
    }
  },

  "clip_stats": {"min_T": 60, "max_T": 380, "median_T": 120},
  "created_at": "2026-07-26T12:00:00Z",
  "generator": "omega_motion_gpt.eval.package_samples/2.0",
  "notes": []
}
```

Key changes from v1.0:

- **`tasks`** — dict of per-task objects instead of single `task` + `provenance`
- **`provenance`** removed from top level — now per-clip in `meta.json`
- **`num_clips`** — total across all tasks
- Clips live under `clips/{task}/` instead of `clips/`

MotionViewer does **not** read protocol v1.0 packages.

## meta.json (per-clip)

```jsonc
{
  "clip_id": "test:abc123",
  "rec_id": "abc123",
  "provenance": "mm_MotionGV",
  "task": "t2m",
  "T": 120,
  "fps": 30.0,
  "prefix_T": 0,
  "caption": "A person walks forward.",

  "gt": {
    "joints22": "gt/joints22.npy",
    "joints77": "gt/joints77.npy",
    "smplx_params": "gt/smplx_params.npz"
  },

  "sources": {
    "omegamotiongpt": {
      "kind": "model",
      "role": "ours",
      "native_rep": "motion_codes",
      "joints22": "omegamotiongpt/joints22.npy",
      "smplx_params": "omegamotiongpt/smplx_params.npz",
      "T": 96,
      "fit_mse": 0.023
    },
    "q04": {
      "kind": "reconstruction_level",
      "num_codebooks_used": 4,
      "total_codebooks": 16,
      "joints22": "q04/joints22.npy",
      "smplx_params": "q04/smplx_params.npz",
      "T": 96,
      "mse_vs_gt": 0.0085,
      "levels_meta": "levels/q04.json"
    }
  }
}
```

## clip_id vs directory name

`clip_id` uses the canonical format `{split}:{rec_id}` (e.g. `test:rec_01KX...`).
Directory names under `clips/{task}/` use a filesystem-safe encoding where `:` → `_`
(e.g. `test_rec_01KX...`).

Downstream tools should canonicalise via `clip_id` from `meta.json`, not by
parsing directory names. MotionViewer accepts either form for `--clip`.

## .smplx_params.npz format

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `joints22` | [T,22,3] | float32 | 22-joint positions (redundant with .npy) |
| `transl` | [T,3] | float32 | Global translation |
| `global_orient` | [T,3] | float32 | Root orientation (axis-angle) |
| `body_pose` | [T,63] | float32 | 21 body joints × 3 axis-angle |
| `betas` | [10] | float32 | Body shape coefficients |
| `prefix_T` | scalar | int32 | Conditioning frame count |
| `fps` | scalar | float32 | Frames per second |
| `source` | string | - | Source identifier |
| `format` | string | - | `smplx_body22_fitted_aa` or `smplx_body22_native_aa` |
| `track` | string | - | Track name |
| `fit_mse` | scalar | float32 | IK fitting residual |

## Task type details

### text_to_motion (t2m)

- `meta.json.caption` = text prompt
- `meta.json.prefix_T` = 0

### motion_prediction (pred)

- `meta.json.prefix_T` = conditioning frame count
- `meta.json.predicted_T` = number of predicted frames
- GT = full sequence; source = predicted continuation only

### reconstruction (recon)

- Sources named `q{NN}` (zero-padded codebook count)
- `manifest.tasks.recon.reconstruction_levels` = ordered codebook counts

## Source unified model

Both model outputs and reconstruction levels are **sources** — alternative
motion sequences for the same clip. Every source has the same file structure:
`joints22.npy` + `smplx_params.npz` (+ optional `joints77.npy` / `native.npy` /
`source_meta.json`).

## Validation

MotionViewer: `motionviewer.packages.validate_package(path, mode=...)`.

Checks (structural / assets):

1. `manifest.json` exists and is valid v2.0 JSON
2. `clips/{task}/` directories exist for each task in manifest
3. Per-clip: meta parses, gt/source assets present (assets mode), shapes/keys when loadable
4. Clip sources must be a **subset** of that task's manifest sources
   (extra in clip → error; missing from clip → OK)
5. Each manifest source appears in **at least one** clip (task-level)
6. T2M clips: `caption` required
7. Pred clips: `prefix_T > 0`
8. Recon tasks: `q{NN}` names match `reconstruction_levels`

Key design: manifest declares the **union** of all sources across clips.
Individual clips may omit some models that lack cached outputs.
