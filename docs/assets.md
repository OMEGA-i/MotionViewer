# Local asset catalogs

MotionViewer does not distribute character models. A local FBX catalog records which separately obtained assets are eligible for deterministic retargeting and random selection.

```text
assets/fbx/
  catalog.json
  character.fbx
```

The catalog contains reusable rig profiles and per-asset entries:

```json
{
  "profiles": {
    "mixamo": {
      "profile_id": "mixamo",
      "rig_family": "mixamo",
      "bone_map": "auto",
      "retarget_mode": "quality",
      "validation_status": "active"
    }
  },
  "assets": [
    {
      "model_id": "character",
      "path": "character.fbx",
      "profile_id": "mixamo",
      "status": "pending",
      "random_eligible": false,
      "reason": "local validation required",
      "evidence": {}
    }
  ]
}
```

Only Blender-compatible binary FBX files are accepted. `approved` plus `random_eligible: true` is required for paper-safe random selection.

```bash
uv run motionviewer fbx check --root assets/fbx
uv run motionviewer fbx check --root assets/fbx --deep --write-report outputs/fbx-audit
```

The deep audit requires Blender, local motion samples, and the repository's audit adapter. Asset licenses are independent of a passing technical audit.
