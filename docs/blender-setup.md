# Blender setup

Blender 5.2 LTS is the reference environment. MotionViewer supports macOS and Linux host runtimes; exact Blender installation paths are not part of render configs.

## Install the SMPL-X addon

MotionViewer's primary mesh backend uses [`smplx_blender_addon`](https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon). The upstream repository contains addon code only. Obtain the prebuilt addon and required body-model data under the upstream terms; never copy those files into this repository.

The integration probes addon metadata and requires the `scene.smplx_add_gender` operator. It does not depend on a fixed addon directory name.

## Select Blender

Resolution order:

1. `--blender /path/to/blender`
2. `MOTIONVIEWER_BLENDER=/path/to/blender`
3. `blender` on `PATH`
4. common macOS and Linux locations

Validate the installation:

```bash
uv run motionviewer doctor
```

The command reports the Blender version, addon module, required operators, tool properties, and whether model data files are present.

## Blender-MCP

The project `.mcp.json` starts Blender-MCP through `uvx`. Enable its Blender addon and run Blender interactively before using MCP tools. Blender-MCP deliberately refuses to start its command server in background mode because background Blender has no interactive command loop.

MotionViewer background renders do not depend on MCP. They execute versioned Python adapters with `blender --background --python ...`.

## Troubleshooting

- `Blender executable was not found`: pass `--blender` or set `MOTIONVIEWER_BLENDER`.
- `SMPL-X addon was not found`: install and enable the addon in the selected Blender profile.
- `scene.smplx_add_gender is unavailable`: verify the addon version and its required data installation.
- MP4 encoding fails: install `ffmpeg` and ensure it is on `PATH`.
- A local FBX test skips: install a separately licensed binary FBX and its catalog entry; public CI has no character assets.
