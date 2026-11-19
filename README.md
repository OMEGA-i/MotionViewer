# MotionViewer

MotionViewer is a Python toolkit for turning SMPL-X motion sequences into reproducible Blender renders. It validates motion inputs, prepares portable render-job bundles, drives Blender in the background, and records the resolved configuration and input hashes used for every output.

The project is aimed at research figures and model comparisons where scale, trajectory, camera, and visual style must remain consistent.

## 当前主线：把动作自然地应用到动漫角色

当前优先使用**原生 SOMA 动作**驱动本地 PMX 角色，产出论文静帧和网页视频。
`t2m_demo400/smplx` 是拟合副本，缺少原生手指动作；展示生成结果应优先使用
`t2m_demo400/soma`。默认关闭动作美化、接触 IK 和平滑，保留原数据的行进距离。
衣物和道具可以单独适配；不同人物比例仍会改变关节位置与接触关系。
先看 [项目现状](项目现状.md) 了解已确认的问题和下一步，再看
[脚本导航](scripts/README.md) 找入口。[HANDOFF](HANDOFF.md) 记录当前接手条件，
旧交接原件保存在 [docs/history/](docs/history/README.md)。

- 最新七角色审阅：[`converted/character_review_20261023/index.html`](converted/character_review_20261023/index.html)。
- 历史样片：[`converted/final/`](converted/final/README.md)，原有 6 个视频及检查图保留，非本轮验收结论。
- 目录地图与本次归档：[`docs/repo-layout.md`](docs/repo-layout.md)。
- 批量参数与验收说明：[`docs/dataset-visualisation.md`](docs/dataset-visualisation.md)。
- [`BATCH.md`](BATCH.md) 是旧数据集的操作记录，不是当前状态清单。

本机 `.local/Blender.app/Contents/MacOS/Blender --version` 已返回 Blender 5.2。
后台渲染直接启动这个程序，**不需要连接 Blender-MCP，也不需要手动打开 Blender**。
下面是使用本地已有数据的命令；新结果写入独立目录，不覆盖 `converted/final/`。

```bash
cd /Users/a26044/projects/MotionViewer

# 宵宫网格布料配置（更多角色见 configs/characters/）
.local/Blender.app/Contents/MacOS/Blender --background \
  --python scripts/render_mmd_compare.py -- \
  --asset assets/fbx/pmx/yoimiya/宵宫.pmx \
  --soma "/Users/a26044/Motion data/t2m_demo400/soma/active/004162.soma.npz" \
  --output outputs/my-review/yoimiya_walk \
  --frames 152 --panels character --toon --expression none \
  --faithful --root-motion source --foot-contact off --upper-smooth 0 \
  --arm-clearance-degrees 0 --bullet \
  --physics-profile configs/characters/yoimiya_mesh_cloth.json

# 旧 SMPL-X 拟合数据的批量入口（并非原生 SOMA 展示）
uv run python scripts/visualize_dataset.py \
  --data "/Users/a26044/Motion data/t2m_demo400/smplx" \
  --task active --clips 100948 --character yoimiya --physics bullet \
  --output converted/review_20261022
```

原始动作位于项目外；角色、Blender 和插件在本地保留。SMPL-X 网格后端所需的
模型权重与 PMX 角色渲染是不同依赖，不能用一个后端的检查结果代替另一个。
本轮七个角色各检查两段完整动作，服装配置按具体模型保存，详见
[角色物理配置](docs/character-physics-profiles.md)。宵宫采用网格布料解算；其他角色
按模型服装结构校准或保留原设置。局部穿插、妮露走路折落及比例造成的滑步仍有残留。
原始身体与手指动作不改，箱子工作已停止。

## Highlights

- SMPL-X mesh rendering through the external SMPL-X Blender addon
- deterministic loaders for SMPL-X NPZ and generic joint arrays
- Sample Packing Protocol package inspection, validation, selection, and preview
- single-actor, multi-view video jobs and transparent qualitative snapshot renders
- optional FBX retargeting, catalog validation, pose grids, and quality audits
- fast CPython test suite; Blender integration tests are opt-in

## Requirements

- macOS or Linux
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Blender 5.2 in the locally verified integration environment
- For the SMPL-X mesh backend: [`smplx_blender_addon`](https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon) plus separately obtained SMPL-X model data
- `ffmpeg` when encoding MP4 output

The addon repository contains code only and uses CC BY-NC 4.0. Its model data has separate terms and is never part of MotionViewer.

## Install

```bash
git clone https://github.com/OMEGA-i/MotionViewer.git
cd MotionViewer
uv sync --dev
uv run motionviewer --help
```

MotionViewer discovers Blender from `--blender`, the `MOTIONVIEWER_BLENDER` environment variable, `PATH`, or common macOS/Linux locations.

```bash
uv run motionviewer doctor
```

## Quick start

Inspect the included redistributable motion sample:

```bash
uv run motionviewer motion inspect \
  data/examples/smplx_body22_fitted_aa/omegamotiongpt.smplx.npz
```

Create and validate a render configuration without launching Blender:

```bash
uv run motionviewer motion config \
  data/examples/smplx_body22_fitted_aa/omegamotiongpt.smplx.npz \
  --output configs/generated/demo.yaml

uv run motionviewer render job \
  --config configs/examples/multiview_single_actor.yaml \
  --dry-run
```

Render with the discovered Blender installation:

```bash
uv run motionviewer render job \
  --config configs/examples/multiview_single_actor.yaml
```

Package and FBX workflows are grouped by domain:

```bash
uv run motionviewer package inspect path/to/package.tar.gz
uv run motionviewer package import path/to/package.tar.gz
uv run motionviewer package preview path/to/package.tar.gz --source model_name
uv run motionviewer fbx check --root assets/fbx
```

## Local data and licensed assets

The repository contains small motion examples only. These paths remain local and are ignored by Git:

- `assets/` — downloaded FBX/PMX characters and local catalogs
- `data/raw/`, `data/local/`, `data/packages/` — research inputs, archives, and caches
- `outputs/` — rendered frames, videos, manifests, and audits
- Blender addon data and SMPL-X weights

Do not remove these ignore rules without checking redistribution terms. See [asset setup](docs/assets.md) and [data layout](data/README.md).

## Development tools

The committed `.mcp.json` starts `blender-mcp` for interactive Blender sessions. Blender-MCP cannot serve commands from a background `blender -b` process; MotionViewer's CLI uses packaged Blender entry scripts for headless work.

Project instructions are hierarchical: start with [AGENTS.md](AGENTS.md), then follow the nearest module-level `AGENTS.md`.

## Documentation

- [Architecture](docs/architecture.md)
- [Blender and addon setup](docs/blender-setup.md)
- [Render jobs](docs/render-jobs.md)
- [Motion formats](docs/formats.md)
- [Sample Packing Protocol](docs/package_protocol.md)
- [Package workflows](docs/archives.md)
- [FBX retargeting](docs/retarget.md)
- [Adding a downloaded character](docs/characters.md)
- [Repository layout](docs/repo-layout.md)
- [Visualising a T2M dataset](docs/dataset-visualisation.md)
- [Reading SOMA-77 directly](docs/soma-direct.md)
- [Visual style](docs/visual-style.md)

## License

MotionViewer source code is released under the [MIT License](LICENSE). Third-party addons, body-model data, motion datasets, and character assets retain their own licenses.
