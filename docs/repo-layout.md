# 项目目录地图与安全整理规则

2026-10-22 核对并整理。当前主线是将 SMPL-X / SOMA 动作应用到动漫角色，
产出自然、好看的视频和论文图。先看 [项目现状](../项目现状.md)，
具体入口见 [脚本导航](../scripts/README.md)。

## 代码与运行入口

| 路径 | 作用 |
| --- | --- |
| `src/motionviewer/core/` | 不依赖 Blender 的运动学、坐标和动作处理 |
| `src/motionviewer/loaders/`、`packages/` | 动作格式、样本包验证与读取 |
| `src/motionviewer/video/` | 任务配置、运行记录、编码与批处理支持 |
| `src/motionviewer/blender/retarget/` | 将源骨架动作求解到角色骨架 |
| `src/motionviewer/blender/mmd_*.py` | MMD 导入、物理、材质、表情和道具 |
| `scripts/pick_clips.py` | 动作预览和人工挑选 |
| `scripts/visualize_dataset.py` | 批量生成 MP4、浏览页和运行记录 |
| `scripts/render_mmd_compare.py` | 单条动作的角色与源骨架渲染 |
| `scripts/render_paper_figure.py` | 论文静帧 |
| `scripts/g1/` | Unitree G1 重定向、SONIC 闭环仿真和视觉 FK；产物在 `outputs/g1_sim/` |
| `scripts/archive/` | 明确归档的旧脚本及原因；不作为当前默认入口 |
| `tests/`、`configs/examples/` | 自动检查与通用任务示例 |

脚本保持既有路径。`pick_clips.py → visualize_dataset.py → score_motion_quality.py`
等直接导入仍然存在；`onboard_character.py` 也会按固定路径调用其他脚本。
此前误归档的 `_extract_pmx_zip.py` 已恢复到 `scripts/`，使单个 ZIP 的角色导入
重新找到它的依赖。移动任何其他脚本前，须先检查调用、导入和测试引用。

## 原始输入、人工记录与成品

| 路径 | 内容与处理原则 |
| --- | --- |
| `/Users/a26044/Motion data/t2m_demo400/` | 项目外的原始 SMPL-X / SOMA 动作、GT、补长版本和元数据；未移动 |
| `assets/fbx/pmx/` | 20 个角色包目录，包含角色、武器、贴图和不同版本；未按名称去重，未移动 |
| `data/examples/` | 可再分发的小型动作示例 |
| `data/caption_zh.json` | 400 条中文动作摘要；保留 |
| `data/picks_backup/` | 人工筛选和标签备份，包括 45 条 keep、8 条 tmp；保留 |
| `data/measurements/` | 17 份历史测量 JSON；保留，可信范围见其 README |
| `outputs/picks/` | 挑选器仍使用的工作记录；保留原路径 |
| `outputs/motion_scores.json` | 旧展示流程引用的评分文件；保留原路径 |
| `converted/final/` | 6 个已有视频与 5 张检查图；保留原路径和内容 |
| `converted/soma_probe/` | SOMA 路线历史比较视频；保留 |
| `converted/figures/figures.json` | 仅有历史生成记录，不代表整套图仍存在 |
| `.local/` | 本地 Blender、MMD 插件和历史输入；运行依赖，未移动 |
| `.venv/` | 项目 Python 环境；未移动 |

`data/measurements/` 的 `skirt_physics.json` 中 `lower_share` 已知选错了顶点范围，
不能因文件被保留就把这个指标当成正确结论。其他历史报告也应结合源代码与画面复核。

## 本次归档了什么

`outputs/` 原有的 **97 个顶层条目、294 个文件，共 94,103,960 字节**，
整体移动到 [`outputs/archive/2026-10-22/`](../outputs/archive/2026-10-22/README.md)。
内容包括旧渲染帧、视频、比较图、日志和测量副本；没有按“缓存”删除任何文件。
`outputs/picks/` 和 `outputs/motion_scores.json` 不在本次移动范围内。

归档中的 [`archive-manifest.json`](../outputs/archive/2026-10-22/archive-manifest.json)
记录了旧路径、新路径、字节数和每个文件的 SHA256。迁移以此核对完整性。
完整移动历史浏览页所在目录，保留其目录内的相对视频、图片链接。

旧 `manifest.json` / `job_bundle.json` 中的原始绝对路径保持原样，表示当时的
输入和输出位置；它们不是今天可直接执行的配置。需要找迁移后的文件时查归档清单，
不要为使旧报告“看起来当前”而改写历史来源记录。

## 后续整理规则

1. **新任务使用独立输出目录。** 中间文件放 `outputs/<本次任务>/`，待审阅视频
   放 `converted/<本次任务>/`；不要覆盖 `converted/final/` 或历史归档。
2. **先查依赖，再移动。** 代码引用、HTML 相对路径、纹理路径、符号链接和
   人工记录都可能依赖当前位置；不能仅凭文件名、年龄或“可重跑”就删除。
3. **模型与原始动作不随清理搬家。** `assets/`、`data/`、`converted/`、`.local/`
   本次均保持原位。相似角色名、不同衣装和武器文件不自动视为重复。
4. **需要归档时保留内容并记录迁移。** 保存旧→新路径和校验值；恢复到旧位置前
   检查是否已有新任务同名输出，避免覆盖。
5. **保留忽略规则。** 角色、模型权重、动作数据、Blender 插件和渲染产物的授权
   各自独立，整理目录不构成发布许可。不要把 `.local/`、`assets/`、`outputs/`
   或 `converted/` 的本地内容加入 Git。

本机 `.local/Blender.app/Contents/MacOS/Blender --version` 已返回 Blender 5.2。
渲染直接运行后台 Blender，不依赖 Blender-MCP；MCP 配置仅用于可选的交互调试。
