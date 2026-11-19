# 保留的渲染结果

这里保存值得保留的成品与比较证据。最新七角色复查为 2026-10-23，其余为历史保留证据；
文件存在不代表旧 agent 对它的所有质量判断已经重新验证。

## 当前内容

| 路径 | 实际内容 | 用途 |
| --- | --- | --- |
| [`character_review_20261023/`](character_review_20261023/index.html) | 七角色、每位两段完整动作的同机位前后对比与源核对记录 | 当前入口；每个角色的改善与残留单独说明 |
| [`fidelity_review_20261022/`](fidelity_review_20261022/index.html) | 原生 SOMA、修正手指与掌轴后的衣物对照、箱体上方放置与逐帧报告 | 历史保真检查；当时走路长衣摆与箱体自然承托未通过 |
| [`cloth_props_review_20261022/`](cloth_props_review_20261022/index.html) | 衣摆、手臂避让、箱子和武器前后视频及真实烘焙检查 | 历史适配实验，含明显人体姿态修正，不作为当前保真主展示 |
| [`contact_review_20261022/`](contact_review_20261022/index.html) | 脚部与上半身去抖前后视频 | 可选接触/平滑适配；改变显示动作，原输入文件保留 |
| [`review_20261022/`](review_20261022/index.html) | 新批量视频、阻尼对照、验证记录 | 早期重新运行的审阅入口 |
| [`final/`](final/README.md) | 6 个 MP4、5 张检查图、README | 宵宫、芙宁娜、妮露的舞蹈；彦卿、流萤、青衣的持武器视频 |
| `final/检查图/` | 六帧检查、布料检查、角色站立合图、握持特写 | 对照旧结果，不能替代新修改后的验证 |
| `soma_probe/` | 5 个 MP4、5 个文本记录 | SOMA 与 SMPL-X 路线的历史比较证据 |
| `figures/figures.json` | 历史生成记录 | 当前没有对应的整套论文图文件，不能当成现成图集 |

旧文档中的 `showcase/`、`showcase_cast/` 和 `figures/yoimiya/` 当前不在此目录中。
历史实验视频和截图在 [`outputs/archive/2026-10-22/`](../outputs/archive/2026-10-22/README.md)，
完整目录说明见 [项目地图](../docs/repo-layout.md)。

## 新结果如何保存

先把新一轮结果写到独立目录，例如 `converted/review_20261022/`。
批量入口生成 MP4、`index.html` 和运行记录；单条渲染入口的帧、诊断和中间文件
放到对应的 `outputs/<本次任务>/`。检查动作、脚部接触、布料、手部和贴图后再决定
是否作为成品保留，避免覆盖 `final/` 中用于对照的旧视频。

```bash
uv run python scripts/visualize_dataset.py \
  --data "/Users/a26044/Motion data/t2m_demo400/smplx" \
  --task active --clips 100948 --character yoimiya --physics bullet \
  --output converted/review_20261022
```

以上批量命令使用 SMPL-X 拟合副本；原生生成展示优先采用项目 README 中的 SOMA 单条入口。
命令从项目根目录执行，使用本地已存在的数据与角色。
后台渲染直接启动 Blender，不依赖 Blender-MCP。
更多入口见 [脚本导航](../scripts/README.md) 和 [数据集渲染说明](../docs/dataset-visualisation.md)。

## 本地文件与授权

`converted/` 除本 README 外全部被 Git 忽略。这条规则应当保留：转换后的
`.blend` 会包含角色网格、权重，可能还打包贴图，分享它也就在分享源模型。
本地部分 PMX 明确禁止再分发；转换格式不会改变原始角色的授权要求。
视频、图片与模型文件应分别遵循各自适用的许可，不把生成成功视为获准发布。

动作数据也有独立的授权范围。`data/examples/` 仅保存允许再分发的示例；
内部动作集、模型权重和本地角色不要因目录整理而加入版本控制。

## 2026-10-22 contact / jitter review

`contact_review_20261022/index.html` contains matched-camera walking/jumping foot A/B videos, a separate subtle upper-body smoothing comparison, review records, and experimental prop closeups. Source motions and all earlier outputs remain unchanged.
