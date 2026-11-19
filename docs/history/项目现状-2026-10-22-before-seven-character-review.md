# 项目现状 · 2026-10-22 保真主线

目标是把生成动作自然地用于动漫角色。**展示原始生成结果时，先保证动作没有被美化或接触修正改写，再单独评价衣物与道具。**
此前的衣物、侧夹箱子和武器适配未获得用户认可，不能继续写成当前已完成或已验收成果。

最新审阅入口：[`converted/fidelity_review_20261022/index.html`](converted/fidelity_review_20261022/index.html)。
详细操作与历史测量见 [HANDOFF.md](HANDOFF.md)、[原生 SOMA 说明](docs/soma-direct.md) 和 [重定向说明](docs/retarget.md)。

最新用户取舍：**停止继续处理箱子**，已有诊断保留；未经新要求不再推进搬箱实验。

## 当前必须保持的条件

- 优先使用 `/Users/a26044/Motion data/t2m_demo400/soma/active/` 的原生 **SOMA-77**。`smplx/active/` 是有损拟合的 body-22 副本，没有原生手指通道。
- 默认 `--faithful --root-motion source`：关闭姿态美化，保留以米计的源根位移。脚部 IK、上半身平滑、肩部避让、箱体接触 IK 和武器手臂避让均为可选适配，不属于保真展示默认项。
- 非拇指对应已改正为 SOMA **2/3/4 → MMD 1/2/3**；拇指保持 1/2/3 → 0/1/2。原生 30 个指骨继续由源数据驱动，不用人工握姿覆盖。
- 手腕参考轴已改为 **Hand → Middle2（中指 MCP）**。旧版使用 Middle1 掌骨根，静止参考轴偏约 18.5°；旧映射视频只保留作诊断。修正后的轴由源 Hand 自身旋转携带，不逐帧追随手指来改腕。
- 当前宵宫映射审计覆盖 **50 个映射骨，其中 30 个指骨**。源方向与根位移的核对不能证明不同人物比例下的绝对关节位置、指尖位置和物体接触也相同；与同一个求解器相符，也不能代替独立解剖对应检查。

## 当前画面结论

| 项目 | 当前结论 |
| --- | --- |
| 宵宫舞蹈 `100948` | 原生 SOMA、原始 PMX 阻尼、短裙转动余量 **20°**、平移余量 **±2 mm** 的候选通过本轮完整 152 帧独立画面检查。明显抽跳已改善，仍有刚片感；不是所有服装、动作均适用的参数，也不是用户已最终验收。 |
| 宵宫走路 `004162` | 正式 20° / ±2 mm、保留长片原始阻尼的诊断片已审完 152 帧。长片仍快速张开、停下后回落慢，**自然性未通过**；99 帧检测到皮肤与选定短裙三角网格交叉。早前带脚部或肩部适配的结果不是当前保真主线的完成态。 |
| 宵宫搬箱 `016783` | 新 `carry_box_supported` 只移动、旋转固定尺寸箱体，让箱体位于既有双手上方；不改身体、腕或原生手指。完整烘焙后逐帧箱体交叉检测为 0，但 **57/152 帧**一只手与箱底的支撑间隙超过 10 mm，最大 **37.45 mm**；**109–139 帧**至少一掌背面朝向箱底。多数接触在指尖，不能承诺自然、连续的承重托举。 |
| 飞霄武器 | 旧方案约 **17°** 肩部避让、手腕最多移动 **11.7 cm**，并使用人工握姿，属于 **adapted** 适配实验。不能用于原生生成动作的保真主展示，也不能继续称为武器已完成。 |

舞蹈的 20° 参数不能直接套到箱子或走路。箱子本轮记录的衣物参数仍是 10° / ±2 mm，采用同参数、同相机的无箱与有箱对照。
衣物余量只作用于所选短裙链：所有动态端点都必须属于目标衣物，允许连接静态腰锚，不能误放松相邻长衣襬的根固定点。
这些是明确的 Blender 衣物适配，**阻尼取反不是 PMX/Blender 单位转换**；`pmx` 保留模型原始阻尼，`legacy-inverted` 仅用于复现旧样片。

箱体“零交叉”只指全部 152 个渲染帧的网格/箱体检测（箱体内缩 0.2 mm），没有证明帧间连续碰撞或自然承重。
新箱体物理前后，58 个映射及根相关骨的世界矩阵、局部姿态和根矩阵比较均为 0 差异；这是物理没有再改映射姿态的证据，不是独立源对应准确度的替代。
两人分段查看了全部 152 帧侧视图，未见整片衣物瞬态翻炸，仍不宣称全部衣物自碰已消除。

## 从哪里继续

| 内容 | 位置 |
| --- | --- |
| 最新保真审阅页 | `converted/fidelity_review_20261022/index.html` |
| 新箱完整同机位对照、报告、Blender 场景 | `outputs/fidelity-2026-10-22/box/mcp_*`；汇总为 `mcp_review_summary.json` |
| 新衣物检查 | `outputs/fidelity-2026-10-22/cloth/` |
| 原角色、武器与贴图 | `assets/fbx/pmx/`，不同版本保留，不按重名去重 |
| 人工挑选与标签 | `outputs/picks/`、`data/picks_backup/` |
| 中文摘要与测量 | `data/caption_zh.json`、`data/measurements/` |
| 脚本与目录导航 | [scripts/README.md](scripts/README.md)、[docs/repo-layout.md](docs/repo-layout.md) |

本机 Blender 5.2.0 LTS 已实际运行。后台脚本直接调用本地 Blender 和 mmd_tools，**不需要连接 Blender-MCP 或手动打开 Blender**；原 MCP 配置保留用于可选交互查看。

```bash
cd /Users/a26044/projects/MotionViewer

# 本轮舞蹈候选；结果另存，不覆盖历史视频
.local/Blender.app/Contents/MacOS/Blender --background \
  --python scripts/render_mmd_compare.py -- \
  --asset assets/fbx/pmx/yoimiya/宵宫.pmx \
  --soma "/Users/a26044/Motion data/t2m_demo400/soma/active/100948.soma.npz" \
  --output outputs/fidelity-2026-10-22/my_dance \
  --frames 152 --panels character --toon --expression none \
  --faithful --root-motion source --foot-contact off --upper-smooth 0 \
  --arm-clearance-degrees 0 --bullet --bullet-damping pmx \
  --bullet-cloth-slack 20 --bullet-cloth-linear-slack .002
```

## 历史与安全边界

旧 `outputs/` 下 97 个条目、294 个文件已整体归档到 `outputs/archive/2026-10-22/`，搬移前后逐文件核对 SHA-256，没有删除资源。
`outputs/picks/` 和 `outputs/motion_scores.json` 保持原位；模型、动作、测量、Blender 与插件没有搬动。
恢复了仍被导入入口调用的 `scripts/_extract_pmx_zip.py`。原始交接原文保存在 [docs/history/](docs/history/README.md)。

`converted/final/`、`converted/contact_review_20261022/`、`converted/cloth_props_review_20261022/` 及对应测量均保留作历史。
旧脚部 IK、去抖、侧夹箱和武器避让的数值仍可用于研究适配，但不能把“文件未改”“零穿模”或较低抖动指标写成保真、自然或用户验收。
箱目录的 `REVIEW_STATUS.json` 与 `DIAGNOSTIC_ONLY.json` 标明被后续手指、手腕映射或衣物选择修正取代的中间结果。

走路长衣摆需要检查其附着、碰撞与约束结构，不能继续把单纯降低阻尼当作完成方案。持物新样本必须标明动作 ID，不能无声替换 `016783`。
已读原 PMX 核实长片确实缝合在短裙端部，24 个接缝顶点混合两侧权重；直接改挂腰部可能拉扯接缝，未实施。结构证据保存于 `outputs/fidelity-2026-10-22/cloth/tail-structure.json`，后续应以连续衣物代理或网格布料与身体碰撞为基础。
新结果始终另建目录。未推送远程仓库，未重写原始动作或角色资产。
