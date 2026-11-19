# 脚本入口与用途

日常工作优先使用下面的入口。脚本暂时保持原路径：批量任务、测试和脚本之间
已经存在直接调用，按文件名前缀搬家会破坏它们。

## 动作应用到动漫角色

| 工作 | 入口 | 运行环境 |
| --- | --- | --- |
| 预览动作、手工挑选 | [`pick_clips.py`](pick_clips.py) | `uv run --group viz python`；使用 Viser |
| 批量渲染、生成视频浏览页 | [`visualize_dataset.py`](visualize_dataset.py) | `uv run python`；自动启动 Blender |
| 单条动作、角色与源骨架对照 | [`render_mmd_compare.py`](render_mmd_compare.py) | Blender `--background --python` |
| 论文静帧 | [`render_paper_figure.py`](render_paper_figure.py) | Blender `--background --python` |
| 拼接论文图 | [`compose_paper_figure.py`](compose_paper_figure.py) | `uv run python` |
| 新角色检查 | [`onboard_character.py`](onboard_character.py) | `uv run python`；自动启动 Blender |
| 双视角握持检查 | [`_render_grip.py`](_render_grip.py) | Blender `--background --python` |

后台渲染直接调用 Blender，不要求连接 Blender-MCP。不要同时启动超过三个 Blender 进程。

## 验证与诊断

| 检查对象 | 脚本 |
| --- | --- |
| SMPL-X → MMD 求解一致性 | [`_validate_mmd_retarget.py`](_validate_mmd_retarget.py) |
| SOMA 求解与对比 | [`_validate_soma_retarget.py`](_validate_soma_retarget.py)、[`_probe_soma_vs_smplx.py`](_probe_soma_vs_smplx.py) |
| 原生动作逐帧保真审计 | [`_audit_soma_display.py`](_audit_soma_display.py)；检查映射方向、根轨迹和当前手部对应，支持已烘焙场景；不替代接触/解剖/画面验收 |
| 原生角色全片复查 | [`_review_native_character.py`](_review_native_character.py)；透传单片参数，保存场景、真实鞋底测量和逐帧源动作审计 |
| 角色骨架、形态键、材质效果 | [`_dump_mmd_rig.py`](_dump_mmd_rig.py)、[`_dump_mmd_morphs.py`](_dump_mmd_morphs.py)、[`_render_mmd_sheet.py`](_render_mmd_sheet.py)、[`_render_face.py`](_render_face.py) |
| 角色可用性、手部穿模 | [`_probe_characters.py`](_probe_characters.py)、[`_check_hand_clipping.py`](_check_hand_clipping.py) |
| 裙摆与物理覆盖 | [`_probe_skirt_flare.py`](_probe_skirt_flare.py)、[`_probe_skirt_physics.py`](_probe_skirt_physics.py) |
| 当前衣物全片审计（原网格、时序、皮肤交叉） | [`_audit_mmd_cloth.py`](_audit_mmd_cloth.py)；透传单片渲染参数，`--audit-only` 跳过图片 |
| 当前脚底接触审计 | [`_audit_mmd_feet.py`](_audit_mmd_feet.py)；实际鞋底顶点与逐帧接触记录 |
| 箱体和握持初步测量 | [`_probe_prop_fit.py`](_probe_prop_fit.py)；不能替代皮肤三角面检查和实际视频 |
| 动作质量 | [`score_motion_quality.py`](score_motion_quality.py)、[`_diagnose_motion.py`](_diagnose_motion.py) |
| 合图与视频抽检 | [`_compose_sheet.py`](_compose_sheet.py)、[`qc_contact_sheet.py`](qc_contact_sheet.py) |

`_probe_cloth_motion.py` 的时序相关性不能判断裙片是否变形。新结果须同时看图或视频。
`_check_hand_clipping.py` 是旧的采样启发式，开放网格上的内外判断不可靠，不能单独作为
“没有穿模”的验收依据。新道具报告会注明实际采样/三角检测的范围和剩余问题。

## 导入、其他输出与早期流程

| 用途 | 脚本 |
| --- | --- |
| 单个 ZIP 导入（`onboard_character.py` 仍在调用） | [`_extract_pmx_zip.py`](_extract_pmx_zip.py) |
| 批量导入与贴图文件名修复 | [`_extract_characters.py`](_extract_characters.py)、[`_repair_texture_names.py`](_repair_texture_names.py) |
| FBX 姿态网格、质量审计、导出 | [`render_fbx_pose_grid.py`](render_fbx_pose_grid.py)、[`retarget_quality_audit.py`](retarget_quality_audit.py)、[`export_retarget_fbx.py`](export_retarget_fbx.py) |
| 通用定性视频 | [`render_qualitative_clip.py`](render_qualitative_clip.py) |
| 早期宵宫单角色入口 | [`render_yoimiya.py`](render_yoimiya.py) |
| 早期展示页与批处理 | [`generate_showcase.py`](generate_showcase.py)、[`build_review_page.py`](build_review_page.py)、[`_batch_review.py`](_batch_review.py)、[`_batch_paper_figures.py`](_batch_paper_figures.py) |
| 历史布料时序实验 | [`_probe_cloth_motion.py`](_probe_cloth_motion.py) |
| 已归档脚本 | [`archive/README.md`](archive/README.md) |

早期入口保留用于复查旧结果；它们不一定使用当前批量入口的参数和物理设置。

`render_mmd_compare.py --physics-profile <JSON> --bullet` 使用与角色文件哈希绑定的
服装配置。原生 SOMA
复查用 `--soma`；`visualize_dataset.py` 当前的数据集校验仍针对 SMPL-X body-22，
不能把拟合的无手指副本当作原始 SOMA 手部展示。

## Unitree G1 闭环仿真

| 工作 | 入口 |
| --- | --- |
| SOMA-77 朝向校准并重定向到 G1 | [`g1/retarget_candidates.py`](g1/retarget_candidates.py) |
| GEAR-SONIC 200 Hz MuJoCo 跟踪 | [`g1/simulate_sonic_g1.py`](g1/simulate_sonic_g1.py) |
| 视觉网格与世界系 FK | [`g1/export_g1_fk.py`](g1/export_g1_fk.py) |
| 本机仿真并直接导出 FK | [`g1/run_local_g1_sim_and_fk.py`](g1/run_local_g1_sim_and_fk.py) |

机器人 XML、SONIC 权重和 SOMA 片段都在仓库外。产物写到 `outputs/g1_sim/`。
路径可用 `MOTIONVIEWER_G1_DIR`、`MOTIONVIEWER_SONIC_POLICY`、`MOTIONVIEWER_SOMA_DIR`、
`MOTIONVIEWER_SOMA_ORIENT`、`MOTIONVIEWER_PICKS39_CSV` 和 `MOTIONVIEWER_G1_WORK` 覆盖。
依赖 `mujoco`、`onnxruntime`、`scipy`，重定向另需本机的 `soma-retargeter`。
论文静帧光追脚本留在本地 `outputs/g1_generated_gallery/scripts/`。

## 整理时必须保留的调用关系

- `pick_clips.py` 导入 `visualize_dataset.py`；后者导入 `score_motion_quality.py`。
- `_repair_texture_names.py` 导入 `_extract_characters.py`。
- `onboard_character.py` 调用 `_extract_pmx_zip.py`、`_dump_mmd_rig.py`、
  `_validate_mmd_retarget.py` 和 `_render_mmd_sheet.py`。
- 多个批处理入口按 `scripts/<文件名>.py` 调用渲染器；测试也直接使用这些路径。

部分诊断脚本仍有本机数据路径或固定输出路径。运行前查看文件头和参数；
不要用历史实验默认值覆盖新结果。原始模型、动作和成品留在仓库外。
