# Unitree G1 仿真运动与论文首页 Teaser 渲染交接文档（2026-11-17）

本文档记录从 **SeMoCo / OmegaMotionGPT 生成的原生 SOMA-77 动作** 出发，经 **NVIDIA `soma-retargeter` 骨架重定向**、**NVIDIA `GEAR-SONIC` 200 Hz MuJoCo 闭环物理跟踪仿真**，到 **Blender 4.5 LTS Cycles OptiX 居家场景分层光追与线性空间频闪合成（Stroboscopic Teaser）** 的完整数据来源、本地产物目录、架构细节与一键复现/改图方法。

---

## 1. 数据溯源与核查结论（重要！）

先后产出了两个本地结果目录，两者的数据来源严格区分如下：

### A. 最新选片主库（100% 模型生成数据，推荐用于论文正图）
- **本地目录**：`outputs/g1_generated_gallery/`
- **网页审阅入口**：直接用浏览器打开 `outputs/g1_generated_gallery/index.html`
- **数据来源**：**全部 18 张图 100% 来自我们自己的生成模型输出**，零真值（GT）混入：
  - 17 条取自 `t2m_demo400/soma/active/<id>.soma.npz`（逐文件校验 `npz["source"] == "omegamotiongpt"`）。
  - 1 条（`p39_000170`）取自用户历史精选生成集 `runs/picks39_smplx` / `_sonic_v1/picks39_g1csv`（校验 `manifest.json` 中 `"which": "gen"`）。
- **包含内容**：
  - `captioned/01_*.jpg` ~ `18_*.jpg`：每张 `1600×1002`，顶部带 `#编号 · ClipID` 角标，底部配完整英文生成 Text Prompt。
  - `clean/01_*.png` ~ `18_*.png`：每张 `1600×920` 无任何文字角标的纯净高分原图（可直接拖入 Illustrator / Figma / LaTeX 任意裁切拼版）。
  - `teasers/overview_sheet_1.jpg`、`overview_sheet_2.jpg`：两张 `3×3` 九宫格带字幕速览表（`#01~#09` 与 `#10~#18`）。
  - `teasers/teaser_gen_option1.png`、`teaser_gen_option2.png`：两版 `3840×1560` 全生成动作 4 分镜 OmniRetarget 风格首页 Teaser 预拼样稿。
  - `manifest.json`：记录全部 18 张图的 `shot_id`、`cid`、定格帧号、透明度梯度、相机视锥 NDC 包围盒、完整英文 Prompt 与中文说明。
  - `scripts/`：全套 17 个端到端脚本与物理评测汇总 `summary.json`。

### B. 首轮 4K 样片归档（含 2 条探针真值，仅作版式对照参考）
- **本地目录**：`outputs/g1_teaser_final/`
- **数据来源说明**：首轮 4 分镜中，`(b) 016783` 与 `(d) 039936` 来自我们生成的 `t2m_demo400/soma/active`；但 `(a) probe_PA6JYF` 与 `(c) probe_SQTX5X` 取自 `soma_retarget_probe`（核查其 `README.md` 确认为 `bones-seed` test 集真值探针，未经过生成器）。因此**正式论文展示请从上方 `outputs/g1_generated_gallery/` 的 18 张纯生成图中挑选**。

---

## 2. 18 张纯模型生成选片清单与对应 Text Prompt

| 编号 | 文件名（`captioned/` & `clean/`） | Clip ID | 居家场景与动作概括 | 原始生成英文 Text Prompt |
| :--- | :--- | :--- | :--- | :--- |
| **#01** | `01_060150_jog_living` | `060150` | 现代客厅 · 匀速慢跑（5帧） | *"A young man is jogging in a park. He moves his arms rhythmically, lifting them alternately while his legs propel him forward. His stride is consistent and energetic, suggesting he is focused on maintaining a steady pace. The outdoor setting is bright and full of trees, adding a lively atmosphere to his run."* |
| **#02** | `02_p39_000170_curved_walk` | `p39_000170` | 北欧客厅 · 弧形转向行走（5帧） | *"A person starts walking towards the left in a circular path and turns twice."* |
| **#03** | `03_044154_crate_walk` | `044154` | 开放式岛台 · 双手抱木箱前行（4帧） | *"A person is slowly walking forward while holding a crate with both hands in the front."* |
| **#04** | `04_016783_parcel_table` | `016783` | 胡桃木餐桌 · 双手抱纸箱停步（3帧） | *"A person carrying a medium-large heavy object with both hands takes two steps forward and then stops."* |
| **#05** | `05_098368_forward_jump` | `098368` | 居家运动垫 · 右足向前跃进（4帧） | *"The person swiftly takes a jump forward with their right foot."* |
| **#06** | `06_039936_stand_to_run` | `039936` | 玄关走廊 · 由站立爆发奔跑（4帧） | *"A woman stands in a field, then she runs forward. She lifts her legs quickly, arms swing back and forth, showing energy. She briefly pauses, then continues with big strides, looking excited and playful as she enjoys the outdoors under gray skies."* |
| **#07** | `07_038806_walk_to_jog` | `038806` | 书房走廊 · 由走转慢跑（4帧） | *"A man is walking and then starts to jog, moving his arms while shifting his weight from one leg to another. He appears focused as he energetically pushes forward, moving in a smooth motion that conveys an active and determined attitude."* |
| **#08** | `08_013842_onehand_carry` | `013842` | 玄关门厅 · 单手携物大步行走（4帧） | *"A person walks while carrying an object in one hand."* |
| **#09** | `09_032472_diagonal_crossover` | `032472` | 北欧客厅 · 交叉大步斜向横行（4帧） | *"The person is walking sideways toward the left diagonal by taking long strides in a crosswise manner at a moderate pace."* |
| **#10** | `10_114905_sideways_crossover` | `114905` | 现代客厅 · 交叉步匀速横移（3帧） | *"A person is steadily walking sideways to the right in a crossover pattern while the arms are swaying by their sides."* |
| **#11** | `11_100257_walk_backward` | `100257` | 书房空间 · 摆臂缓慢倒退走（4帧） | *"A person is slowly walking backward while swaying their arms."* |
| **#12** | `12_018149_relaxed_walk` | `018149` | 开放式餐厨 · 从容摆臂慢走（4帧） | *"A person is slowly walking forward while swaying their arms back and forth."* |
| **#13** | `13_100948_energetic_dance` | `100948` | 客厅地毯 · 展臂摆髋热舞（3帧） | *"A woman is dancing energetically. She lifts her arms to the sides, then moves her hips and legs side to side in rhythm. As she moves, her feet shuffle slightly, and she occasionally spins, showcasing her lively movement style. The overall vibe feels joyful and spirited, set in an indoor environment."* |
| **#14** | `14_078234_jump_dance` | `078234` | 北欧起居室 · 屈膝律动前跃舞步（3帧） | *"A man is enthusiastically dancing. He begins by bending his knees slightly and moving side to side. His arms are raised and he swings them as he shifts weight, then he steps forward with a playful bounce. The movements show energy and joy, making the dance feel lively and engaging."* |
| **#15** | `15_046600_martial_arts_punch` | `046600` | 榻榻米练习室 · 武术移重心出拳（3帧） | *"A man is performing martial arts movements. He starts with a strong stance, then quickly shifts his body weight to the side while thrusting his arm forward, executing swift and precise strikes. His legs remain stable as his arms alternate between striking and guarding, showing focus and determination throughout the routine."* |
| **#16** | `16_019543_side_kick` | `019543` | 居家健身角 · 屈膝交替侧踢腿（3帧） | *"A woman is performing an alternating side kick exercise. She starts by bending slightly forward, then raises her right leg to the side while keeping her balance, returning to the starting position, and repeating with her left leg. This movement is controlled and steady, showcasing her focus and determination."* |
| **#17** | `17_069096_crouch_sidewalk` | `069096` | 居家健身角 · 深蹲低姿横移（3帧） | *"A person in a crouch position starts to walk sideways in the right direction while facing forward."* |
| **#18** | `18_128701_overhead_strike` | `128701` | 榻榻米练习室 · 双手高举过头劈击（3帧） | *"A man is practicing a sword movement in a garden. He first raises the sword overhead with both hands, then steps forward while slicing downward, followed by a slight rotation of his body. He is precise and smooth, showing a strong focus and confidence in his movements."* |

---

## 3. 技术管线架构与踩坑备忘（接手必读）

整套流水线位于 `outputs/g1_generated_gallery/scripts/`，分为 4 个解耦阶段，任何后续 Agent 或开发者都可单独重跑其中一环：

### 阶段 1：原生 SOMA-77 → 朝向校准 BVH → Unitree G1 50Hz 重定向
- **脚本**：`retarget_candidates.py`
- **核心细节（踩坑点 1）**：
  - 我们的 `.soma.npz` 中 `rotmat77` 是局部旋转，链式相乘得到纯父子世界旋转 `W(j)`；而 NVIDIA 官方 `soma-retargeter` 期望输入的 BVH 隐含 `W(j) @ O[j]`（其中 `O[j]` 是 `soma_joint_orient.npz` 中的 78 关节静态参考系）。
  - 脚本内通过共轭变换 `L'(k) = O[parent]^T @ L(k) @ O[k]` 与骨长实测恢复（`bone_lengths`），与原始 `joints77` 的正向运动学位置误差验证为 **`0.000 mm`**。
  - 调用 `soma_retargeter.pipelines.newton_pipeline.NewtonPipeline(skel, "soma", "unitree_g1")` 在 CUDA 上执行 24 步 LM IK + 脚底稳定器（`feet_stabilizer`），输出 50 Hz CSV。

### 阶段 2：NVIDIA `GEAR-SONIC` 200Hz MuJoCo 闭环物理跟踪仿真
- **脚本**：`simulate_sonic_g1.py`
- **依赖模型**：HuggingFace `nvidia/GEAR-SONIC` 发布的 `model_encoder.onnx`（输入 1762 维，输出 64 维 token）+ `model_decoder.onnx`（输入 994 维，输出 29 维动作）+ `observation_config.yaml`。
- **核心细节（踩坑点 2 & 3）**：
  - **关节顺序映射**：MuJoCo XML 与 `soma-retargeter` CSV 采用 **DFS 顺序**（左腿6 → 右腿6 → 腰3 → 左臂7 → 右臂7），而 SONIC 策略观测与输出动作采用 **IsaacLab BFS 顺序**。必须在观测输入侧用 `MUJOCO_TO_ISAACLAB` 置换，在动作输出侧用 `ISAACLAB_TO_MUJOCO` 还原（常量严格取自 `policy_parameters.hpp`）。
  - **电机转子惯量（Armature）**：直接加载裸 `g1_29dof.xml` 而不设置 `dof_armature` 会导致手腕/脚踝小惯量关节在高增益 PD 控制下第 0.05s 数值发散（`QACC NaN`）。必须按 `policy_parameters.hpp` 写入电机转子惯量（`ARMATURE_7520_22 / 7520_14 / 5020 / 4010`）并启用 `mjINT_IMPLICITFAST`（50 Hz 控制、200 Hz 物理步进，每帧 4 子步）。
  - **冷启动伪帧剔除**：`soma-retargeter` 输出的 CSV 第 0 行是 IK 种子帧（与第 1 行跳变约 29°），脚本自动剔除冷启动跳变帧并在第 0 帧做 15 个控制周期（0.3s）物理驻留沉降后再开始跟踪，35+ 条生成动作闭环不倒率达 **97%+**。

### 阶段 3：G1 非重叠视觉网格与世界系 FK 导出
- **脚本**：`export_g1_fk.py`
- **核心细节（踩坑点 4）**：
  - `g1_29dof.xml` 中每个连杆同时定义了 `group=1, contype=0` 的视觉网格和 `group=0, contype=1` 的同名碰撞网格。导出到 Blender 时**必须只保留 `group == 1` 的 36 个视觉几何体**，否则两个完全重合的网格会在 Cycles 光追中产生严重的黑斑 Z-fighting。
  - 自动读取 `left_rubber_hand` / `right_rubber_hand` 等完整手部网格，并按全片段最低支撑脚底第 10 分位校准 `z = 0.000 m`，保证既不悬空也不穿地。

### 阶段 4：Blender 4.5 OptiX 居家场景自动构图、分层光追与线性空间频闪合成
- **脚本**：`render_teaser_v3.py`（4 格定制比例渲染）、`render_gallery_18.py`（18 组居家场景批量渲染）、`compose_gallery_18.py`（线性空间合成与配字拼版）。
- **核心细节**：
  1. **解析式视锥 NDC 自动构图（`solve_auto_camera`）**：利用全部定格姿态 36 个网格的 8 顶点世界包围盒做二分距离与视点求解，严格保证所有机器人的横向/纵向投影落在目标区间（如 `u ∈ [0.09, 0.91]`），**彻底杜绝画面边缘裁切脚尖或背部**。
  2. **三通道分层渲染避免内透脏影**：如果在 3D 里直接把机器人材质设成半透明，关节内部重叠的零件会互相透视发黑。管线对每个镜头分别渲染：`bg.png`（纯空房间光追底图）、`pose_k.png`（第 `k` 个时刻的 100% 实体光追图，含真实地板软阴影与木地板反射）、`mask_k.png`（房间设为 `is_holdout=True` 的亚像素级遮罩）。
  3. **sRGB ↔ Linear RGB 物理合成**：在 `compose_gallery_18.py` 中先转线性光空间，提取遮罩外的真实光追阴影比率叠到底图，再按时间递增权重（如 `0.44 → 0.63 → 0.81 → 1.00`）叠加实体着色机器人，最后转回 sRGB。

---

## 4. 常用后续操作指南

### 4.1 从 18 张图中任选 4 张重新拼出 `3840 × 1560` 论文首页 Teaser（无需 GPU，本机秒出！）

当你从 `outputs/g1_generated_gallery/index.html` 挑定最满意的 4 张图（例如分别作为 `(a)` 左大图、`(b)` 中上、`(c)` 中下、`(d)` 右竖图）后，直接在本机运行以下命令即可立刻合成新的 4K 首页 Teaser：

```bash
cd /Users/a26044/projects/MotionViewer
.venv/bin/python - << 'PY'
import sys
from pathlib import Path
from PIL import Image

# 直接复用已同步到本地的合成器函数
sys.path.insert(0, "outputs/g1_generated_gallery/scripts")
import compose_gallery_18 as cg

clean_dir = Path("outputs/g1_generated_gallery/clean")
shots_map = {p.stem: (Image.open(p), {}) for p in sorted(clean_dir.glob("*.png"))}

# 把下面 4 个名字替换为你挑中的 4 张图文件名（不含 .png）即可：
PICK_4 = (
    "02_p39_000170_curved_walk",    # (a) 左侧主视觉大图 (1860 x 1560)
    "04_016783_parcel_table",       # (b) 中上交互图 (1120 x 774)
    "17_069096_crouch_sidewalk",    # (c) 中下低姿图 (1120 x 774)
    "06_039936_stand_to_run",       # (d) 右侧冲刺竖图 (836 x 1560)
)

out_file = Path("outputs/g1_generated_gallery/teasers/teaser_custom_pick.png")
cg.assemble_omni_teaser(out_file, shots_map, PICK_4)
print("Saved custom 4K teaser to:", out_file)
PY
```
> **提示**：如果选定了最终 4 个分镜，想按每个分镜精确长宽比（如 `(a)` `1860×1560`、`(d)` `836×1560` 竖构图）在 Blender 中原生重渲一版零裁切终稿，只需把对应 4 个 `cid` 和 `frames` 填入 `scripts/render_teaser_v3.py` 的 `panels` 列表，在任意 CUDA 节点或本机 `.local/Blender.app` 运行约 2.5 分钟即可。

### 4.2 在新申请的 CUDA 节点上从零恢复整套环境（约 2 分钟）

当前远程节点 `a26163@10.239.121.12 -p 31284` 上的工作目录为 `/tmp/g1_teaser/`（含后台保活脚本 `/tmp/g1_teaser/keepalive_loop.sh`；**若不再使用该 4090 节点，请执行 `pkill -f keepalive_loop.sh` 让集群正常回收显卡**）。

若后续节点释放、在新容器节点重跑（`/home/a26163` 为 CephFS 持久化共享盘，`soma-retargeter`、`GR00T-WholeBodyControl` 与 `t2m_demo400` 均永久保留在 `~/projects/` 下），只需执行：

```bash
# 1. 把本地保存的脚本推到新节点
scp -P <NEW_PORT> -r outputs/g1_generated_gallery/scripts a26163@<HOST>:/tmp/g1_teaser_scripts

# 2. 在新节点一键拉起 Blender 4.5 LTS + MuJoCo/ONNX 环境 + SONIC 权重
ssh -p <NEW_PORT> a26163@<HOST> '
  set -e
  sudo apt-get update -qq && sudo apt-get install -y -qq libxfixes3 libxi6 libxkbcommon0 libxxf86vm1 libxrender1 libgl1 libsm6 libice6 libxext6
  mkdir -p /tmp/g1_teaser && cp /tmp/g1_teaser_scripts/* /tmp/g1_teaser/
  cd /tmp/g1_teaser
  curl -L -sS -o blender.tar.xz https://download.blender.org/release/Blender4.5/blender-4.5.16-linux-x64.tar.xz
  mkdir -p blender && tar -xf blender.tar.xz --strip-components=1 -C blender && rm -f blender.tar.xz
  ~/.local/bin/uv venv /tmp/g1_teaser/.venv --python 3.12
  ~/.local/bin/uv pip install --python /tmp/g1_teaser/.venv/bin/python mujoco onnxruntime huggingface_hub scipy pillow matplotlib pyyaml numpy
  /tmp/g1_teaser/.venv/bin/python ~/projects/third_party/GR00T-WholeBodyControl/download_from_hf.py --output-dir /tmp/g1_teaser/sonic_deploy --no-planner
  ln -sfn ~/projects/third_party/GR00T-WholeBodyControl/gear_sonic_deploy/g1/meshes /tmp/g1_teaser/meshes
'

# 3. 按需运行四步流水线：
#   (a) 重定向新 SOMA 动作：cd ~/projects/third_party/soma-retargeter && .venv/bin/python /tmp/g1_teaser/retarget_candidates.py
#   (b) SONIC 200Hz 闭环物理仿真：/tmp/g1_teaser/.venv/bin/python /tmp/g1_teaser/simulate_sonic_g1.py
#   (c) 导出 G1 视觉网格与 FK：/tmp/g1_teaser/.venv/bin/python /tmp/g1_teaser/export_g1_fk.py
#   (d) Blender OptiX 渲染与合成：/tmp/g1_teaser/blender/blender -b -P /tmp/g1_teaser/render_gallery_18.py && /tmp/g1_teaser/.venv/bin/python /tmp/g1_teaser/compose_gallery_18.py
```

---

## 5. 目录结构速查

```text
outputs/g1_generated_gallery/          # 本次 100% 模型生成选片主目录（Git 忽略，本地保留）
├── index.html                         # 本地交互式深色选片画廊网页（含双拼 Teaser + 18 张大图与中英 Prompt）
├── manifest.json                      # 18 组镜头的完整元数据（帧号、NDC 范围、英文原句、数据来源）
├── captioned/                         # 18 张带编号与底部完整英文 Text Prompt 的成图 (1600×1002 JPG)
├── clean/                             # 18 张无字纯净原图 (1600×920 PNG，供论文自由排版)
├── inspect/                           # 轻量级速览图（含 overview_sheet_1.jpg / overview_sheet_2.jpg）
├── teasers/                           # 预拼好的 3840×1560 4格首页 Teaser (teaser_gen_option1/2.png)
└── scripts/                           # 全套可复现 Python/Bash 源码与物理评测 summary.json
```
