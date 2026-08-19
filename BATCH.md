# 批量生成操作指南

一条命令搞定。所有设置都已经是调好的默认值，你不需要传任何调参选项。

---

## 0. 前提检查（30 秒）

```bash
cd /Users/a26044/projects/MotionViewer
uv run python scripts/visualize_dataset.py --list
```

应该看到你放进去的每个任务，每个都是 `N clips, N valid, 0 unusable`：

```
live500/gen: 500 clips, 500 valid, 0 unusable
new500/gen:  500 clips, 500 valid, 0 unusable
old500/gen:  500 clips, 500 valid, 0 unusable
picks38/gen:  39 clips,  39 valid, 0 unusable
picks38/gt:   39 clips,  39 valid, 0 unusable
```

**任务名是从目录结构自动生成的** —— 相对路径去掉 `_smplx` 后缀：

```
/Users/a26044/Motion data/live500_smplx/gen/000123.smplx.npz   →   任务名 live500/gen
```

以后新放一个 `xxx_smplx/gen/` 进去，任务名自动就是 `xxx/gen`，不用改代码。

如果报 `--data ... is not a directory`，说明数据不在 `/Users/a26044/Motion data`，加 `--data <你的路径>`。

---

## 1. 最常用的三条命令

**A. picks38 的 gen vs gt 并排对比**（论文里最有说服力的一种）

```bash
cd /Users/a26044/projects/MotionViewer
nohup uv run python -u scripts/visualize_dataset.py \
  --task picks38/gen --compare picks38/gt \
  --output converted/picks38_compare \
  > outputs/logs/picks38.log 2>&1 &
```

- 39 条视频，每条 2074×1440，左 gt 右 gen，同一相机
- **约 4 小时**

**B. 单独渲某个任务**（不做对比）

```bash
nohup uv run python -u scripts/visualize_dataset.py \
  --task old500/gen --output converted/old500 \
  > outputs/logs/old500.log 2>&1 &
```

- 500 条，每条 1037×1440
- **约 12 小时**（`new500` 同理）

**C. 先小批量试水**（推荐第一次这么做）

```bash
uv run python scripts/visualize_dataset.py \
  --task old500/gen --limit 5 --output converted/试水
```

不加 `nohup`，直接在终端跑完，5 条大概 10 分钟。看着满意再跑全量。

---

## 2. 怎么看进度

```bash
tail -f outputs/logs/picks38.log
```

每条会打印一行：

```
[12/39] picks38-gen_011124_yoimiya  300f  act 0.61  6891 frames left  eta 69 min
```

`eta` 是按已完成的帧速算的，前两条不准，之后就准了。

想只看有没有出错：

```bash
grep -E "failed|failure" outputs/logs/picks38.log
```

---

## 3. 怎么看结果

```bash
open converted/picks38_compare/index.html
```

**边跑边看** —— 页面每渲完一条就更新一次，不用等全部结束。

页面上：
- 顶部搜索框可以按 caption、片段号、角色筛（比如输 `dance`、`kick`）
- 下拉框可以按「最活跃」/「最干净」/「最长」重排
- 橙色徽章标的是**源动作本身**的问题（脚滑、抖动、陷地），不是渲染问题，故意留着让你能看见

---

## 4. 中断了怎么办

**直接把同一条命令再跑一次。** 已经渲好的会跳过，从断点继续。

```bash
# 想停：
pkill -f visualize_dataset.py

# 想接着跑：把原来那条命令原样再执行一次
```

同一条命令重跑还会**顺便刷新已渲片段的指标**（不重渲），所以指标改了也不用重来。

---

## 5. 换角色

```bash
--character yoimiya      # 宵宫（默认）
--character furina       # 芙宁娜
--character silverwolf   # 银狼
--character furina_wild  # 芙宁娜（荒）
```

可以叠加，一次渲多个角色（时间成倍）：

```bash
--character yoimiya --character furina
```

---

## 6. 挑片段

```bash
--clips 000015,028509,045969   # 只渲这几条
--limit 50                     # 只渲前 50 条
--sort activity                # 排序：activity(默认) / penalty / frames / travel / name
--min-frames 30                # 太短的跳过
--max-penalty 2                # 只渲质量分好于 2 的（0 = 不筛）
```

`--sort activity` 是「动作幅度大的优先」，`--sort penalty` 是「最干净的优先」。挑论文素材建议先 `--sort activity --limit 60` 渲一批，再从页面上挑。

---

## 7. 画质和时长的取舍

默认 1440p。想更快或更清晰：

```bash
--resolution 1080    # 约 0.7 倍时间，细节 -12%
--resolution 1440    # 默认
--resolution 1920    # 约 1.3 倍时间，细节 +10%
```

500 条在 1080p 下大约 **8.5 小时**，1440p 约 **12 小时**。

---

## 8. 输出在哪，text 在哪

```
converted/<你指定的名字>/
├── index.html          ← 打开这个，页面上每条视频下面就是它的 caption
├── captions.tsv        ← 视频文件名 → caption 的对照表，可直接用 Excel/Numbers 打开
├── videos/
│   ├── live500-gen_023908_yoimiya__the-person-is-performing-a-yoga-pose.mp4
│   └── live500-gen_023908_yoimiya__the-person-is-performing-a-yoga-pose.txt   ← 完整 caption
├── manifest.json       ← 每条的全部指标 + caption + 源文件路径
└── trimmed/            ← 修剪过的中间 npz（可以删，重跑会重建）
```

**动作对应的 text 有三个地方可以拿**：

1. **文件名里就带**：`<任务>_<片段号>_<角色>__<caption 前 56 字符>`，在 Finder 里直接能看。
   不想要这段的话加 `--no-caption-in-name`。
2. **每个视频旁边一个同名 `.txt`**，里面是完整 caption —— 单独把 mp4 拷走时把 txt 一起带上就行。
3. **`captions.tsv`**，一行一条，制表符分隔，列是
   `video / task / clip_id / character / frames / caption`。做论文表格或批量筛选用这个最省事。

中间帧会边渲边删，不会堆磁盘。500 条视频大约 **1.5 GB**。

---

## 9. 可能遇到的问题

| 现象 | 原因 / 处理 |
|---|---|
| `no task matched` | `--task` 名字写错，先跑 `--list` 看可选名字 |
| `character asset missing` | `assets/` 下缺模型文件，模型没提交进仓库，需要本地放好 |
| 某条 `render failed` | 单条失败不影响其它，跑完会汇总列出；重跑同命令会重试它 |
| 磁盘满 | 删 `converted/*/trimmed/`，或删掉不要的 `converted/` 目录 |
| 电脑要睡眠 | `caffeinate -i nohup uv run python -u ...`（前面加 `caffeinate -i`） |

---

## 10. 一次跑完全部（如果你就想挂着不管）

```bash
cd /Users/a26044/projects/MotionViewer
caffeinate -i bash -c '
  uv run python -u scripts/visualize_dataset.py --task picks38/gen --compare picks38/gt \
    --output converted/picks38_compare
  uv run python -u scripts/visualize_dataset.py --task old500/gen --output converted/old500
  uv run python -u scripts/visualize_dataset.py --task new500/gen --output converted/new500
' > outputs/logs/all.log 2>&1 &
```

三批依次跑，总共约 **28 小时**。中间任何时候可以 `pkill -f visualize_dataset.py` 停下，再把同样的命令跑一遍接着来。

---

## 附：当前已定稿的效果设置（都是默认值，无需传）

| 项目 | 设置 | 为什么 |
|---|---|---|
| 分辨率 | 1440p/面板 | 比 800p 多 56% 边缘细节，1920p 只多 16% 但慢 30% |
| 像素滤波 | 0.9 | Blender 默认 1.5 会柔化每条边，赛璐璃损失最大 |
| EEVEE 采样 | 64 | 128 实测无差别（5.38 vs 5.41），纯浪费一倍时间 |
| 编码 | crf 16 | PSNR 47.0 dB，大面积平色不出色带 |
| 相机 | follow | 角色大小不随走多远而变；静态相机在 6 m 位移下只剩 100 px 高 |
| 地面 | 0.5 m 淡网格 | 跟随相机在无特征地面上会让走路变成跑步机 |
| 描边 | 逐材质授权色 + 1.6 px | 头发暖棕、衣服黑，是模型自己指定的；固定像素宽不随分辨率变粗 |
| 表情 | `smile`（眉 0.5 + 嘴角 0.75） | 不动眼睛 —— 全身尺度下眯眼会让虹膜消失 |
| 修剪 | auto | 自动丢掉 gen 的第 0 帧锚定姿态 / old500 的 14 帧静止前缀 |
| 平滑 | 关 | 会改动作，需要你明确要求（`--smooth 5`） |
