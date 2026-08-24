"""Interactive clip picker: browse the motion sets in 3D and mark the ones to render.

Rendering 1578 clips on a character takes days; picking the fifty worth it takes an
afternoon, and picking them from a still frame or a filename does not work. This is
a Viser server over the same dataset tree ``visualize_dataset.py`` reads, showing the
SMPL-X body-22 skeleton animated, the caption beside it, and a Keep button whose
output feeds straight back in::

    uv run --group viz python scripts/pick_clips.py
    # open the URL it prints, mark clips, then:
    uv run python scripts/visualize_dataset.py --task live500/gen \\
        --clips "$(cat outputs/picks/live500-gen.txt)" --output converted/picked

Design notes:

- **Loads on demand and caches.** All 1578 clips' joints are only 61 MB, so this
  could preload, but startup latency is what makes a picker unpleasant to use.
- **Skeleton, not the character.** The point is to judge the *motion*; putting Yoimiya
  on it costs a Blender launch per clip and hides the motion behind hair and skirt.
- **Picks and tags are written on every change**, so closing the tab never loses them.
- **Tags are free text you type per clip**, kept in their own file so un-keeping a clip
  does not lose the note. A caption says what the model was asked for; a tag says what
  you actually saw and why you kept it.
- **The same trimming the renderer applies** is applied here, so what you judge is
  what gets rendered — a clip whose first frame is the conditioning pose would
  otherwise show a jump the render will not have.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from visualize_dataset import DEFAULT_DATA, detect_trim, discover, preprocess, validate  # noqa: E402

from motionviewer.core.smplx_fk import SMPLX_BODY22_PARENTS  # noqa: E402

# Bones as (parent, child) index pairs, which is what the batched mesh needs.
EDGES = np.array(
    [(parent, child) for child, parent in enumerate(SMPLX_BODY22_PARENTS) if parent >= 0],
    dtype=np.int64,
)

_KEPT = np.array([102, 187, 106], np.uint8)
_PLAIN = np.array([120, 144, 156], np.uint8)


def _bone_mesh(width: float = 0.035, knuckle: float = 0.12):
    """Unit octahedral bone along +Y, 0 to 1. Scaled per bone by its length."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [width, knuckle, 0.0],
            [0.0, knuckle, width],
            [-width, knuckle, 0.0],
            [0.0, knuckle, -width],
        ],
        np.float32,
    )
    faces = np.array(
        [
            [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 2],
            [1, 3, 2], [1, 4, 3], [1, 5, 4], [1, 2, 5],
        ],
        np.int32,
    )  # fmt: skip
    return vertices, faces


def _quat_from_y(directions: np.ndarray) -> np.ndarray:
    """Per-row quaternions (wxyz) taking +Y onto each unit direction."""
    up = np.array([0.0, 1.0, 0.0], np.float32)
    target = directions.astype(np.float32)
    dot = np.clip(target[:, 1], -1.0, 1.0)
    axis = np.cross(np.broadcast_to(up, target.shape), target)
    length = np.linalg.norm(axis, axis=1)
    angle = np.arccos(dot)
    quaternions = np.zeros((len(target), 4), np.float32)
    safe = length > 1e-6
    unit = np.zeros_like(target)
    unit[safe] = axis[safe] / length[safe, None]
    quaternions[:, 0] = np.cos(angle / 2.0)
    quaternions[:, 1:] = unit * np.sin(angle / 2.0)[:, None]
    quaternions[(~safe) & (dot > 0)] = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    quaternions[(~safe) & (dot < 0)] = np.array([0.0, 1.0, 0.0, 0.0], np.float32)
    return quaternions


@dataclass
class Loaded:
    joints: np.ndarray  # (T, 22, 3), Y-up, grounded, root-centred horizontally
    fps: float
    caption: str
    trim: int
    trim_reason: str
    score: dict = field(default_factory=dict)
    # Horizontal root motion, so "does this clip drift?" is a number rather than an
    # impression. `span` is the bounding extent of the path, `net` the start-to-end
    # displacement: a clip that wanders and comes back has a large span and small net.
    drift_span_m: float = 0.0
    drift_net_m: float = 0.0


def load(path: Path, trim_mode: str, smooth: int) -> Loaded | None:
    payload, error = validate(path)
    if payload is None:
        print(f"skip {path.name}: {error}", flush=True)
        return None
    trim, reason = (0, "") if trim_mode == "none" else detect_trim(payload["joints22"])
    preprocess(payload, trim, smooth)
    joints = np.asarray(payload["joints22"], dtype=np.float32)

    # Ground on the lower quartile of the per-frame lowest joint, not on the clip
    # minimum. The minimum puts exactly one frame on the grid and leaves every other
    # frame hovering: measured over eight live500 clips, that was 2.2-5.1 cm at the
    # median and 18.3 cm at worst, which on a 1.6 m figure reads as the whole skeleton
    # drifting up and down off the floor. A quartile lets the planted frames touch and
    # the lowest few dip a centimetre under, and it matches how the renderer places its
    # floor, so the picker shows what will actually be rendered.
    per_frame_low = joints[..., 1].min(axis=1)
    joints = joints - np.array([0.0, float(np.percentile(per_frame_low, 25)), 0.0], np.float32)

    # Only the horizontal root offset is removed, and only its frame-0 value, so the
    # figure starts centred but its travel stays visible — a clip that runs away is
    # something to see, not to hide.
    centre = joints[0, 0].copy()
    centre[1] = 0.0
    joints = joints - centre
    root = joints[:, 0][:, [0, 2]].astype(np.float64)
    return Loaded(
        joints=joints,
        fps=payload["_fps"],
        caption=payload["_caption"],
        trim=trim,
        trim_reason=reason,
        drift_span_m=float(np.linalg.norm(root.max(axis=0) - root.min(axis=0))),
        drift_net_m=float(np.linalg.norm(root[-1] - root[0])),
    )


def picks_file(picks_dir: Path, task: str) -> Path:
    """The comma-separated id list, shaped to paste straight into ``--clips``."""
    return picks_dir / f"{task.replace('/', '-')}.txt"


def tags_file(picks_dir: Path, task: str) -> Path:
    """Tags live in their own file, so un-keeping a clip does not lose its note."""
    return picks_dir / f"{task.replace('/', '-')}.tags.json"


def read_picks(picks_dir: Path, task: str) -> list[str]:
    path = picks_file(picks_dir, task)
    if not path.is_file():
        return []
    return [item for item in path.read_text(encoding="utf-8").strip().split(",") if item]


def read_tags(picks_dir: Path, task: str) -> dict[str, str]:
    path = tags_file(picks_dir, task)
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # A half-written file must not take the session down; the worst case is that
        # one tag is lost, and the next edit rewrites the file cleanly.
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(key): str(value) for key, value in loaded.items() if str(value).strip()}


def write_tags(picks_dir: Path, task: str, tags: dict[str, str]) -> None:
    kept = {key: value.strip() for key, value in tags.items() if value.strip()}
    tags_file(picks_dir, task).write_text(
        json.dumps(dict(sorted(kept.items())), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_shortlist(
    picks_dir: Path,
    task: str,
    ids: list[str],
    tags: dict[str, str],
    meta: dict[str, tuple[str, int]],
) -> None:
    """Write both halves of a shortlist: the id list and a readable JSON beside it.

    The JSON carries the caption *and* the tag, because they answer different
    questions later: a caption is what the model was asked for, a tag is what you
    actually saw and why you kept it.
    """
    ordered = sorted(set(ids))
    picks_file(picks_dir, task).write_text(",".join(ordered) + ("\n" if ordered else ""), encoding="utf-8")
    detail = [
        {
            "clip_id": item,
            "tag": tags.get(item, ""),
            "caption": meta[item][0],
            "frames": meta[item][1],
        }
        for item in ordered
        if item in meta
    ]
    picks_file(picks_dir, task).with_suffix(".json").write_text(
        json.dumps({"task": task, "clips": detail}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_summaries(path: Path) -> dict[str, str]:
    """Short Chinese one-liners keyed by clip id, for the row under the caption.

    The generated captions run 30-60 words, which the Viser text row cannot show, so
    the summary is what is actually readable while picking. It is a lookup rather than
    something derived here: a useful summary needs to know that "steps in shit then
    shakes it" is about the foot, and no rule over the English text gets that right.

    A missing or malformed file is not fatal — the picker is still usable without it,
    and inventing a summary from the first few words would be worse than showing none.
    """
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"[picker] ignoring {path}: {exc}", flush=True)
        return {}
    if not isinstance(raw, dict):
        return {}
    # Accept either {clip: text} or {task: {clip: text}}; later sets win ties, and in
    # this data a clip id carries the same caption in every set it appears in.
    flat: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            flat[str(key)] = value
        elif isinstance(value, dict):
            flat.update({str(k): str(v) for k, v in value.items() if isinstance(v, str)})
    return flat


def read_padding_meta(path: Path) -> dict[str, dict]:
    """``meta.json`` rows keyed by clip id, for the padding marker.

    Clips whose ground truth is shorter than 3 s were generated to 3 s anyway, so part
    of what plays is not the model answering the caption: in ``active``/``quiet`` the
    tail is extrapolation, and in the ``*_hold`` variant it is a frozen last frame.
    Neither is visible in the skeleton — a frozen tail looks like a deliberate hold —
    so it has to be read off the metadata and shown.
    """
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"[picker] ignoring {path}: {exc}", flush=True)
        return {}
    if not isinstance(rows, list):
        return {}
    return {str(row["stem"]): row for row in rows if isinstance(row, dict) and "stem" in row}


def padding_note(task: str, clip_id: str, frames: int, meta: dict[str, dict]) -> str:
    """One line saying how much of this clip is real motion."""
    row = meta.get(clip_id)
    if row is None:
        return ""
    gt = int(row.get("gt_frames") or 0)
    if not row.get("padded"):
        return f"原生长度 {frames}f，未补长"
    if task.endswith("_hold"):
        frozen = max(0, frames - gt)
        return f"⏸ 冻结末帧：GT {gt}f 有动作 + 后 {frozen}f 静止"
    extra = max(0, frames - gt)
    return f"⚠ 已补长：GT {gt}f + 模型外推 {extra}f（有 _hold 版可对照）"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--task", action="append", default=None, help="Repeatable, glob allowed")
    parser.add_argument("--picks", type=Path, default=ROOT / "outputs/picks")
    parser.add_argument(
        "--summaries",
        type=Path,
        default=ROOT / "data/caption_zh.json",
        help="clip id -> short Chinese summary, shown under the full caption",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="dataset meta.json, for the padding marker; defaults to <data>/../meta.json",
    )
    parser.add_argument("--trim", default="auto", help="Match the renderer's trimming; 'none' to see raw")
    parser.add_argument("--smooth", type=int, default=0)
    parser.add_argument("--sort", default="activity", choices=("activity", "penalty", "frames", "name"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    import viser

    tasks = discover(args.data, args.task or ["*"], args.trim, args.smooth)
    keys = {
        "activity": lambda clip: -clip.score["activity"],
        "penalty": lambda clip: clip.score["penalty"],
        "frames": lambda clip: -clip.frames,
        "name": lambda clip: clip.clip_id,
    }
    catalogue: dict[str, list] = {}
    for task, clips in tasks.items():
        usable = sorted((clip for clip in clips if not clip.error), key=keys[args.sort])
        if usable:
            catalogue[task] = usable
    if not catalogue:
        raise SystemExit(f"no usable clips under {args.data}")
    for task, clips in catalogue.items():
        print(f"{task}: {len(clips)} clips", flush=True)

    args.picks.mkdir(parents=True, exist_ok=True)

    summaries = read_summaries(args.summaries)
    if summaries:
        covered = sum(1 for clips in catalogue.values() for clip in clips if clip.clip_id in summaries)
        total = sum(len(clips) for clips in catalogue.values())
        print(f"[picker] 摘要 {covered}/{total} clips from {args.summaries}", flush=True)
    else:
        print(f"[picker] no summaries at {args.summaries}; the 摘要 row will read (无摘要)", flush=True)

    meta_path = args.meta or (args.data.parent / "meta.json")
    padding = read_padding_meta(meta_path)
    if padding:
        n = sum(1 for row in padding.values() if row.get("padded"))
        print(f"[picker] 补长标记 {len(padding)} clips from {meta_path} ({n} padded)", flush=True)
    else:
        print(f"[picker] no meta.json at {meta_path}; the 补长 row stays blank", flush=True)

    def meta_for(task: str) -> dict[str, tuple[str, int]]:
        return {clip.clip_id: (clip.caption, clip.frames) for clip in catalogue[task]}

    def write_picks(task: str, ids: list[str]) -> None:
        write_shortlist(args.picks, task, ids, read_tags(args.picks, task), meta_for(task))

    def read_picks_now(task: str) -> list[str]:
        return read_picks(args.picks, task)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+y")
    server.scene.add_grid(
        "/ground",
        plane="xz",
        position=(0.0, 0.0, 0.0),
        infinite_grid=True,
        cell_size=0.5,
        section_size=1.0,
        fade_distance=14.0,
    )

    task_names = list(catalogue)
    gui_task = server.gui.add_dropdown("Set", options=task_names, initial_value=task_names[0])
    gui_clip = server.gui.add_dropdown(
        "Clip", options=[clip.clip_id for clip in catalogue[task_names[0]]], initial_value=None
    )
    gui_caption = server.gui.add_text("Caption", initial_value="", disabled=True)
    gui_summary = server.gui.add_text("摘要", initial_value="", disabled=True)
    gui_padding = server.gui.add_text("补长", initial_value="", disabled=True)
    gui_stats = server.gui.add_text("Clip", initial_value="", disabled=True)
    gui_tags = server.gui.add_text("Tags", initial_value="")
    gui_keep = server.gui.add_button("Keep this clip")
    gui_drop = server.gui.add_button("Un-keep")
    gui_prev = server.gui.add_button("< Prev")
    gui_next = server.gui.add_button("Next >")
    gui_kept = server.gui.add_text("Kept", initial_value="", disabled=True)
    gui_play = server.gui.add_checkbox("Play", initial_value=True)
    gui_speed = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
    gui_frame = server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    gui_trail = server.gui.add_checkbox("Show root trail", initial_value=True)

    bone_vertices, bone_faces = _bone_mesh()
    state: dict = {
        "loaded": None,
        "handle": None,
        "trail": None,
        "cache": {},
        # Set while show() populates the tag box, so the on_update it triggers does not
        # write the value straight back to disk.
        "loading_tag": False,
    }

    def kept_ids() -> list[str]:
        return read_picks_now(gui_task.value)

    def refresh_kept_label() -> None:
        task = gui_task.value
        ids = kept_ids()
        total = len(catalogue[task])
        tagged = len(read_tags(args.picks, task))
        gui_kept.value = f"{len(ids)}/{total} kept, {tagged} tagged -> {picks_file(args.picks, task).name}"

    def draw(frame: int) -> None:
        loaded: Loaded | None = state["loaded"]
        if loaded is None:
            return
        index = min(frame, len(loaded.joints) - 1)
        points = loaded.joints[index]
        start = points[EDGES[:, 0]]
        delta = points[EDGES[:, 1]] - start
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length, 1e-6)[:, None]
        colour = _KEPT if gui_clip.value in kept_ids() else _PLAIN
        handle = state["handle"]
        with server.atomic():
            handle.batched_positions = start.astype(np.float32)
            handle.batched_wxyzs = _quat_from_y(direction)
            handle.batched_scales = length.astype(np.float32)
            handle.batched_colors = colour
        gui_stats.value = (
            f"{index + 1}/{len(loaded.joints)} frames  {index / loaded.fps:.1f}s  ·  "
            f"drift {loaded.drift_span_m:.2f} m span / {loaded.drift_net_m:.2f} m net  ·  "
            f"foot {points[[10, 11], 1].min() * 100:+.1f} cm off floor"
            f"{'  ·  ' + loaded.trim_reason if loaded.trim_reason else ''}"
        )

    def show(clip_id: str) -> None:
        task = gui_task.value
        cache_key = (task, clip_id)
        loaded = state["cache"].get(cache_key)
        if loaded is None:
            clip = next(item for item in catalogue[task] if item.clip_id == clip_id)
            loaded = load(clip.path, args.trim, args.smooth)
            if loaded is None:
                return
            loaded.score = clip.score
            state["cache"][cache_key] = loaded
        state["loaded"] = loaded

        if state["handle"] is None:
            count = len(EDGES)
            state["handle"] = server.scene.add_batched_meshes_simple(
                "/skeleton",
                bone_vertices,
                bone_faces,
                np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (count, 1)),
                np.zeros((count, 3), np.float32),
                batched_scales=np.ones(count, np.float32),
                batched_colors=_PLAIN,
                flat_shading=True,
            )

        if state["trail"] is not None:
            state["trail"].remove()
            state["trail"] = None
        if gui_trail.value and len(loaded.joints) > 1:
            root = loaded.joints[:, 0].copy()
            root[:, 1] = 0.01
            state["trail"] = server.scene.add_spline_catmull_rom(
                "/trail", positions=root.astype(np.float32), color=(255, 170, 60), line_width=2.0
            )

        state["loading_tag"] = True
        try:
            gui_tags.value = read_tags(args.picks, task).get(clip_id, "")
        finally:
            state["loading_tag"] = False

        score = loaded.score
        gui_caption.value = loaded.caption or "(no caption)"
        gui_summary.value = summaries.get(clip_id) or "(无摘要)"
        gui_padding.value = padding_note(task, clip_id, len(loaded.joints), padding)
        gui_frame.max = max(1, len(loaded.joints) - 1)
        gui_frame.value = 0
        badges = []
        if score.get("ground_penetration_m", 0.0) > 0.05:
            badges.append(f"sinks {score['ground_penetration_m'] * 100:.0f}cm")
        if score.get("foot_skate_m_s", 0.0) > 0.35:
            badges.append(f"skate {score['foot_skate_m_s']:.2f}")
        if score.get("jitter_deg", 0.0) > 0.5:
            badges.append(f"jitter {score['jitter_deg']:.2f}deg")
        if badges:
            gui_caption.value = f"{gui_caption.value}   [{', '.join(badges)}]"
        refresh_kept_label()
        draw(0)

    def switch_task(task: str) -> None:
        ids = [clip.clip_id for clip in catalogue[task]]
        gui_clip.options = ids
        gui_clip.value = ids[0]
        show(ids[0])

    @gui_task.on_update
    def _(_=None) -> None:
        switch_task(gui_task.value)

    @gui_clip.on_update
    def _(_=None) -> None:
        if gui_clip.value:
            show(gui_clip.value)

    @gui_tags.on_update
    def _(_=None) -> None:
        if state["loading_tag"] or not gui_clip.value:
            return
        task = gui_task.value
        tags = read_tags(args.picks, task)
        tags[gui_clip.value] = gui_tags.value
        write_tags(args.picks, task, tags)
        write_picks(task, read_picks_now(task))
        refresh_kept_label()

    @gui_frame.on_update
    def _(_=None) -> None:
        draw(int(gui_frame.value))

    @gui_trail.on_update
    def _(_=None) -> None:
        if gui_clip.value:
            show(gui_clip.value)

    def advance(step: int = 1) -> None:
        ids = [clip.clip_id for clip in catalogue[gui_task.value]]
        if not ids:
            return
        position = ids.index(gui_clip.value) if gui_clip.value in ids else -1
        gui_clip.value = ids[(position + step) % len(ids)]

    @gui_keep.on_click
    def _(_=None) -> None:
        write_picks(gui_task.value, [*kept_ids(), gui_clip.value])
        refresh_kept_label()
        advance()

    @gui_drop.on_click
    def _(_=None) -> None:
        write_picks(gui_task.value, [item for item in kept_ids() if item != gui_clip.value])
        refresh_kept_label()
        draw(int(gui_frame.value))

    @gui_next.on_click
    def _(_=None) -> None:
        advance(1)

    @gui_prev.on_click
    def _(_=None) -> None:
        advance(-1)

    def play_loop() -> None:
        # Driven off the wall clock rather than sleeping a fixed interval per frame:
        # sleeping 1/fps ignores however long the draw took, so playback ran slower
        # than the motion really is, which is misleading when the thing being judged
        # is whether a movement looks too fast or too slow.
        next_due = time.monotonic()
        while True:
            loaded: Loaded | None = state["loaded"]
            if not (gui_play.value and loaded is not None):
                next_due = time.monotonic()
                time.sleep(0.05)
                continue
            period = 1.0 / max(1e-3, loaded.fps * gui_speed.value)
            nxt = (int(gui_frame.value) + 1) % len(loaded.joints)
            gui_frame.value = nxt
            draw(nxt)
            next_due += period
            delay = next_due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                # Behind schedule: give up the lost time instead of accumulating debt.
                next_due = time.monotonic()

    threading.Thread(target=play_loop, daemon=True).start()
    switch_task(task_names[0])
    print(f"[picker] open http://{args.host}:{args.port}  — picks land in {args.picks}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
