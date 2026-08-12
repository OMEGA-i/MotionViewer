from pathlib import Path

from PIL import Image

from motionviewer.video.overlay import draw_instruction_banner, draw_legend_on_frames


def _make_frames(tmp_path: Path, count: int = 2) -> Path:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for idx in range(count):
        Image.new("RGBA", (320, 180), (10, 10, 10, 255)).save(frames_dir / f"frame_{idx:04d}.png")
    return frames_dir


def test_draw_instruction_banner_wraps_long_text_without_error(tmp_path: Path) -> None:
    frames_dir = _make_frames(tmp_path)
    long_instruction = (
        "A person lifts their arms and picks an object from a higher elevation "
        "and then places it at a medium level while facing forward."
    )

    draw_instruction_banner(frames_dir, instruction=long_instruction)

    for frame in sorted(frames_dir.glob("*.png")):
        image = Image.open(frame)
        assert image.size == (320, 180)
        # Banner should have painted some non-background pixels near the top of the frame.
        assert image.convert("RGB").getpixel((160, 30)) != (10, 10, 10)


def test_draw_instruction_banner_is_noop_for_empty_text(tmp_path: Path) -> None:
    frames_dir = _make_frames(tmp_path)
    before = (frames_dir / "frame_0000.png").read_bytes()

    draw_instruction_banner(frames_dir, instruction="   ")

    after = (frames_dir / "frame_0000.png").read_bytes()
    assert before == after


def test_draw_legend_on_frames_paints_swatches(tmp_path: Path) -> None:
    frames_dir = _make_frames(tmp_path)

    draw_legend_on_frames(
        frames_dir, labels=["gt", "hymotion"], palette="soft_paper", instruction="walks forward"
    )

    for frame in sorted(frames_dir.glob("*.png")):
        image = Image.open(frame)
        assert image.size == (320, 180)
