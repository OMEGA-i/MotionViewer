from motionviewer.core.palette import Color, temporal_color
from motionviewer.video.spec import RenderJob


def test_lavender_to_purple_ramp_endpoints() -> None:
    base = Color(0.2, 0.4, 0.6, 0.5)

    start = temporal_color(base, 0.0, "lavender_to_purple")
    end = temporal_color(base, 1.0, "lavender_to_purple")

    assert start.rgba() == (231 / 255, 219 / 255, 249 / 255, 0.5)
    assert end.rgba() == (130 / 255, 81 / 255, 219 / 255, 0.5)


def test_render_job_template_uses_rectangle_ground_and_purple_ramp() -> None:
    job = RenderJob.template([{"path": "sample.smplx.npz"}])

    assert job.ground.mode == "trajectory_rectangle"
    assert job.style.temporal_ramp == "lavender_to_purple"
