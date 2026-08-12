"""Color types and palette definitions shared across Blender and non-Blender modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Color:
    r: float
    g: float
    b: float
    a: float = 1.0

    def rgba(self) -> tuple[float, float, float, float]:
        return (self.r, self.g, self.b, self.a)

    def mix(self, other: Color, t: float) -> Color:
        t = min(max(t, 0.0), 1.0)
        return Color(
            r=self.r * (1 - t) + other.r * t,
            g=self.g * (1 - t) + other.g * t,
            b=self.b * (1 - t) + other.b * t,
            a=self.a * (1 - t) + other.a * t,
        )


PALETTES = {
    "paper": [
        Color(0.20, 0.45, 0.82),
        Color(0.86, 0.34, 0.28),
        Color(0.18, 0.62, 0.42),
        Color(0.58, 0.36, 0.74),
        Color(0.92, 0.58, 0.18),
    ],
    "soft": [
        Color(0.35, 0.58, 0.88),
        Color(0.90, 0.48, 0.40),
        Color(0.42, 0.72, 0.55),
        Color(0.68, 0.50, 0.82),
    ],
    "soft_paper": [
        Color(0.42, 0.64, 0.86),
        Color(0.82, 0.50, 0.42),
        Color(0.45, 0.70, 0.56),
        Color(0.66, 0.55, 0.78),
        Color(0.84, 0.66, 0.38),
    ],
    "neutral": [Color(0.35, 0.35, 0.35), Color(0.62, 0.62, 0.62)],
}

PREFIX_NEUTRAL = Color(0.72, 0.75, 0.78, 1.0)
LAVENDER_RAMP_START = Color(231 / 255, 219 / 255, 249 / 255, 1.0)
LAVENDER_RAMP_END = Color(130 / 255, 81 / 255, 219 / 255, 1.0)


def palette_color(index: int, palette: str = "paper") -> Color:
    colors = PALETTES.get(palette, PALETTES["paper"])
    return colors[index % len(colors)]


def prefix_color_from_spec(spec: dict | None = None) -> Color:
    if not spec:
        return PREFIX_NEUTRAL
    rgb = spec.get("color", (0.72, 0.75, 0.78))
    return Color(float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def temporal_color(
    base: Color,
    t: float,
    ramp: str = "light_to_dark",
    *,
    start_lightness: float = 0.55,
    end_lightness: float = 0.0,
) -> Color:
    t = min(max(t, 0.0), 1.0)
    if ramp == "light_to_dark":
        mix = start_lightness * (1.0 - t) + end_lightness * t
        light = Color(1.0, 1.0, 1.0, base.a)
        dark = Color(base.r * 0.55, base.g * 0.55, base.b * 0.55, base.a)
        return light.mix(dark, 1.0 - mix)
    if ramp == "lavender_to_purple":
        return Color(
            LAVENDER_RAMP_START.r,
            LAVENDER_RAMP_START.g,
            LAVENDER_RAMP_START.b,
            base.a,
        ).mix(
            Color(
                LAVENDER_RAMP_END.r,
                LAVENDER_RAMP_END.g,
                LAVENDER_RAMP_END.b,
                base.a,
            ),
            t,
        )
    return base
