"""Fading-adaptive pilot weighting for Satori-style voltage recovery."""

from collections.abc import Sequence
from math import sqrt

Point = complex


def _side_lengths(a: Point, b: Point, c: Point) -> tuple[float, float, float]:
    return abs(b - c), abs(a - c), abs(a - b)


def triangle_area(points: Sequence[Point]) -> float:
    """Return triangle area from three complex-plane vertices."""
    if len(points) != 3:
        return 0.0
    a, b, c = points
    return abs(((b.real - a.real) * (c.imag - a.imag)) - ((c.real - a.real) * (b.imag - a.imag))) * 0.5


def circumradius(points: Sequence[Point]) -> float:
    """Return circumradius R_out; very noise-sensitive in practice."""
    if len(points) != 3:
        return 0.0
    a, b, c = points
    sa, sb, sc = _side_lengths(a, b, c)
    area = triangle_area(points)
    if area <= 1e-12:
        return 0.0
    return (sa * sb * sc) / (4.0 * area)


def inradius(points: Sequence[Point]) -> float:
    """Return inradius R_in of the triangle."""
    if len(points) != 3:
        return 0.0
    a, b, c = points
    sa, sb, sc = _side_lengths(a, b, c)
    area = triangle_area(points)
    perimeter = sa + sb + sc
    if perimeter <= 1e-12:
        return 0.0
    return (2.0 * area) / perimeter


def centroid_distance_sum(points: Sequence[Point]) -> float:
    """Return D_c = sum distance from centroid to each vertex."""
    if len(points) != 3:
        return 0.0
    centroid = (points[0] + points[1] + points[2]) / 3.0
    return abs(points[0] - centroid) + abs(points[1] - centroid) + abs(points[2] - centroid)


def fading_weight(points: Sequence[Point], metric: str = "dc") -> float:
    """Return pilot weight w_p from triangle geometry.

    Supported metrics: "dc", "sqrt_area", "rin", "rout".
    """
    m = metric.lower()
    if m == "dc":
        return max(0.0, centroid_distance_sum(points))
    if m == "sqrt_area":
        return max(0.0, sqrt(max(0.0, triangle_area(points))))
    if m == "rin":
        return max(0.0, inradius(points))
    if m == "rout":
        return max(0.0, circumradius(points))
    raise ValueError(f"unknown fading metric: {metric}")


def weighted_voltage_recovery(
    pilot_voltages: Sequence[float],
    pilot_triangles: Sequence[Sequence[Point]],
    metric: str = "dc",
    eps: float = 1e-9,
) -> tuple[float, list[float]]:
    """Apply Eq. (12): weighted average of per-pilot recovered voltages.

    Returns (voltage_hat, normalized_weights).
    """
    if len(pilot_voltages) != len(pilot_triangles) or len(pilot_voltages) == 0:
        return 0.0, []

    raw_weights = [fading_weight(tri, metric=metric) for tri in pilot_triangles]
    wsum = sum(raw_weights)
    if wsum <= eps:
        uniform = [1.0 / float(len(pilot_voltages))] * len(pilot_voltages)
        return sum(v * w for v, w in zip(pilot_voltages, uniform)), uniform

    norm = [w / wsum for w in raw_weights]
    return sum(v * w for v, w in zip(pilot_voltages, norm)), norm
