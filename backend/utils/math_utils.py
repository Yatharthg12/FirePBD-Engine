"""
FirePBD Engine — Mathematical Utilities
========================================
Numerical helpers used across fire, evacuation, and risk modules.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


# ─── Interpolation ────────────────────────────────────────────────────────────

def interpolate_table(x: float, table: List[Tuple[float, float]]) -> float:
    """
    Linear interpolation over a sorted (x, y) table.
    Clamps to edge values outside the table range.

    Parameters
    ----------
    x : float
        Query value
    table : list of (x_val, y_val) sorted by x_val ascending

    Returns
    -------
    float : interpolated y at query x
    """
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        x0, y0 = table[i]
        x1, y1 = table[i + 1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return table[-1][1]


# ─── Statistics ───────────────────────────────────────────────────────────────

def percentile_safe(data: List[float], pct: float) -> float:
    """Return the given percentile of data; return 0 if data is empty."""
    if not data:
        return 0.0
    return float(np.percentile(data, pct))


def mean_safe(data: List[float]) -> float:
    if not data:
        return 0.0
    return float(np.mean(data))


def std_safe(data: List[float]) -> float:
    if not data:
        return 0.0
    return float(np.std(data))


def confidence_interval(
    data: List[float],
    confidence: float = 0.90,
) -> Tuple[float, float]:
    """
    Return (lower, upper) confidence interval for a data sample.
    Uses normal approximation (for large N) suitable for MC outputs.
    """
    if len(data) < 2:
        v = data[0] if data else 0.0
        return v, v
    arr = np.array(data, dtype=float)
    n = len(arr)
    m = np.mean(arr)
    se = np.std(arr, ddof=1) / math.sqrt(n)
    # z for 90% CI ≈ 1.645
    from scipy.stats import t as t_dist
    z = t_dist.ppf((1 + confidence) / 2, df=n - 1)
    return float(m - z * se), float(m + z * se)


# ─── FED Utilities ────────────────────────────────────────────────────────────

def heat_incapacitation_time_s(temperature_c: float, table: list) -> float:
    """
    Return estimated time-to-incapacitation from heat (seconds) at given temperature.
    Uses SFPE / ISO 13571 heat FED table.
    """
    return interpolate_table(temperature_c, table)


def fed_increment_co(co_ppm: float, dt_s: float, normaliser: float = 35000.0) -> float:
    """
    FED increment per timestep from CO inhalation.
    Based on Purser (1988) simplified model.

    FED_CO = (CO_ppm / 35000) × (dt_min)
    """
    dt_min = dt_s / 60.0
    return (co_ppm / normaliser) * dt_min


def fed_increment_heat(temperature_c: float, dt_s: float, heat_table: list) -> float:
    """FED increment per timestep from heat exposure."""
    t_incap = heat_incapacitation_time_s(temperature_c, heat_table)
    if t_incap <= 0:
        return 1.0  # immediate
    return dt_s / t_incap


def fed_increment_o2(oxygen_pct: float, dt_s: float, threshold: float = 20.9) -> float:
    """
    FED increment per timestep from O₂ depletion.
    Simplified: linear contribution below ambient level.
    Incapacitation at 12%; below 8% → immediate.
    """
    if oxygen_pct >= threshold:
        return 0.0
    if oxygen_pct <= 8.0:
        return dt_s / 5.0  # very rapid incapacitation
    # linear interpolation: 0 FED/s at 20.9%, 0.003 FED/s at 12%
    rate = 0.003 * (threshold - oxygen_pct) / (threshold - 12.0)
    return rate * dt_s


# ─── Visibility ───────────────────────────────────────────────────────────────

def smoke_to_visibility(smoke_index: float, k: float = 0.08) -> float:
    """
    Convert smoke density index to visibility (metres).
    V = V_max / (1 + k × smoke)
    """
    from backend.core.constants import VISIBILITY_ILLUMINATED_SIGNS_M
    if smoke_index <= 0:
        return VISIBILITY_ILLUMINATED_SIGNS_M
    return VISIBILITY_ILLUMINATED_SIGNS_M / (1.0 + k * smoke_index)


def visibility_to_speed_fraction(visibility_m: float, curve: list) -> float:
    """Map visibility (m) to fraction of free-flow walking speed."""
    return interpolate_table(visibility_m, curve)


# ─── Geometry ─────────────────────────────────────────────────────────────────

def polygon_area(points: List[Tuple[float, float]]) -> float:
    """Shoelace formula for polygon area."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ─── Risk ─────────────────────────────────────────────────────────────────────

def normalise_risk_score(raw: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp and round a risk score to [0, 100]."""
    return round(clamp(raw, lo, hi), 1)
