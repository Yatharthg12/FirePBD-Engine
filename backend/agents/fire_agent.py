"""
FirePBD Engine — Fire Simulation Agent
=======================================
Physics-based fire and smoke simulation on the Grid.

Physics Model:
  - Wall-masked probabilistic spread (fire blocked by walls)
  - NFPA heat release rate (HRR) per cell based on fuel type
  - Smoke and CO generation from SFPE combustion yields
  - O₂ depletion tracking
  - FED (Fractional Effective Dose) computation per zone
  - Flashover detection (T > 600°C upper layer, NFPA 72 Annex B)
  - Vectorised NumPy operations — O(N) vs original O(N²) Python loop

Fire Spread Model:
  - Each burning cell spreads to non-wall neighbours
  - Spread probability = BASE_SPREAD × temperature_factor × neighbour_count
  - Opening cells (doors) accelerate spread (OPENING_SPREAD_MULTIPLIER)
  - Burn-out: when fuel_remaining → 0, cell transitions to state=2

References:
  - NFPA 72 Annex B: Heat Release Rate
  - SFPE Handbook Table 2-6.8: Combustion Yields
  - ISO 13571:2012: FED model
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import convolve

from backend.core.constants import (
    AMBIENT_OXYGEN_PERCENT,
    AMBIENT_TEMPERATURE_C,
    BASE_SPREAD_PROB,
    CELL_BURNING,
    CELL_NORMAL,
    CO_DIFFUSION_COEFF,
    CO_G_M3_TO_PPM,
    CO_YIELD_G_G,
    FED_INCAPACITATION_THRESHOLD,
    FUEL_BURN_RATE_PER_STEP,
    HEAT_DECAY_FACTOR,
    HEAT_FED_TABLE,
    OPENING_SPREAD_MULTIPLIER,
    SIMULATION_TIMESTEP_S,
    SMOKE_DIFFUSION_COEFF,
    SMOKE_YIELD_G_G,
    TEMP_BURNOUT_C,
    TEMP_FLASHOVER_C,
    TEMP_IGNITION_C,
    TENABILITY_TEMP_MAX_C,
    TENABILITY_CO_MAX_PPM,
    TENABILITY_VISIBILITY_MIN_M,
)
from backend.core.grid_model import Grid
from backend.utils.logger import get_logger
from backend.utils.math_utils import (
    fed_increment_co,
    fed_increment_heat,
    fed_increment_o2,
    smoke_to_visibility,
)

logger = get_logger(__name__)

# 8-connected spread kernel weights
_SPREAD_KERNEL = np.array([
    [0.50, 0.70, 0.50],
    [0.70, 0.00, 0.70],
    [0.50, 0.70, 0.50],
], dtype=np.float32)


class FireSimulator:
    """
    Vectorised, wall-aware fire spread simulator.

    Parameters
    ----------
    grid : Grid
        The simulation grid (must have wall_mask populated by TopologyAgent)
    dt_s : float
        Timestep in seconds (default from constants)
    base_spread_prob : float
        Override default BASE_SPREAD_PROB if desired
    """

    def __init__(
        self,
        grid: Grid,
        dt_s: float = SIMULATION_TIMESTEP_S,
        base_spread_prob: float = BASE_SPREAD_PROB,
    ) -> None:
        self.grid = grid
        self.dt_s = dt_s
        self.base_spread_prob = base_spread_prob

        # Density factor: d × cell_size² = mass fuel per cell (kg) — for smoke calculation
        # Assume average fuel density 600 MJ/m² and ~20 MJ/kg → 30 kg/m²
        self.fuel_density_kg_m2 = 30.0

        # Track total fuel burned per step for smoke/CO budget
        self._step_count = 0

        logger.debug(f"FireSimulator init: {grid}, dt={dt_s}s, p_spread={base_spread_prob}")

    def step(self) -> dict:
        """
        Advance fire simulation by one timestep.

        Returns a dict of step metrics:
          burning_cells, max_temp, avg_smoke, flashover_detected
        """
        grid = self.grid
        self._step_count += 1

        burning = grid.burning_mask                  # bool array
        open_cells = grid.open_mask                  # ~wall_mask
        normal = grid.normal_mask & open_cells

        # ── Fire Spread ───────────────────────────────────────────────────────
        if burning.any():
            # Convolve burning mask with spread kernel → weighted ignition score
            spread_score = convolve(
                burning.astype(np.float32),
                _SPREAD_KERNEL,
                mode="constant",
                cval=0.0,
            )

            # Openings accelerate spread (simulates chimney / door effect)
            opening_bonus = np.where(
                grid.opening_mask,
                OPENING_SPREAD_MULTIPLIER,
                1.0,
            )
            spread_score = spread_score * opening_bonus

            # Temperature factor: hotter neighbours → faster spread
            temp_factor = np.clip(grid.temperature / TEMP_IGNITION_C, 0.5, 2.0)
            spread_prob = np.clip(
                spread_score * self.base_spread_prob * temp_factor,
                0.0,
                1.0,
            )

            # Stochastic ignition of normal cells
            rand = np.random.random(grid.state.shape).astype(np.float32)
            new_ignitions = normal & (rand < spread_prob)
            grid.state[new_ignitions] = CELL_BURNING
            grid.temperature[new_ignitions] = np.maximum(
                grid.temperature[new_ignitions], TEMP_IGNITION_C
            )

        # ── Heat Release (burning cells) ──────────────────────────────────────
        burning = grid.burning_mask              # refresh after ignition
        if burning.any():
            hrr_per_cell = (
                grid.fuel_load_mj * grid.fuel_remaining * FUEL_BURN_RATE_PER_STEP
            )  # MJ consumed this step

            # Temperature rise: Q = m × c_p × ΔT, simplified volumetric model
            # ΔT ≈ HRR × dt / (ρ_air × c_p × V_cell)
            # Approximated as proportional to HRR fraction
            delta_T = np.where(
                burning,
                np.clip(hrr_per_cell * 50.0, 0.0, 200.0),  # empirical scaling
                0.0,
            )
            grid.temperature += delta_T

            # Fuel consumption
            grid.fuel_remaining -= np.where(
                burning,
                np.minimum(FUEL_BURN_RATE_PER_STEP, grid.fuel_remaining),
                0.0,
            )

        # ── Smoke Generation ──────────────────────────────────────────────────
        if burning.any():
            smoke_gen = np.where(
                burning,
                SMOKE_YIELD_G_G * grid.fuel_load_mj * grid.fuel_remaining * 0.01,
                0.0,
            )
            grid.smoke += smoke_gen

        # ── CO Generation ────────────────────────────────────────────────────
        if burning.any():
            co_gen = np.where(
                burning,
                CO_YIELD_G_G * grid.fuel_load_mj * grid.fuel_remaining * 0.008
                * CO_G_M3_TO_PPM,
                0.0,
            )
            grid.co_ppm += co_gen

        # ── Smoke Diffusion ───────────────────────────────────────────────────
        smoke_kernel = np.array([
            [0.05, 0.10, 0.05],
            [0.10, 0.00, 0.10],
            [0.05, 0.10, 0.05],
        ], dtype=np.float32)
        smoke_spread = convolve(
            grid.smoke * open_cells.astype(np.float32),
            smoke_kernel,
            mode="constant",
            cval=0.0,
        )
        grid.smoke = (
            grid.smoke * (1.0 - SMOKE_DIFFUSION_COEFF)
            + smoke_spread * SMOKE_DIFFUSION_COEFF
        )
        grid.smoke = np.maximum(grid.smoke, 0.0)

        # ── CO Diffusion ──────────────────────────────────────────────────────
        co_spread = convolve(
            grid.co_ppm * open_cells.astype(np.float32),
            smoke_kernel,
            mode="constant",
            cval=0.0,
        )
        grid.co_ppm = (
            grid.co_ppm * (1.0 - CO_DIFFUSION_COEFF)
            + co_spread * CO_DIFFUSION_COEFF
        )
        grid.co_ppm = np.maximum(grid.co_ppm, 0.0)

        # ── O₂ Depletion ──────────────────────────────────────────────────────
        o2_consumed = np.where(
            burning,
            0.002 * self.dt_s / 60.0,  # small depletion per step per burning cell
            0.0,
        )
        grid.oxygen = np.maximum(
            grid.oxygen - o2_consumed,
            0.0,
        )

        # ── Heat Decay ────────────────────────────────────────────────────────
        grid.temperature = np.where(
            open_cells,
            np.maximum(
                grid.temperature * HEAT_DECAY_FACTOR,
                AMBIENT_TEMPERATURE_C,
            ),
            grid.temperature,
        )

        # ── Burn-Out ──────────────────────────────────────────────────────────
        burnout_mask = burning & (grid.fuel_remaining <= 0.01)
        grid.state[burnout_mask] = 2  # burned
        grid.temperature[burnout_mask] = np.maximum(
            grid.temperature[burnout_mask] * 0.90,
            AMBIENT_TEMPERATURE_C,
        )

        # ── Flashover Check ───────────────────────────────────────────────────
        flashover_detected = bool(np.any(grid.temperature > TEMP_FLASHOVER_C))
        if flashover_detected and self._step_count % 10 == 0:
            logger.warning(
                f"FLASHOVER condition detected at step {self._step_count} "
                f"(max_T={grid.temperature.max():.0f}°C)"
            )

        # ── Visibility Update ─────────────────────────────────────────────────
        grid.update_visibility()

        # ── Step Metrics ──────────────────────────────────────────────────────
        burning_now = grid.burning_mask
        return {
            "step": self._step_count,
            "burning_cells": int(burning_now.sum()),
            "burned_cells": int((grid.state == 2).sum()),
            "max_temp_c": float(grid.temperature[open_cells].max()) if open_cells.any() else 0.0,
            "avg_smoke": float(grid.smoke[open_cells].mean()) if open_cells.any() else 0.0,
            "min_visibility_m": float(grid.visibility[open_cells].min()) if open_cells.any() else 10.0,
            "flashover_detected": flashover_detected,
        }

    def ignite_zone(self, zone_id: str) -> int:
        """Ignite all cells in a zone. Returns count of cells ignited."""
        grid = self.grid
        cells = grid.get_zone_cells(zone_id)
        count = 0
        for r, c in cells:
            if grid.state[r, c] == CELL_NORMAL and not grid.wall_mask[r, c]:
                grid.ignite_cell(r, c)
                count += 1
        # Also ignite via flood pattern: ignite centre cells
        if count == 0 and cells:
            r, c = cells[len(cells) // 2]
            grid.ignite_cell(r, c)
            count = 1
        logger.info(f"Zone '{zone_id}' ignited — {count} cells")
        return count


# ─── Fire Analyzer ────────────────────────────────────────────────────────────

class FireAnalyzer:
    """
    Analyses the fire grid relative to zones.

    For each zone computes:
      - Average and peak temperature (°C)
      - Average smoke concentration
      - Average CO (ppm)
      - Minimum visibility (m)
      - FED accumulated (per-step increment)
      - Danger level: LOW / MEDIUM / HIGH / UNTENABLE
      - Tenability flag: False if any limit is breached

    Also detects ASET (Available Safe Egress Time): first step where
    ANY occupied zone exceeds tenability limits.
    """

    DANGER_LOW = "LOW"
    DANGER_MEDIUM = "MEDIUM"
    DANGER_HIGH = "HIGH"
    DANGER_UNTENABLE = "UNTENABLE"

    def __init__(
        self,
        grid: Grid,
        zones: dict,
        dt_s: float = SIMULATION_TIMESTEP_S,
    ) -> None:
        self.grid = grid
        self.zones = zones           # dict: zone_id → Zone
        self.dt_s = dt_s
        self._zone_cell_cache: Dict[str, List[Tuple[int, int]]] = {}

    def _get_zone_cells(self, zone_id: str) -> List[Tuple[int, int]]:
        if zone_id not in self._zone_cell_cache:
            self._zone_cell_cache[zone_id] = self.grid.get_zone_cells(zone_id)
        return self._zone_cell_cache[zone_id]

    def analyze_zones(self) -> Dict[str, dict]:
        """
        Compute hazard status for all zones.
        Returns dict: {zone_id: {temperature, smoke, co_ppm, visibility, danger, tenable, fed_increment}}
        """
        grid = self.grid
        zone_status = {}

        for zone_id, zone in self.zones.items():
            cells = self._get_zone_cells(zone_id)
            if not cells:
                zone_status[zone_id] = self._empty_status(zone_id)
                continue

            rows, cols = zip(*cells)
            temps = grid.temperature[list(rows), list(cols)]
            smokes = grid.smoke[list(rows), list(cols)]
            co_ppms = grid.co_ppm[list(rows), list(cols)]
            vis = grid.visibility[list(rows), list(cols)]
            o2s = grid.oxygen[list(rows), list(cols)]

            avg_temp = float(temps.mean())
            peak_temp = float(temps.max())
            avg_smoke = float(smokes.mean())
            avg_co = float(co_ppms.mean())
            min_vis = float(vis.min())
            avg_o2 = float(o2s.mean())

            danger, tenable = self._classify_danger(avg_temp, avg_smoke, avg_co, min_vis)

            # FED increment this step
            fed_co = fed_increment_co(avg_co, self.dt_s)
            fed_heat = fed_increment_heat(avg_temp, self.dt_s, HEAT_FED_TABLE)
            fed_o2 = fed_increment_o2(avg_o2, self.dt_s)
            fed_inc = fed_co + fed_heat + fed_o2

            zone_status[zone_id] = {
                "zone_id": zone_id,
                "avg_temp_c": round(avg_temp, 1),
                "peak_temp_c": round(peak_temp, 1),
                "avg_smoke": round(avg_smoke, 2),
                "avg_co_ppm": round(avg_co, 1),
                "min_visibility_m": round(min_vis, 1),
                "avg_oxygen_pct": round(avg_o2, 2),
                "danger": danger,
                "tenable": tenable,
                "fed_increment": round(fed_inc, 4),
                "cell_count": len(cells),
            }

        return zone_status

    def _classify_danger(
        self,
        temp: float,
        smoke: float,
        co_ppm: float,
        visibility: float,
    ) -> Tuple[str, bool]:
        """
        Classify danger level per ISO 13571 tenability limits.
        Returns (danger_level, is_tenable).
        """
        # Check tenability limits
        untenable = (
            temp > TENABILITY_TEMP_MAX_C or
            co_ppm > TENABILITY_CO_MAX_PPM or
            visibility < TENABILITY_VISIBILITY_MIN_M
        )
        if untenable:
            return self.DANGER_UNTENABLE, False

        # Scaled thresholds
        if temp > 50 or smoke > 60 or co_ppm > 800:
            return self.DANGER_HIGH, True
        elif temp > 35 or smoke > 25 or co_ppm > 300:
            return self.DANGER_MEDIUM, True
        else:
            return self.DANGER_LOW, True

    def _empty_status(self, zone_id: str) -> dict:
        return {
            "zone_id": zone_id,
            "avg_temp_c": AMBIENT_TEMPERATURE_C,
            "peak_temp_c": AMBIENT_TEMPERATURE_C,
            "avg_smoke": 0.0,
            "avg_co_ppm": 0.0,
            "min_visibility_m": 10.0,
            "avg_oxygen_pct": AMBIENT_OXYGEN_PERCENT,
            "danger": self.DANGER_LOW,
            "tenable": True,
            "fed_increment": 0.0,
            "cell_count": 0,
        }

    def compute_aset(self, zone_statuses: Dict[str, dict]) -> Optional[float]:
        """
        Return the simulation time at which any occupied zone becomes untenable.
        Only considers non-exit zones.
        """
        for zone_id, status in zone_statuses.items():
            zone = self.zones.get(zone_id)
            if zone and zone.is_exit:
                continue
            if not status.get("tenable", True):
                return True  # ASET breached this step
        return None

    def zone_hazard_weights(self, zone_statuses: Dict[str, dict]) -> Dict[str, float]:
        """
        Return dict of {zone_id: hazard_penalty} for pathfinding rerouting.
        HIGH danger → high penalty; UNTENABLE → very high penalty.
        """
        weights = {}
        scale = {
            "LOW": 1.0,
            "MEDIUM": 3.0,
            "HIGH": 8.0,
            "UNTENABLE": 50.0,
        }
        for zone_id, status in zone_statuses.items():
            weights[zone_id] = scale.get(status.get("danger", "LOW"), 1.0)
        return weights