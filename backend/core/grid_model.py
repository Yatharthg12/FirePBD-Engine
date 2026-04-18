"""
FirePBD Engine — Simulation Grid
=================================
The simulation grid is the continuous physical space discretised into cells.
Each cell tracks:
  - fire state      (normal / burning / burned / wall / opening)
  - temperature     (°C)
  - smoke           (smoke density index)
  - CO              (ppm)
  - oxygen          (% remaining)
  - visibility      (metres)
  - fuel_remaining  (fraction of initial fuel load, 0–1)
  - wall_mask       (bool — True = impassable wall)
  - opening_mask    (bool — True = open door/window gap → fire accelerated)

Uses NumPy arrays throughout for vectorised operations.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from backend.core.constants import (
    AMBIENT_OXYGEN_PERCENT,
    AMBIENT_TEMPERATURE_C,
    CELL_BURNING,
    CELL_NORMAL,
    CELL_OPENING,
    CELL_WALL,
    EXTINCTION_COEFF_K,
    TEMP_IGNITION_C,
    VISIBILITY_ILLUMINATED_SIGNS_M,
)


class Grid:
    """
    Discretised simulation space.

    Parameters
    ----------
    width_m : float
        Physical width of the simulated space (metres)
    height_m : float
        Physical height of the simulated space (metres)
    cell_size_m : float
        Size of each square cell (metres). Default 0.5 m.
    """

    def __init__(
        self,
        width_m: float,
        height_m: float,
        cell_size_m: float = 0.5,
    ) -> None:
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.cell_size_m = float(cell_size_m)

        self.cols: int = max(1, int(np.ceil(width_m / cell_size_m)))
        self.rows: int = max(1, int(np.ceil(height_m / cell_size_m)))

        shape: Tuple[int, int] = (self.rows, self.cols)

        # ── Fire State ────────────────────────────────────────────────────────
        # 0=normal, 1=burning, 2=burned, 3=wall, 4=opening
        self.state: np.ndarray = np.zeros(shape, dtype=np.int8)

        # ── Physical Fields ───────────────────────────────────────────────────
        self.temperature: np.ndarray = np.full(shape, AMBIENT_TEMPERATURE_C, dtype=np.float32)
        self.smoke: np.ndarray = np.zeros(shape, dtype=np.float32)
        self.co_ppm: np.ndarray = np.zeros(shape, dtype=np.float32)
        self.oxygen: np.ndarray = np.full(shape, AMBIENT_OXYGEN_PERCENT, dtype=np.float32)
        self.visibility: np.ndarray = np.full(shape, VISIBILITY_ILLUMINATED_SIGNS_M, dtype=np.float32)

        # ── Fuel ──────────────────────────────────────────────────────────────
        # fraction remaining [0, 1] — 1.0 = untouched, 0.0 = fully burned
        self.fuel_remaining: np.ndarray = np.ones(shape, dtype=np.float32)
        # fuel load per cell (MJ) — set by topology agent from zone data
        self.fuel_load_mj: np.ndarray = np.zeros(shape, dtype=np.float32)

        # ── Structure Masks ───────────────────────────────────────────────────
        # True = solid wall cell (fire cannot spread here)
        self.wall_mask: np.ndarray = np.zeros(shape, dtype=bool)
        # True = opening (door/window gap) cell — fire spreads faster
        self.opening_mask: np.ndarray = np.zeros(shape, dtype=bool)

        # ── Zone Cell Lookup ──────────────────────────────────────────────────
        # zone_id per cell — None if unassigned. Filled by TopologyAgent.
        self.zone_map: np.ndarray = np.empty(shape, dtype=object)

        # ── Neighbour offset cache (4-connected used for fire spread in walls) ─
        self._neighbour_offsets_8 = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        self._neighbour_offsets_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # ─── Coordinate Conversion ────────────────────────────────────────────────

    def world_to_cell(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert world coordinates (m) to grid (row, col)."""
        c = int(x_m / self.cell_size_m)
        r = int(y_m / self.cell_size_m)
        return (
            max(0, min(self.rows - 1, r)),
            max(0, min(self.cols - 1, c)),
        )

    def cell_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid (row, col) to world centre coordinates (m)."""
        x = (col + 0.5) * self.cell_size_m
        y = (row + 0.5) * self.cell_size_m
        return x, y

    # ─── Ignition ─────────────────────────────────────────────────────────────

    def ignite(self, x_m: float, y_m: float) -> bool:
        """
        Ignite the cell containing world point (x_m, y_m).
        Returns True if ignition was possible (not a wall, not already burning).
        """
        r, c = self.world_to_cell(x_m, y_m)
        if self.wall_mask[r, c]:
            return False
        if self.state[r, c] in (CELL_BURNING, 2):
            return False
        self.state[r, c] = CELL_BURNING
        self.temperature[r, c] = TEMP_IGNITION_C
        return True

    def ignite_cell(self, row: int, col: int) -> bool:
        if self.wall_mask[row, col]:
            return False
        self.state[row, col] = CELL_BURNING
        self.temperature[row, col] = TEMP_IGNITION_C
        return True

    # ─── Wall / Opening Registration ─────────────────────────────────────────

    def set_wall(self, row: int, col: int) -> None:
        """Mark a cell as a solid wall."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.wall_mask[row, col] = True
            self.state[row, col] = CELL_WALL
            self.fuel_remaining[row, col] = 0.0

    def set_wall_line(
        self,
        x1_m: float, y1_m: float,
        x2_m: float, y2_m: float,
        thickness_cells: int = 1,
    ) -> None:
        """
        Rasterise a wall line segment onto the grid using Bresenham's algorithm.
        """
        r1, c1 = self.world_to_cell(x1_m, y1_m)
        r2, c2 = self.world_to_cell(x2_m, y2_m)
        for r, c in self._bresenham(r1, c1, r2, c2):
            for dr in range(-thickness_cells // 2, thickness_cells // 2 + 1):
                for dc in range(-thickness_cells // 2, thickness_cells // 2 + 1):
                    self.set_wall(r + dr, c + dc)

    def set_opening(self, row: int, col: int) -> None:
        """Mark a cell as an opening (door/window gap) — overrides wall."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.wall_mask[row, col] = False
            self.opening_mask[row, col] = True
            self.state[row, col] = CELL_OPENING

    def set_wall_polygon(self, polygon_coords: list, thickness_cells: int = 1) -> None:
        """Rasterise a wall polygon boundary onto the grid."""
        n = len(polygon_coords)
        for i in range(n):
            x1, y1 = polygon_coords[i]
            x2, y2 = polygon_coords[(i + 1) % n]
            self.set_wall_line(x1, y1, x2, y2, thickness_cells)

    # ─── Visibility Calculation ───────────────────────────────────────────────

    def update_visibility(self) -> None:
        """
        Recompute visibility from smoke density.
        Formula: V = C / (K × Cs)  where Cs = smoke index, K = extinction coeff.
        Simplified: V = VISIBILITY_MAX / (1 + K × smoke)
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            vis = VISIBILITY_ILLUMINATED_SIGNS_M / (
                1.0 + EXTINCTION_COEFF_K * np.maximum(self.smoke, 0.0)
            )
        self.visibility = np.clip(vis, 0.0, VISIBILITY_ILLUMINATED_SIGNS_M)

    # ─── Neighbour Utilities ──────────────────────────────────────────────────

    def get_open_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Return 8-connected neighbours that are NOT solid walls."""
        result = []
        for dr, dc in self._neighbour_offsets_8:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if not self.wall_mask[nr, nc]:
                    result.append((nr, nc))
        return result

    def get_all_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Return all valid 8-connected neighbours (walls included)."""
        result = []
        for dr, dc in self._neighbour_offsets_8:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                result.append((nr, nc))
        return result

    # ─── Boolean Masks ────────────────────────────────────────────────────────

    @property
    def burning_mask(self) -> np.ndarray:
        return self.state == CELL_BURNING

    @property
    def normal_mask(self) -> np.ndarray:
        return self.state == CELL_NORMAL

    @property
    def burned_mask(self) -> np.ndarray:
        return self.state == 2

    @property
    def open_mask(self) -> np.ndarray:
        """Cells where fire CAN exist (not solid wall)."""
        return ~self.wall_mask

    # ─── Cell Zone Query ──────────────────────────────────────────────────────

    def get_zone_cells(self, zone_id: str) -> List[Tuple[int, int]]:
        """Return all (row, col) cells belonging to a given zone_id."""
        rows, cols = np.where(self.zone_map == zone_id)
        return list(zip(rows.tolist(), cols.tolist()))

    def assign_zone_map(self, zones: dict) -> None:
        """
        Populate zone_map array from Zone.polygon objects.
        zones: dict of {zone_id: Zone}
        """
        from shapely.geometry import Point as SPoint
        for r in range(self.rows):
            for c in range(self.cols):
                if self.wall_mask[r, c]:
                    continue
                x, y = self.cell_to_world(r, c)
                pt = SPoint(x, y)
                for zid, zone in zones.items():
                    if zone.polygon.contains(pt):
                        self.zone_map[r, c] = zid
                        break

    # ─── Snapshot ────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return a lightweight serialisable snapshot of all grid fields."""
        return {
            "state": self.state.tolist(),
            "temperature": self.temperature.tolist(),
            "smoke": self.smoke.tolist(),
            "co_ppm": self.co_ppm.tolist(),
            "oxygen": self.oxygen.tolist(),
            "visibility": self.visibility.tolist(),
            "fuel_remaining": self.fuel_remaining.tolist(),
        }

    def snapshot_compact(self) -> dict:
        """Compact snapshot (rounded, for WebSocket streaming)."""
        return {
            "state": self.state.tolist(),
            "temperature": np.round(self.temperature, 1).tolist(),
            "smoke": np.round(self.smoke, 2).tolist(),
            "visibility": np.round(self.visibility, 1).tolist(),
        }

    # ─── Internal: Bresenham Line ─────────────────────────────────────────────

    @staticmethod
    def _bresenham(r0, c0, r1, c1) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm — yields (row, col) pairs."""
        points = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        r, c = r0, c0
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        if dc > dr:
            err = dc / 2
            while c != c1:
                points.append((r, c))
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
                c += sc
        else:
            err = dr / 2
            while r != r1:
                points.append((r, c))
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
                r += sr
        points.append((r, c))
        return points

    def __repr__(self) -> str:
        return (
            f"Grid({self.width_m:.0f}×{self.height_m:.0f}m, "
            f"cells={self.rows}×{self.cols}, "
            f"cell_size={self.cell_size_m}m)"
        )