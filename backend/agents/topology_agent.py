"""
FirePBD Engine — Topology Agent
=================================
Converts a BuildingModel into:
  1. A SpatialGraph (navigation backbone)
  2. A Grid (fire simulation space) with wall mask registered

This is the bridge between the Blueprint Extraction Agent and all
simulation agents. It produces the fully initialised simulation environment.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from shapely.geometry import Point

from backend.core.geometry import BuildingModel, Opening, Zone
from backend.core.graph_model import SpatialGraph
from backend.core.grid_model import Grid
from backend.config import DEFAULT_GRID_CELL_SIZE_M
from backend.utils.logger import get_logger
from backend.utils.repair import AutoRepair
from backend.utils.validation import BuildingModelValidator

logger = get_logger(__name__)


class TopologyAgent:
    """
    Builds the spatial graph and grid from a BuildingModel.

    Usage:
        agent = TopologyAgent()
        graph, grid = agent.build(model)
    """

    def __init__(self, cell_size_m: float = DEFAULT_GRID_CELL_SIZE_M) -> None:
        self.cell_size_m = cell_size_m

    def build(
        self,
        model: BuildingModel,
        auto_repair: bool = True,
    ) -> Tuple[SpatialGraph, Grid]:
        """
        Full topology construction pipeline.

        1. Build SpatialGraph from zones + openings
        2. Auto-repair disconnected graph (if enabled)
        3. Construct Grid matching building bounding box
        4. Rasterise wall segments onto grid
        5. Mark opening cells on grid
        6. Assign zone_map to grid cells
        7. Set fuel loads per cell

        Parameters
        ----------
        model : BuildingModel
        auto_repair : bool
            Apply AutoRepair to fix disconnected graph / missing connections.

        Returns
        -------
        (SpatialGraph, Grid)
        """
        logger.info(f"Building topology for: {model}")

        # ── 1. Build graph ────────────────────────────────────────────────────
        graph = self._build_graph(model)

        # ── 2. Auto-repair ────────────────────────────────────────────────────
        if auto_repair:
            repair_log = AutoRepair.run_all(model, graph)
            total_repairs = sum(len(v) for v in repair_log.values())
            if total_repairs > 0:
                logger.info(f"AutoRepair: {total_repairs} actions taken")

        # ── 3. Validate ───────────────────────────────────────────────────────
        validation = BuildingModelValidator.validate(model, graph)
        if not validation.is_valid():
            logger.warning(
                f"Validation issues: {len(validation.errors)} errors, "
                f"{len(validation.warnings)} warnings"
            )

        # ── 4. Build grid ─────────────────────────────────────────────────────
        grid = self._build_grid(model)

        # ── 5. Register walls ─────────────────────────────────────────────────
        self._rasterise_walls(model, grid)

        # ── 6. Register openings ──────────────────────────────────────────────
        self._rasterise_openings(model, grid)

        # ── 7. Assign zone map ────────────────────────────────────────────────
        self._assign_zone_map(model, grid)

        # ── 8. Set fuel loads ─────────────────────────────────────────────────
        self._assign_fuel_loads(model, grid)

        logger.info(
            f"Topology built: graph={graph}, grid={grid}"
        )
        return graph, grid

    # ─── Graph Construction ───────────────────────────────────────────────────

    def _build_graph(self, model: BuildingModel) -> SpatialGraph:
        graph = SpatialGraph()

        for zone in model.zones.values():
            graph.add_zone(zone)

        for opening in model.openings.values():
            graph.add_opening(opening)

        logger.debug(
            f"Graph built: {graph.graph.number_of_nodes()} nodes, "
            f"{graph.graph.number_of_edges()} edges"
        )
        return graph

    # ─── Grid Construction ────────────────────────────────────────────────────

    def _build_grid(self, model: BuildingModel) -> Grid:
        bb = model.bounding_box  # (minx, miny, maxx, maxy)
        # Add 2-cell margin on each side
        margin = self.cell_size_m * 2
        width_m = (bb[2] - bb[0]) + 2 * margin
        height_m = (bb[3] - bb[1]) + 2 * margin

        grid = Grid(
            width_m=width_m,
            height_m=height_m,
            cell_size_m=self.cell_size_m,
        )
        logger.debug(f"Grid created: {grid}")
        return grid

    def _world_offset(self, model: BuildingModel) -> Tuple[float, float]:
        """Return (offset_x, offset_y) to shift world coords to grid origin."""
        bb = model.bounding_box
        margin = self.cell_size_m * 2
        return (-(bb[0] - margin), -(bb[1] - margin))

    # ─── Wall Rasterisation ───────────────────────────────────────────────────

    def _rasterise_walls(self, model: BuildingModel, grid: Grid) -> None:
        """Burn wall segments and zone boundaries into the grid wall_mask."""
        ox, oy = self._world_offset(model)
        wall_count = 0

        # 1. Explicit wall segments from blueprint parser
        for wall in model.walls:
            grid.set_wall_line(
                wall.x1 + ox, wall.y1 + oy,
                wall.x2 + ox, wall.y2 + oy,
                thickness_cells=1,
            )
            wall_count += 1

        # 2. Zone polygon boundaries → infer walls at edges between zones
        if wall_count == 0:
            logger.debug("No explicit walls — inferring from zone boundaries")
            for zone in model.zones.values():
                coords = list(zone.polygon.exterior.coords)
                for i in range(len(coords) - 1):
                    x1, y1 = coords[i]
                    x2, y2 = coords[i + 1]
                    grid.set_wall_line(
                        x1 + ox, y1 + oy,
                        x2 + ox, y2 + oy,
                        thickness_cells=1,
                    )

        # 3. Grid border = wall
        for r in range(grid.rows):
            grid.set_wall(r, 0)
            grid.set_wall(r, grid.cols - 1)
        for c in range(grid.cols):
            grid.set_wall(0, c)
            grid.set_wall(grid.rows - 1, c)

        logger.debug(
            f"Wall rasterisation: {int(grid.wall_mask.sum())} wall cells "
            f"({100 * grid.wall_mask.sum() / grid.wall_mask.size:.1f}%)"
        )

    # ─── Opening Rasterisation ────────────────────────────────────────────────

    def _rasterise_openings(self, model: BuildingModel, grid: Grid) -> None:
        """
        Clear wall cells at opening midpoints — creates passable gap in wall mask.
        """
        ox, oy = self._world_offset(model)

        for opening in model.openings.values():
            if opening.midpoint:
                mx, my = opening.midpoint
                r, c = grid.world_to_cell(mx + ox, my + oy)
                # Clear a gap in the wall proportional to door width
                half_gap = max(1, int(opening.width / (2 * self.cell_size_m)))
                self._clear_opening_gap(grid, r, c, half_gap, opening.zone_a, opening.zone_b)
            else:
                # No midpoint — clear cells between zone centroids (interpolated)
                if opening.zone_a in model.zones and opening.zone_b in model.zones:
                    za = model.zones[opening.zone_a]
                    zb = model.zones[opening.zone_b]
                    mx = (za.centroid.x + zb.centroid.x) / 2
                    my = (za.centroid.y + zb.centroid.y) / 2
                    r, c = grid.world_to_cell(mx + ox, my + oy)
                    half_gap = max(1, int(opening.width / (2 * self.cell_size_m)))
                    self._clear_opening_gap(grid, r, c, half_gap, opening.zone_a, opening.zone_b)

    def _clear_opening_gap(
        self,
        grid: Grid,
        row: int,
        col: int,
        half_gap: int,
        za_id: str,
        zb_id: str,
    ) -> None:
        """Clear a gap in the wall and mark as opening cells."""
        for dr in range(-half_gap, half_gap + 1):
            for dc in range(-half_gap, half_gap + 1):
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid.rows and 0 <= nc < grid.cols:
                    grid.set_opening(nr, nc)

    # ─── Zone Map Assignment ──────────────────────────────────────────────────

    def _assign_zone_map(self, model: BuildingModel, grid: Grid) -> None:
        """
        Assign each grid cell to a zone using point-in-polygon tests.
        Vectorised using cell centre coordinates.
        """
        ox, oy = self._world_offset(model)

        for r in range(grid.rows):
            for c in range(grid.cols):
                if grid.wall_mask[r, c]:
                    continue
                wx, wy = grid.cell_to_world(r, c)
                # Shift back to model coordinates
                mx, my = wx - ox, wy - oy
                pt = Point(mx, my)

                for zid, zone in model.zones.items():
                    if zone.polygon.contains(pt):
                        grid.zone_map[r, c] = zid
                        break

        assigned = np.sum(grid.zone_map != None)
        logger.debug(
            f"Zone map: {assigned}/{grid.rows * grid.cols - int(grid.wall_mask.sum())} "
            f"non-wall cells assigned"
        )

    # ─── Fuel Load Assignment ─────────────────────────────────────────────────

    def _assign_fuel_loads(self, model: BuildingModel, grid: Grid) -> None:
        """Set fuel_load_mj per cell from zone fuel load density."""
        cell_area_m2 = self.cell_size_m ** 2

        for r in range(grid.rows):
            for c in range(grid.cols):
                zid = grid.zone_map[r, c]
                if zid and zid in model.zones:
                    zone = model.zones[zid]
                    grid.fuel_load_mj[r, c] = zone.fuel_load_density * cell_area_m2
                elif not grid.wall_mask[r, c]:
                    grid.fuel_load_mj[r, c] = 420.0 * cell_area_m2  # default

        logger.debug("Fuel loads assigned to grid cells")

    # ─── Exit Zone Update ─────────────────────────────────────────────────────

    def update_graph_exit_flags(
        self, graph: SpatialGraph, model: BuildingModel
    ) -> None:
        """Sync is_exit flags from model into graph node attributes."""
        for zid in model.zones:
            if zid in graph.graph.nodes:
                graph.graph.nodes[zid]["is_exit"] = model.zones[zid].is_exit
