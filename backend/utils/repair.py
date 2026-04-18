"""
FirePBD Engine — Auto-Repair Module
=====================================
Diagnoses and repairs broken or incomplete building models and spatial graphs.
Handles imperfect blueprint extractions gracefully.
"""
from __future__ import annotations

import uuid
from typing import List, Optional, Tuple

from shapely.geometry import Point
from shapely.ops import unary_union

from backend.core.geometry import BuildingModel, Opening, Zone
from backend.core.graph_model import SpatialGraph
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class AutoRepair:
    """
    Static repair methods for zone polygons and spatial graphs.
    All methods return a list of actions taken for audit logging.
    """

    # ─── Polygon Repair ───────────────────────────────────────────────────────

    @staticmethod
    def repair_zone_polygons(model: BuildingModel) -> List[str]:
        """
        Apply Shapely buffer(0) and make_valid to all zones.
        Removes self-intersections and degenerate geometries.
        """
        from shapely.validation import make_valid
        actions = []
        to_remove = []

        for zone_id, zone in model.zones.items():
            if not zone.polygon.is_valid:
                fixed = make_valid(zone.polygon)
                if fixed.is_empty or fixed.area < 0.5:
                    to_remove.append(zone_id)
                    actions.append(f"REMOVED degenerate zone '{zone_id}'")
                else:
                    zone.polygon = fixed
                    zone.area = fixed.area
                    zone.centroid = fixed.centroid
                    actions.append(f"REPAIRED polygon for zone '{zone_id}'")

        for zid in to_remove:
            del model.zones[zid]

        return actions

    # ─── Graph Repair ─────────────────────────────────────────────────────────

    @staticmethod
    def connect_nearest_zones(
        graph: SpatialGraph,
        max_distance_m: float = 15.0,
        max_boundary_gap_m: float = 1.5,
    ) -> List[str]:
        """
        Connect pairs of zones that are close but have no opening between them.
        Infers virtual openings with width=1.0m and type="inferred".
        Used when blueprint extraction misses some doors.
        """
        actions = []
        zone_ids = list(graph.zones.keys())

        for i in range(len(zone_ids)):
            for j in range(i + 1, len(zone_ids)):
                z1, z2 = zone_ids[i], zone_ids[j]
                if graph.graph.has_edge(z1, z2):
                    continue

                p1 = Point(graph.graph.nodes[z1]["pos"])
                p2 = Point(graph.graph.nodes[z2]["pos"])
                dist = p1.distance(p2)

                zone_a = graph.zones.get(z1)
                zone_b = graph.zones.get(z2)
                boundary_gap = None
                if zone_a is not None and zone_b is not None:
                    try:
                        boundary_gap = zone_a.polygon.distance(zone_b.polygon)
                    except Exception:
                        boundary_gap = None

                if boundary_gap is not None and boundary_gap > max_boundary_gap_m:
                    continue

                if dist < max_distance_m:
                    opening_id = f"INF_{z1}_{z2}"
                    graph.graph.add_edge(
                        z1, z2,
                        weight=dist / 1.0,
                        width=1.0,
                        opening_id=opening_id,
                        inferred=True,
                    )
                    action = (
                        f"INFERRED opening between '{z1}' and '{z2}' "
                        f"(dist={dist:.1f}m, gap={boundary_gap if boundary_gap is not None else 'n/a'})"
                    )
                    actions.append(action)
                    logger.debug(action)

        return actions

    @staticmethod
    def fix_disconnected_graph(
        graph: SpatialGraph,
        model: Optional[BuildingModel] = None,
    ) -> List[str]:
        """
        Fix a disconnected graph by connecting components using the nearest
        zone pair between components.
        """
        actions = []
        components = list(graph.get_components())

        if len(components) <= 1:
            return actions

        logger.warning(
            f"Disconnected graph: {len(components)} components — attempting repair"
        )

        # Connect each component to the next via nearest zone pair
        for i in range(len(components) - 1):
            comp_a = list(components[i])
            comp_b = list(components[i + 1])

            best_dist = float("inf")
            best_pair: Optional[Tuple[str, str]] = None

            for za in comp_a:
                for zb in comp_b:
                    if za not in graph.graph.nodes or zb not in graph.graph.nodes:
                        continue
                    p1 = Point(graph.graph.nodes[za]["pos"])
                    p2 = Point(graph.graph.nodes[zb]["pos"])
                    d = p1.distance(p2)
                    if d < best_dist:
                        best_dist = d
                        best_pair = (za, zb)

            if best_pair:
                z1, z2 = best_pair
                graph.graph.add_edge(
                    z1, z2,
                    weight=best_dist,
                    width=1.0,
                    inferred=True,
                )
                action = (
                    f"CONNECTED components via '{z1}'↔'{z2}' "
                    f"(dist={best_dist:.1f}m)"
                )
                actions.append(action)
                logger.info(action)

        return actions

    @staticmethod
    def ensure_exit_reachable(
        graph: SpatialGraph,
        model: BuildingModel,
    ) -> List[str]:
        """
        Verify all non-exit zones have a path to at least one exit.
        If not, infer a connection to the nearest exit zone.
        """
        actions = []
        exit_ids = model.exit_zone_ids

        if not exit_ids:
            actions.append("NO EXITS defined — cannot ensure reachability")
            return actions

        for zone_id in model.zones:
            if zone_id in exit_ids:
                continue
            reachable = False
            for ex_id in exit_ids:
                path = graph.shortest_path(zone_id, ex_id)
                if path:
                    reachable = True
                    break
            if not reachable:
                # Connect to nearest exit
                pos0 = graph.graph.nodes.get(zone_id, {}).get("pos")
                if pos0 is None:
                    continue
                p0 = Point(pos0)
                best_dist = float("inf")
                best_exit = exit_ids[0]
                for ex_id in exit_ids:
                    pos_ex = graph.graph.nodes.get(ex_id, {}).get("pos")
                    if pos_ex is None:
                        continue
                    d = p0.distance(Point(pos_ex))
                    if d < best_dist:
                        best_dist = d
                        best_exit = ex_id
                graph.graph.add_edge(
                    zone_id, best_exit,
                    weight=best_dist,
                    width=0.9,
                    inferred=True,
                )
                action = (
                    f"FORCED exit path: '{zone_id}'→'{best_exit}' "
                    f"(dist={best_dist:.1f}m)"
                )
                actions.append(action)
                logger.warning(action)

        return actions

    @staticmethod
    def run_all(
        model: BuildingModel,
        graph: SpatialGraph,
        max_connect_dist_m: float = 15.0,
    ) -> dict:
        """
        Run full repair pipeline and return audit dict.
        """
        result = {
            "polygon_repairs": AutoRepair.repair_zone_polygons(model),
            "inferred_connections": AutoRepair.connect_nearest_zones(
                graph, max_connect_dist_m
            ),
            "component_fixes": AutoRepair.fix_disconnected_graph(graph, model),
            "exit_reachability": AutoRepair.ensure_exit_reachable(graph, model),
        }
        total = sum(len(v) for v in result.values())
        logger.info(f"AutoRepair complete — {total} actions taken")
        return result
