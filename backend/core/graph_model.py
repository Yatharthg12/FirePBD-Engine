"""
FirePBD Engine — Spatial Navigation Graph
==========================================
Converts BuildingModel geometry into a weighted navigation graph.

Nodes = zone centroids
Edges = openings between zones

Edge weight = distance / (width × type_factor)
  → wider opening = lower cost = preferred evacuation path

Provides:
  - A* shortest path (Dijkstra with weight)
  - Hazard-weighted rerouting
  - Accessibility checks
  - Bottleneck detection
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import networkx as nx
from shapely.geometry import Point

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Opening type multipliers — higher = better (wider effective passage)
_OPENING_TYPE_FACTOR: Dict[str, float] = {
    "emergency_exit": 2.0,
    "double_door":    1.8,
    "archway":        1.5,
    "door":           1.0,
    "window":         0.3,
    "inferred":       0.8,
    "default":        1.0,
}


class SpatialGraph:
    """
    Weighted undirected graph for navigation and evacuation routing.

    All edge weights represent routing cost (lower = preferred).
    w = distance / (width × type_factor)
    """

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self.zones: Dict[str, object] = {}   # zone_id → Zone
        self.openings: Dict[str, object] = {}  # opening_id → Opening

    # ─── Construction ─────────────────────────────────────────────────────────

    def add_zone(self, zone) -> None:
        """Register a Zone (or any object with .id, .polygon, .area)."""
        self.zones[zone.id] = zone
        centroid = zone.polygon.centroid
        self.graph.add_node(
            zone.id,
            pos=(centroid.x, centroid.y),
            area=zone.area,
            is_exit=getattr(zone, "is_exit", False),
            label=getattr(zone, "label", "zone"),
        )

    def add_opening(self, opening) -> None:
        """Register an Opening and add the corresponding graph edge."""
        self.openings[opening.id] = opening
        z1, z2 = opening.zone_a, opening.zone_b

        if z1 not in self.graph.nodes or z2 not in self.graph.nodes:
            logger.warning(
                f"Opening '{opening.id}' references unknown zone(s): "
                f"'{z1}', '{z2}'"
            )
            return

        weight = self._compute_weight(z1, z2, opening.width, opening.type)
        self.graph.add_edge(
            z1, z2,
            weight=weight,
            width=opening.width,
            opening_id=opening.id,
            opening_type=opening.type,
            is_exit_door=getattr(opening, "is_exit_door", False),
            inferred=False,
        )

    def _compute_weight(
        self,
        z1: str,
        z2: str,
        width: float,
        opening_type: str = "door",
    ) -> float:
        """Compute routing cost between two connected zones."""
        p1 = Point(self.graph.nodes[z1]["pos"])
        p2 = Point(self.graph.nodes[z2]["pos"])
        distance = p1.distance(p2)
        type_factor = _OPENING_TYPE_FACTOR.get(opening_type, 1.0)
        return distance / max(width * type_factor, 0.1)

    # ─── Pathfinding ──────────────────────────────────────────────────────────

    def shortest_path(
        self,
        start_zone: str,
        end_zone: str,
        blocked_zones: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """
        Find the shortest (minimum-cost) path from start to end.
        Optionally exclude blocked (dangerous) zones.

        Returns list of zone IDs, or None if no path exists.
        """
        if start_zone not in self.graph.nodes or end_zone not in self.graph.nodes:
            return None

        if blocked_zones:
            # Create temporary subgraph excluding blocked zones
            # Keep start and end even if "blocked" (agent must start/end somewhere)
            excluded = set(blocked_zones) - {start_zone, end_zone}
            subgraph = self.graph.subgraph(
                [n for n in self.graph.nodes if n not in excluded]
            )
        else:
            subgraph = self.graph

        try:
            path = nx.shortest_path(
                subgraph,
                source=start_zone,
                target=end_zone,
                weight="weight",
            )
            return list(path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def shortest_path_to_any_exit(
        self,
        start_zone: str,
        exit_zone_ids: List[str],
        blocked_zones: Optional[List[str]] = None,
    ) -> Optional[List[str]]:
        """Find the shortest path from start to any exit zone."""
        best_path: Optional[List[str]] = None
        best_cost = float("inf")

        for exit_id in exit_zone_ids:
            if exit_id == start_zone:
                return [start_zone]
            path = self.shortest_path(start_zone, exit_id, blocked_zones)
            if path:
                cost = self._path_cost(path)
                if cost < best_cost:
                    best_cost = cost
                    best_path = path

        return best_path

    def _path_cost(self, path: List[str]) -> float:
        """Sum of edge weights along a path."""
        total = 0.0
        for i in range(len(path) - 1):
            data = self.graph.edges.get((path[i], path[i + 1]), {})
            total += data.get("weight", 1.0)
        return total

    # ─── Hazard-Weighted Rerouting ─────────────────────────────────────────────

    def shortest_path_hazard_weighted(
        self,
        start_zone: str,
        exit_zone_ids: List[str],
        zone_hazard: Dict[str, float],
    ) -> Optional[List[str]]:
        """
        Pathfinding where edges are penalised by destination zone hazard.
        zone_hazard: dict of {zone_id: penalty_factor}  (1.0 = normal, 10.0 = very dangerous)
        """
        import copy
        subgraph = copy.deepcopy(self.graph)
        for n in subgraph.nodes:
            hazard = zone_hazard.get(n, 1.0)
            for edge in subgraph.edges(n, data=True):
                edge[2]["weight"] = edge[2].get("weight", 1.0) * hazard

        best_path = None
        best_cost = float("inf")
        for exit_id in exit_zone_ids:
            if exit_id == start_zone:
                return [start_zone]
            try:
                path = nx.shortest_path(subgraph, start_zone, exit_id, weight="weight")
                cost = self._path_cost(path)
                if cost < best_cost:
                    best_cost = cost
                    best_path = list(path)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass
        return best_path

    # ─── Graph Properties ─────────────────────────────────────────────────────

    def get_neighbors(self, zone_id: str) -> List[str]:
        return list(self.graph.neighbors(zone_id))

    def is_fully_connected(self) -> bool:
        if len(self.graph.nodes) == 0:
            return False
        return nx.is_connected(self.graph)

    def get_components(self) -> List[set]:
        return list(nx.connected_components(self.graph))

    def get_opening_between(self, z1: str, z2: str) -> Optional[dict]:
        """Return edge data dict for the opening between two zones."""
        return self.graph.edges.get((z1, z2)) or self.graph.edges.get((z2, z1))

    # ─── Bottleneck Analysis ──────────────────────────────────────────────────

    def compute_betweenness_centrality(self) -> Dict[str, float]:
        """
        Node betweenness centrality — high score = bottleneck node.
        Used by risk agent for bottleneck identification.
        """
        return nx.betweenness_centrality(self.graph, weight="weight", normalized=True)

    def identify_bottleneck_edges(self) -> List[Tuple[str, str, float]]:
        """
        Return edges (openings) with high edge betweenness → narrowing hazard.
        Returns list of (z1, z2, centrality_score).
        """
        centrality = nx.edge_betweenness_centrality(
            self.graph, weight="weight", normalized=True
        )
        return sorted(
            [(u, v, c) for (u, v), c in centrality.items()],
            key=lambda x: x[2],
            reverse=True,
        )

    # ─── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        nodes = []
        for n, data in self.graph.nodes(data=True):
            nodes.append({
                "id": n,
                "pos": {"x": data["pos"][0], "y": data["pos"][1]},
                "area": data.get("area", 0),
                "is_exit": data.get("is_exit", False),
            })
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "from": u, "to": v,
                "weight": round(data.get("weight", 0), 3),
                "width": round(data.get("width", 1), 2),
                "opening_type": data.get("opening_type", "door"),
                "inferred": data.get("inferred", False),
            })
        return {"nodes": nodes, "edges": edges}

    def __repr__(self) -> str:
        return (
            f"SpatialGraph(nodes={self.graph.number_of_nodes()}, "
            f"edges={self.graph.number_of_edges()})"
        )