"""
FirePBD Engine — Core Geometry Primitives
==========================================
Language-agnostic, semantics-free spatial primitives.
Everything is geometry + topology. No room labels.

Classes:
  Zone         — a spatial region (polygon-based)
  Opening      — a passable connection between two zones
  WallSegment  — a barrier segment in world coordinates
  BuildingModel — container for a full parsed building
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from backend.core.constants import (
    FUEL_LOAD_DENSITY_MJ_M2,
    OCCUPANCY_LOAD_P_M2,
    HRR_PER_M2_KW,
)


# ─── Zone ────────────────────────────────────────────────────────────────────

class Zone:
    """
    A spatial zone (room / corridor / space) represented by a Shapely polygon.

    Attributes
    ----------
    id : str
        Unique identifier (e.g. "Z0001", "corridor_3")
    polygon : Polygon
        Shapely polygon in world coordinates (metres)
    area : float
        Zone area in m²
    confidence : float
        Extraction confidence [0, 1]
    label : str
        Generic label — not semantic (e.g. "zone", "corridor"). NEVER "kitchen".
    material_type : str
        Fuel type key matching FUEL_LOAD_DENSITY_MJ_M2
    fuel_load_density : float
        MJ/m² — total combustible energy density
    hrr_per_m2 : float
        kW/m² — peak heat release rate
    occupancy_load : float
        persons/m² — from NFPA 101
    floor_level : int
        Floor number (0 = ground)
    is_exit : bool
        True if this zone is or leads directly to an external exit
    pixel_bbox : Tuple[int,int,int,int]
        (min_x, min_y, max_x, max_y) in original image pixels — for overlay
    centroid : Point
        Cached centroid
    """

    def __init__(
        self,
        zone_id: str,
        polygon_points: List[Tuple[float, float]],
        confidence: float = 1.0,
        label: str = "zone",
        material_type: str = "default",
        floor_level: int = 0,
        is_exit: bool = False,
        pixel_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self.id: str = zone_id
        raw_poly = Polygon(polygon_points)
        self.polygon: Polygon = make_valid(raw_poly)

        # If make_valid returned a MultiPolygon, take the largest component
        if isinstance(self.polygon, MultiPolygon):
            self.polygon = max(self.polygon.geoms, key=lambda p: p.area)

        self.area: float = self.polygon.area
        self.confidence: float = float(confidence)
        self.label: str = label
        self.material_type: str = material_type
        self.floor_level: int = floor_level
        self.is_exit: bool = is_exit
        self.pixel_bbox: Optional[Tuple[int, int, int, int]] = pixel_bbox

        # Physics properties (derived from material type by default)
        self.fuel_load_density: float = FUEL_LOAD_DENSITY_MJ_M2.get(
            material_type, FUEL_LOAD_DENSITY_MJ_M2["default"]
        )
        self.hrr_per_m2: float = HRR_PER_M2_KW.get(
            material_type, HRR_PER_M2_KW["default"]
        )
        self.occupancy_load: float = OCCUPANCY_LOAD_P_M2.get(
            material_type, OCCUPANCY_LOAD_P_M2["default"]
        )
        self.centroid: Point = self.polygon.centroid

    @property
    def max_occupants(self) -> int:
        """Maximum occupants based on NFPA 101 load calculation."""
        return max(1, int(self.area * self.occupancy_load))

    @property
    def total_fuel_energy_mj(self) -> float:
        """Total fuel energy content of this zone (MJ)."""
        return self.area * self.fuel_load_density

    def contains_point(self, x: float, y: float) -> bool:
        return self.polygon.contains(Point(x, y))

    def to_dict(self) -> dict:
        coords = list(self.polygon.exterior.coords)
        return {
            "id": self.id,
            "polygon": [{"x": x, "y": y} for x, y in coords],
            "area": round(self.area, 2),
            "confidence": round(self.confidence, 3),
            "label": self.label,
            "material_type": self.material_type,
            "floor_level": self.floor_level,
            "is_exit": self.is_exit,
            "max_occupants": self.max_occupants,
            "fuel_load_density": self.fuel_load_density,
            "centroid": {"x": round(self.centroid.x, 2), "y": round(self.centroid.y, 2)},
        }

    def __repr__(self) -> str:
        return f"Zone(id={self.id!r}, area={self.area:.1f}m², exit={self.is_exit})"


# ─── Opening ─────────────────────────────────────────────────────────────────

class Opening:
    """
    A passable connection (door, window, archway, corridor mouth) between two zones.

    The 'width' drives both evacuation flow rate and fire spread acceleration.
    """

    TYPES = {"door", "double_door", "window", "archway", "emergency_exit", "inferred"}

    def __init__(
        self,
        opening_id: str,
        zone_a: str,
        zone_b: str,
        width: float,
        opening_type: str = "door",
        confidence: float = 1.0,
        sill_height: float = 0.0,
        lintel_height: float = 2.1,
        is_exit_door: bool = False,
        midpoint: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.id: str = opening_id
        self.zone_a: str = zone_a
        self.zone_b: str = zone_b
        self.width: float = float(width)
        self.type: str = opening_type if opening_type in self.TYPES else "door"
        self.confidence: float = float(confidence)
        self.sill_height: float = float(sill_height)
        self.lintel_height: float = float(lintel_height)
        self.is_exit_door: bool = is_exit_door
        self.midpoint: Optional[Tuple[float, float]] = midpoint  # world coordinates

    @property
    def clear_width(self) -> float:
        """Effective clear width for flow calculations (subtract door frame)."""
        return max(0.0, self.width - 0.1)  # 5cm per side frame

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "zone_a": self.zone_a,
            "zone_b": self.zone_b,
            "width": round(self.width, 2),
            "type": self.type,
            "confidence": round(self.confidence, 3),
            "is_exit_door": self.is_exit_door,
            "midpoint": {"x": self.midpoint[0], "y": self.midpoint[1]} if self.midpoint else None,
        }

    def __repr__(self) -> str:
        return f"Opening(id={self.id!r}, {self.zone_a}↔{self.zone_b}, w={self.width:.2f}m)"


# ─── Wall Segment ─────────────────────────────────────────────────────────────

@dataclass
class WallSegment:
    """
    A physical wall segment defined by two endpoints in world coordinates (metres).
    Used to rasterise wall masks onto the simulation grid.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    thickness_m: float = 0.2       # default wall thickness
    material: str = "concrete"

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    def to_dict(self) -> dict:
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "thickness_m": self.thickness_m,
            "material": self.material,
        }


# ─── Building Model ───────────────────────────────────────────────────────────

class BuildingModel:
    """
    Container for a complete parsed building.

    Holds all zones, openings, wall segments, and coordinate metadata.
    This is the interface object passed between all modules.
    """

    def __init__(
        self,
        building_id: Optional[str] = None,
        scale_m_per_px: float = 1.0,
        origin_px: Tuple[float, float] = (0.0, 0.0),
        floor_level: int = 0,
        source_path: str = "",
    ) -> None:
        self.building_id: str = building_id or str(uuid.uuid4())[:8]
        self.scale_m_per_px: float = scale_m_per_px    # metres per pixel
        self.origin_px: Tuple[float, float] = origin_px
        self.floor_level: int = floor_level
        self.source_path: str = source_path

        self.zones: Dict[str, Zone] = {}
        self.openings: Dict[str, Opening] = {}
        self.walls: List[WallSegment] = []

        # World-space bounding box (metres) — computed after zones are added
        self._bbox: Optional[Tuple[float, float, float, float]] = None

    # ── Mutators ──────────────────────────────────────────────────────────────

    def add_zone(self, zone: Zone) -> None:
        self.zones[zone.id] = zone
        self._bbox = None  # invalidate cache

    def add_opening(self, opening: Opening) -> None:
        self.openings[opening.id] = opening

    def add_wall(self, wall: WallSegment) -> None:
        self.walls.append(wall)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def exit_zones(self) -> List[Zone]:
        return [z for z in self.zones.values() if z.is_exit]

    @property
    def exit_zone_ids(self) -> List[str]:
        return [z.id for z in self.exit_zones]

    @property
    def total_area_m2(self) -> float:
        return sum(z.area for z in self.zones.values())

    @property
    def total_occupants(self) -> int:
        occupants = [z.max_occupants for z in self.zones.values() if not z.is_exit]
        if occupants:
            return sum(occupants)
        return sum(z.max_occupants for z in self.zones.values())

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) in world metres."""
        if self._bbox is None and self.zones:
            all_polys = [z.polygon for z in self.zones.values()]
            union = unary_union(all_polys)
            self._bbox = union.bounds  # (minx, miny, maxx, maxy)
        return self._bbox or (0.0, 0.0, 100.0, 100.0)

    @property
    def width_m(self) -> float:
        bb = self.bounding_box
        return bb[2] - bb[0]

    @property
    def height_m(self) -> float:
        bb = self.bounding_box
        return bb[3] - bb[1]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def zone_at_point(self, x: float, y: float) -> Optional[Zone]:
        """Return the zone containing world-coordinate point (x, y)."""
        pt = Point(x, y)
        for zone in self.zones.values():
            if zone.polygon.contains(pt):
                return zone
        return None

    def are_adjacent(self, zone_id_a: str, zone_id_b: str) -> bool:
        """True if the two zones share any Opening."""
        for op in self.openings.values():
            if {op.zone_a, op.zone_b} == {zone_id_a, zone_id_b}:
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "building_id": self.building_id,
            "source_path": self.source_path,
            "scale_m_per_px": self.scale_m_per_px,
            "total_area_m2": round(self.total_area_m2, 1),
            "total_occupants": self.total_occupants,
            "zones": [z.to_dict() for z in self.zones.values()],
            "openings": [o.to_dict() for o in self.openings.values()],
            "walls": [w.to_dict() for w in self.walls],
            "exit_zone_ids": self.exit_zone_ids,
            "bounding_box": {
                "min_x": self.bounding_box[0], "min_y": self.bounding_box[1],
                "max_x": self.bounding_box[2], "max_y": self.bounding_box[3],
            },
        }

    def __repr__(self) -> str:
        return (
            f"BuildingModel(id={self.building_id!r}, "
            f"zones={len(self.zones)}, openings={len(self.openings)}, "
            f"area={self.total_area_m2:.0f}m²)"
        )
