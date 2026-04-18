"""
FirePBD Engine — Blueprint Extraction Agent
=============================================
Converts building blueprint files (PNG or SVG) into a BuildingModel.

PRIMARY PATH: SVG Parser (CubiCasa model.svg)
  → Parses wall polygons, room boundaries, doors directly from SVG annotation.
  → No ML model required. Exact geometry from dataset ground truth.

FALLBACK PATH: OpenCV Image Pipeline
  → Preprocessing → Wall detection → Contour extraction → Door gap detection.
  → Works on any raster blueprint image.

Usage:
    agent = BlueprintAgent()
    model = agent.process(path="datasets/CubiCasa5K/.../model.svg")
    # or
    model = agent.process(path="my_blueprint.png")
"""
from __future__ import annotations

import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid

from backend.config import (
    CONTOUR_APPROX_EPSILON,
    DEFAULT_GRID_CELL_SIZE_M,
    DOOR_GAP_MAX_CELLS,
    DOOR_GAP_MIN_CELLS,
    MIN_OPENING_WIDTH_M,
    MIN_ZONE_AREA_M2,
)
from backend.core.geometry import BuildingModel, Opening, WallSegment, Zone
from backend.utils.image_processing import (
    detect_door_gaps,
    extract_room_contours,
    preprocess_for_wall_detection,
    skeletonise,
    contour_to_polygon,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# SVG namespace for CubiCasa files
_SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


class BlueprintAgent:
    """
    Top-level blueprint processing agent.

    Automatically selects SVG or image processing based on file extension.
    Produces a validated BuildingModel ready for simulation.
    """

    def __init__(
        self,
        cell_size_m: float = DEFAULT_GRID_CELL_SIZE_M,
    ) -> None:
        self.cell_size_m = cell_size_m
        self._svg_parser = SVGParser()
        self._image_pipeline = ImagePipeline(cell_size_m)

    def process(self, path: str) -> BuildingModel:
        """
        Main entry point. Returns a BuildingModel from any blueprint path.

        Parameters
        ----------
        path : str
            Path to .svg (preferred) or .png/.jpg/.bmp file.

        Returns
        -------
        BuildingModel
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Blueprint not found: {path}")

        ext = p.suffix.lower()
        logger.info(f"Processing blueprint: {path} (format: {ext})")

        if ext == ".svg":
            model = self._svg_parser.parse(str(p))
        elif ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            model = self._image_pipeline.process(str(p))
        else:
            raise ValueError(f"Unsupported blueprint format: {ext}")

        model.source_path = str(p)
        self._post_process(model)
        logger.info(f"Blueprint processed: {model}")
        return model

    def _post_process(self, model: BuildingModel) -> None:
        """Final cleanup: re-index zone IDs, validate areas, identify exits."""
        # Re-index zones sequentially if IDs are raw SVG element IDs
        zones_valid = {
            zid: z for zid, z in model.zones.items()
            if z.area >= MIN_ZONE_AREA_M2 and z.polygon.is_valid
        }
        model.zones = zones_valid

        # Identify perimeter zones as candidate exits (zones touching building edge)
        bbox = model.bounding_box
        margin = max(0.5, min(model.width_m, model.height_m) * 0.02)
        outdoor_labels = {"outdoor", "balcony", "terrace", "porch", "entrance"}

        for zone in model.zones.values():
            b = zone.polygon.bounds  # (minx, miny, maxx, maxy)
            touches_perimeter = (
                b[0] <= bbox[0] + margin or
                b[1] <= bbox[1] + margin or
                b[2] >= bbox[2] - margin or
                b[3] >= bbox[3] - margin
            )
            if touches_perimeter and zone.label in outdoor_labels and not zone.is_exit:
                # Mark as exit candidate — perimeter-touching zones are usually exits
                # or corridors leading to exits. In practice, the user can refine this.
                zone.is_exit = True
                logger.debug(f"Auto-marked exit: zone '{zone.id}'")

        # Degenerate single-zone fallback: create a tiny exterior exit zone so
        # the simulation still has occupants, an exit, and a traversable path.
        if len(model.zones) == 1 and not model.openings and not model.exit_zones:
            only_zone = next(iter(model.zones.values()))
            min_x, min_y, max_x, max_y = only_zone.polygon.bounds
            exit_w = max(1.2, min(3.0, max(only_zone.polygon.length * 0.03, 1.2)))
            exit_h = max(1.5, min(3.0, max(model.height_m * 0.05, 1.5)))
            exit_poly = [
                (min_x - exit_w, min_y),
                (min_x, min_y),
                (min_x, min_y + exit_h),
                (min_x - exit_w, min_y + exit_h),
            ]
            exit_zone = Zone(
                zone_id="Z_EXIT",
                polygon_points=exit_poly,
                confidence=0.85,
                label="outdoor",
                is_exit=True,
            )
            model.add_zone(exit_zone)
            model.add_opening(
                Opening(
                    opening_id="O_EXIT",
                    zone_a=only_zone.id,
                    zone_b=exit_zone.id,
                    width=1.2,
                    opening_type="emergency_exit",
                    confidence=0.80,
                    is_exit_door=True,
                    midpoint=((min_x + (min_x - exit_w)) / 2.0, min_y + exit_h / 2.0),
                )
            )
            logger.warning(
                "No usable exit found — created synthetic exterior exit zone for preview runs"
            )

        # Ensure at least one exit
        if not model.exit_zones:
            # Fall back: largest zone touching perimeter
            perimeter_zones = sorted(
                model.zones.values(),
                key=lambda z: z.area,
                reverse=True,
            )
            if perimeter_zones:
                perimeter_zones[0].is_exit = True
                logger.warning(
                    f"No perimeter exit found — forced '{perimeter_zones[0].id}' as exit"
                )

        logger.info(
            f"Post-process: {len(model.zones)} valid zones, "
            f"{len(model.exit_zones)} exits, "
            f"{len(model.openings)} openings, "
            f"area={model.total_area_m2:.1f}m²"
        )


# ─── SVG Parser ───────────────────────────────────────────────────────────────

class SVGParser:
    """
    Parses CubiCasa5K SVG annotation files (model.svg).

    CubiCasa SVGs contain labelled groups:
      <g id="Wall">       → wall polygon elements
      <g id="Door">       → door arc/line elements
      <g id="Room">       → room space polygons (space boundaries)
      <g id="Window">     → window elements

    Falls back to path/polygon parsing if groups are not found.
    """

    # Typical CubiCasa SVG pixels to metres: 1px ≈ 0.01m (varies per plan)
    DEFAULT_SCALE_M_PER_PX = 0.01

    def parse(self, svg_path: str) -> BuildingModel:
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Detect namespace
        tag = root.tag
        ns = ""
        if tag.startswith("{"):
            ns = tag[: tag.index("}") + 1]

        width_px, height_px = self._get_svg_dimensions(root, ns)
        scale = self._estimate_scale(root, ns, width_px, height_px)

        logger.info(
            f"SVG: {width_px:.0f}×{height_px:.0f}px, "
            f"scale={scale:.4f} m/px"
        )

        model = BuildingModel(
            building_id=self._derive_building_id(Path(svg_path)),
            scale_m_per_px=scale,
        )

        # Extract rooms / spaces
        zones = self._extract_rooms(root, ns, scale)
        for z in zones:
            model.add_zone(z)

        # Extract walls
        walls = self._extract_walls(root, ns, scale)
        for w in walls:
            model.add_wall(w)

        # Extract doors
        openings = self._extract_doors(root, ns, scale, model)
        for o in openings:
            model.add_opening(o)

        # If no explicit rooms found, fall back to wall-bounded zone extraction
        if not model.zones:
            logger.warning("No Room group found in SVG — falling back to polygon scan")
            zones_fb = self._fallback_polygon_zones(root, ns, scale)
            for z in zones_fb:
                model.add_zone(z)
            if not model.openings and model.zones:
                for opening in self._infer_openings_from_adjacency(model):
                    model.add_opening(opening)

        return model

    @staticmethod
    def _derive_building_id(svg_path: Path) -> str:
        stem = svg_path.stem
        parent = svg_path.parent.name
        generic_dirs = {"input_blueprints", "processed", "outputs", "custom_blueprints"}
        if stem.lower() == "model" and parent not in generic_dirs:
            return parent
        return stem

    def _get_svg_dimensions(self, root, ns: str) -> Tuple[float, float]:
        w = root.get("width", "1000").replace("px", "").replace("pt", "")
        h = root.get("height", "1000").replace("px", "").replace("pt", "")
        try:
            return float(w), float(h)
        except ValueError:
            vb = root.get("viewBox", "0 0 1000 1000").split()
            return float(vb[2]), float(vb[3])

    def _estimate_scale(self, root, ns: str, w_px: float, h_px: float) -> float:
        """
        Estimate metres-per-pixel from SVG metadata or heuristic.
        CubiCasa plans are typically ~10–50m × 10–50m for residential.
        Heuristic: assume 20m wide building → scale = 20 / width_px.
        """
        # Try to find a <text> or metadata with scale hint
        assumed_width_m = 20.0  # Typical residential floor plan width
        if w_px > 0:
            return assumed_width_m / w_px
        return self.DEFAULT_SCALE_M_PER_PX

    def _extract_rooms(self, root, ns: str, scale: float) -> List[Zone]:
        """Extract room/space geometry from SVG."""
        zones: List[Zone] = []
        room_keywords = {"room", "space", "area", "floorspace"}

        for group in root.iter():
            tag = group.tag.split("}")[-1].lower()
            if tag != "g":
                continue

            group_id = (group.get("id") or "").lower()
            group_class = (group.get("class") or "").lower()
            if not any(k in group_id or k in group_class for k in room_keywords):
                continue

            polygon = None
            for child in list(group):
                child_tag = child.tag.split("}")[-1].lower()
                if child_tag not in {"polygon", "path", "rect"}:
                    continue
                polygon = self._largest_polygon(self._elem_to_polygon(child, child_tag, scale))
                if polygon is not None:
                    break

            if polygon is None or polygon.area < MIN_ZONE_AREA_M2:
                continue

            zone_id = f"Z{len(zones):04d}"
            zone = Zone(
                zone_id=zone_id,
                polygon_points=list(polygon.exterior.coords),
                confidence=0.95,
                label=self._label_from_class(group_class),
            )
            zones.append(zone)

        logger.info(f"SVG: extracted {len(zones)} room zones")
        return zones

    def _extract_walls(self, root, ns: str, scale: float) -> List[WallSegment]:
        """Extract wall geometry from SVG."""
        walls: List[WallSegment] = []
        wall_keywords = {"wall", "walls"}

        for group in root.iter():
            gid = (group.get("id") or "").lower()
            if any(k in gid for k in wall_keywords):
                for elem in group:
                    tag = elem.tag.split("}")[-1].lower()
                    if tag == "rect":
                        segs = self._rect_to_wall_segments(elem, scale)
                        walls.extend(segs)
                    elif tag == "line":
                        seg = self._line_to_wall_segment(elem, scale)
                        if seg:
                            walls.append(seg)
                    elif tag in {"polyline", "polygon"}:
                        segs = self._poly_to_wall_segments(elem, tag, scale)
                        walls.extend(segs)

        logger.info(f"SVG: extracted {len(walls)} wall segments")
        return walls

    def _extract_doors(
        self, root, ns: str, scale: float, model: BuildingModel
    ) -> List[Opening]:
        """Extract door/opening geometry from SVG."""
        openings: List[Opening] = []
        door_keywords = {"door", "window"}
        op_counter = 0

        for group in root.iter():
            gid = (group.get("id") or "").lower()
            if any(k in gid for k in door_keywords):
                opening_type = "door" if "door" in gid else "window"
                for elem in group:
                    tag = elem.tag.split("}")[-1].lower()
                    if tag in {"line", "rect", "path", "polygon"}:
                        door_info = self._elem_to_door(
                            elem, tag, scale, opening_type
                        )
                        if door_info:
                            mid_x, mid_y, width_m = door_info
                            # Find which two zones this door connects
                            z_pair = self._find_adjacent_zones(
                                mid_x, mid_y, model, search_radius=3.0
                            )
                            if z_pair and len(z_pair) >= 2:
                                oid = f"O{op_counter:04d}"
                                o = Opening(
                                    opening_id=oid,
                                    zone_a=z_pair[0],
                                    zone_b=z_pair[1],
                                    width=max(width_m, MIN_OPENING_WIDTH_M),
                                    opening_type=opening_type,
                                    confidence=0.85,
                                    midpoint=(mid_x, mid_y),
                                )
                                openings.append(o)
                                op_counter += 1

        # If no explicit doors found, infer from adjacent zone proximity
        if not openings and model.zones:
            logger.warning("No doors found in SVG — inferring connections from zone adjacency")
            openings = self._infer_openings_from_adjacency(model)

        logger.info(f"SVG: extracted {len(openings)} openings")
        return openings

    def _find_adjacent_zones(
        self, x: float, y: float, model: BuildingModel, search_radius: float = 3.0
    ) -> List[str]:
        """Find zones near a door midpoint (candidates to connect)."""
        pt = Point(x, y)
        nearby = []
        for zid, zone in model.zones.items():
            dist = zone.polygon.distance(pt)
            if dist <= search_radius:
                nearby.append((dist, zid))
        nearby.sort()
        return [zid for _, zid in nearby[:2]]

    def _infer_openings_from_adjacency(self, model: BuildingModel) -> List[Opening]:
        """Infer openings from zones with touching or near-touching boundaries."""
        openings = []
        zone_list = list(model.zones.values())
        op_counter = 0

        for i in range(len(zone_list)):
            for j in range(i + 1, len(zone_list)):
                za = zone_list[i]
                zb = zone_list[j]
                dist = za.polygon.distance(zb.polygon)

                # Adjacent zones have dist ≈ 0 (touching) or < wall thickness
                if dist < 1.0:
                    # Estimate shared boundary length as opening width
                    try:
                        shared = za.polygon.intersection(zb.polygon.buffer(0.5))
                        width = max(shared.length, MIN_OPENING_WIDTH_M)
                        midpoint_geom = shared.centroid if not shared.is_empty else None
                    except Exception:
                        width = 1.0
                        midpoint_geom = None

                    midpoint = None
                    if midpoint_geom is not None and not midpoint_geom.is_empty:
                        midpoint = (midpoint_geom.x, midpoint_geom.y)
                    else:
                        midpoint = (
                            (za.centroid.x + zb.centroid.x) / 2.0,
                            (za.centroid.y + zb.centroid.y) / 2.0,
                        )

                    oid = f"O{op_counter:04d}"
                    o = Opening(
                        opening_id=oid,
                        zone_a=za.id,
                        zone_b=zb.id,
                        width=min(width, 4.0),
                        opening_type="inferred",
                        confidence=0.60,
                        midpoint=midpoint,
                    )
                    openings.append(o)
                    op_counter += 1

        return openings

    def _fallback_polygon_zones(
        self, root, ns: str, scale: float
    ) -> List[Zone]:
        """Last-resort: collect all SVG polygons > MIN_ZONE_AREA and treat as zones."""
        zones = []
        counter = 0
        for elem in root.iter():
            tag = elem.tag.split("}")[-1].lower()
            if tag in {"polygon", "path"}:
                poly = self._largest_polygon(self._elem_to_polygon(elem, tag, scale))
                if poly and poly.area >= MIN_ZONE_AREA_M2:
                    parent_class = (elem.get("class") or "").lower()
                    zid = f"Z{counter:04d}"
                    zones.append(Zone(zid, list(poly.exterior.coords), label=self._label_from_class(parent_class)))
                    counter += 1
        return zones

    @staticmethod
    def _label_from_class(group_class: str) -> str:
        tokens = [token for token in group_class.split() if token.lower() != "space"]
        return tokens[0].lower() if tokens else "zone"

    @staticmethod
    def _largest_polygon(geometry) -> Optional[Polygon]:
        if geometry is None:
            return None
        if isinstance(geometry, Polygon):
            return geometry if not geometry.is_empty else None
        if isinstance(geometry, MultiPolygon):
            polygons = [poly for poly in geometry.geoms if not poly.is_empty]
            return max(polygons, key=lambda poly: poly.area) if polygons else None
        if isinstance(geometry, GeometryCollection):
            polygons = [
                geom for geom in geometry.geoms
                if isinstance(geom, Polygon) and not geom.is_empty
            ]
            return max(polygons, key=lambda poly: poly.area) if polygons else None
        return None

    # ─── SVG Element Parsers ──────────────────────────────────────────────────

    def _elem_to_polygon(
        self, elem, tag: str, scale: float
    ) -> Optional[Polygon]:
        try:
            if tag == "rect":
                return self._rect_to_polygon(elem, scale)
            elif tag == "polygon":
                return self._svg_polygon_to_polygon(elem, scale)
            elif tag == "path":
                return self._path_to_polygon(elem, scale)
        except Exception as e:
            logger.debug(f"Element parse error ({tag}): {e}")
        return None

    def _rect_to_polygon(self, elem, scale: float) -> Optional[Polygon]:
        x = float(elem.get("x", 0)) * scale
        y = float(elem.get("y", 0)) * scale
        w = float(elem.get("width", 0)) * scale
        h = float(elem.get("height", 0)) * scale
        if w <= 0 or h <= 0:
            return None
        return Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])

    def _svg_polygon_to_polygon(self, elem, scale: float) -> Optional[Polygon]:
        pts_str = elem.get("points", "")
        pts = self._parse_svg_points(pts_str, scale)
        if len(pts) < 3:
            return None
        return make_valid(Polygon(pts))

    def _path_to_polygon(self, elem, scale: float) -> Optional[Polygon]:
        """Parse SVG <path d="..."> to Polygon. Handles M/L/Z commands."""
        d = elem.get("d", "")
        pts = self._parse_svg_path(d, scale)
        if len(pts) < 3:
            return None
        return make_valid(Polygon(pts))

    def _rect_to_wall_segments(self, elem, scale: float) -> List[WallSegment]:
        try:
            x = float(elem.get("x", 0)) * scale
            y = float(elem.get("y", 0)) * scale
            w = float(elem.get("width", 0)) * scale
            h = float(elem.get("height", 0)) * scale
            return [
                WallSegment(x, y, x + w, y),
                WallSegment(x + w, y, x + w, y + h),
                WallSegment(x + w, y + h, x, y + h),
                WallSegment(x, y + h, x, y),
            ]
        except Exception:
            return []

    def _line_to_wall_segment(self, elem, scale: float) -> Optional[WallSegment]:
        try:
            x1 = float(elem.get("x1", 0)) * scale
            y1 = float(elem.get("y1", 0)) * scale
            x2 = float(elem.get("x2", 0)) * scale
            y2 = float(elem.get("y2", 0)) * scale
            return WallSegment(x1, y1, x2, y2)
        except Exception:
            return None

    def _poly_to_wall_segments(
        self, elem, tag: str, scale: float
    ) -> List[WallSegment]:
        pts_str = elem.get("points", "")
        pts = self._parse_svg_points(pts_str, scale)
        segments = []
        close = tag == "polygon"
        for i in range(len(pts) - 1):
            segments.append(WallSegment(*pts[i], *pts[i + 1]))
        if close and len(pts) >= 2:
            segments.append(WallSegment(*pts[-1], *pts[0]))
        return segments

    def _elem_to_door(
        self, elem, tag: str, scale: float, opening_type: str
    ) -> Optional[Tuple[float, float, float]]:
        """Return (mid_x, mid_y, width_m) for a door element."""
        try:
            if tag == "rect":
                x = float(elem.get("x", 0)) * scale
                y = float(elem.get("y", 0)) * scale
                w = float(elem.get("width", 0)) * scale
                h = float(elem.get("height", 0)) * scale
                cx, cy = x + w / 2, y + h / 2
                width = max(w, h)
                return cx, cy, width
            elif tag == "line":
                x1 = float(elem.get("x1", 0)) * scale
                y1 = float(elem.get("y1", 0)) * scale
                x2 = float(elem.get("x2", 0)) * scale
                y2 = float(elem.get("y2", 0)) * scale
                import math
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                width = math.hypot(x2 - x1, y2 - y1)
                return cx, cy, width
            elif tag in {"polygon", "path"}:
                polygon = self._largest_polygon(self._elem_to_polygon(elem, tag, scale))
                if polygon is None:
                    return None
                min_x, min_y, max_x, max_y = polygon.bounds
                cx, cy = polygon.centroid.x, polygon.centroid.y
                width = max(max_x - min_x, max_y - min_y)
                return cx, cy, width
        except Exception as e:
            logger.debug(f"Door element parse error: {e}")
        return None

    @staticmethod
    def _parse_svg_points(pts_str: str, scale: float) -> List[Tuple[float, float]]:
        """Parse SVG points attribute: "x1,y1 x2,y2 ..." """
        pts = []
        tokens = re.split(r"[\s,]+", pts_str.strip())
        it = iter(tokens)
        try:
            while True:
                x = float(next(it)) * scale
                y = float(next(it)) * scale
                pts.append((x, y))
        except StopIteration:
            pass
        return pts

    @staticmethod
    def _parse_svg_path(d: str, scale: float) -> List[Tuple[float, float]]:
        """Minimal SVG path parser — handles M, L, H, V, Z commands."""
        pts = []
        commands = re.findall(r"([MmLlHhVvZz])\s*([-\d.\s,]*)", d)
        cx, cy = 0.0, 0.0
        for cmd, args_str in commands:
            args = [float(v) for v in re.split(r"[\s,]+", args_str.strip()) if v]
            if cmd in "Mm":
                for i in range(0, len(args) - 1, 2):
                    if cmd == "M":
                        cx, cy = args[i] * scale, args[i + 1] * scale
                    else:
                        cx += args[i] * scale
                        cy += args[i + 1] * scale
                    pts.append((cx, cy))
            elif cmd in "Ll":
                for i in range(0, len(args) - 1, 2):
                    if cmd == "L":
                        cx, cy = args[i] * scale, args[i + 1] * scale
                    else:
                        cx += args[i] * scale
                        cy += args[i + 1] * scale
                    pts.append((cx, cy))
            elif cmd in "Hh":
                for a in args:
                    cx = a * scale if cmd == "H" else cx + a * scale
                    pts.append((cx, cy))
            elif cmd in "Vv":
                for a in args:
                    cy = a * scale if cmd == "V" else cy + a * scale
                    pts.append((cx, cy))
            elif cmd in "Zz":
                if pts:
                    pts.append(pts[0])
        return pts


# ─── OpenCV Image Pipeline ────────────────────────────────────────────────────

class ImagePipeline:
    """
    Full OpenCV-based pipeline for raster blueprint images.
    Used when a PNG/JPG blueprint is provided instead of SVG.
    """

    def __init__(self, cell_size_m: float = 0.5) -> None:
        self.cell_size_m = cell_size_m

    def process(self, img_path: str) -> BuildingModel:
        import cv2

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")

        h, w = img.shape
        logger.info(f"Image pipeline: {w}×{h}px")

        # Scale estimation (default: 0.05 m/px for A4 at ~150dpi)
        scale = 0.05

        # Wall binary mask
        wall_binary = preprocess_for_wall_detection(img)

        # Room contours
        min_area_px = int((2.0 / scale) ** 2)  # 2m² minimum in pixels
        contours = extract_room_contours(wall_binary, min_area_px=min_area_px)

        model = BuildingModel(
            building_id=Path(img_path).stem,
            scale_m_per_px=scale,
        )

        zone_counter = 0
        for contour in contours:
            poly = contour_to_polygon(
                contour,
                epsilon_fraction=CONTOUR_APPROX_EPSILON,
                scale_m_per_px=scale,
            )
            if poly and poly.area >= MIN_ZONE_AREA_M2:
                zid = f"Z{zone_counter:04d}"
                z = Zone(zid, list(poly.exterior.coords), confidence=0.75)
                model.add_zone(z)
                zone_counter += 1

        logger.info(f"Image pipeline: {zone_counter} zones extracted")

        # Door gap detection
        skeleton = skeletonise(wall_binary)
        gap_px = int(MIN_OPENING_WIDTH_M / scale)
        gaps = detect_door_gaps(skeleton, min_gap_px=gap_px, max_gap_px=gap_px * 4)

        op_counter = 0
        for (mid_r, mid_c, gap_size, orientation) in gaps:
            mid_x = mid_c * scale
            mid_y = mid_r * scale
            width_m = gap_size * scale

            # Find adjacent zones
            pt = Point(mid_x, mid_y)
            nearby = []
            for zid, zone in model.zones.items():
                d = zone.polygon.distance(pt)
                nearby.append((d, zid))
            nearby.sort()

            if len(nearby) >= 2:
                oid = f"O{op_counter:04d}"
                o = Opening(
                    opening_id=oid,
                    zone_a=nearby[0][1],
                    zone_b=nearby[1][1],
                    width=max(width_m, MIN_OPENING_WIDTH_M),
                    opening_type="door",
                    confidence=0.70,
                    midpoint=(mid_x, mid_y),
                )
                model.add_opening(o)
                op_counter += 1

        # Add wall segments from wall mask boundary
        wall_contours, _ = cv2.findContours(
            wall_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for wc in wall_contours:
            pts = [(float(p[0][0]) * scale, float(p[0][1]) * scale) for p in wc]
            for i in range(len(pts) - 1):
                model.add_wall(WallSegment(*pts[i], *pts[i + 1]))

        logger.info(
            f"Image pipeline complete: {len(model.zones)} zones, "
            f"{len(model.openings)} openings"
        )
        return model
