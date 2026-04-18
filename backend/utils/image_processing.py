"""
FirePBD Engine — Image Processing Utilities
=============================================
OpenCV-based helpers for blueprint image preprocessing,
wall detection, and contour extraction.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

from backend.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Image Preprocessing ──────────────────────────────────────────────────────

def load_blueprint_image(path: str) -> np.ndarray:
    """Load a blueprint image as grayscale uint8."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    logger.info(f"Loaded blueprint: {path}, size={img.shape[::-1]}")
    return img


def preprocess_for_wall_detection(img: np.ndarray) -> np.ndarray:
    """
    Convert a raw blueprint image to a binary wall mask.

    Pipeline:
      1. Normalise intensity
      2. Gaussian denoise
      3. Otsu thresholding → binary
      4. Morphological close to fill small gaps in walls

    Returns: uint8 binary image (255=wall, 0=open)
    """
    # Normalise to full 0-255 range
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    # Denoise
    denoised = cv2.GaussianBlur(norm, (3, 3), 0)

    # Otsu threshold — walls are typically darker than floors
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological close: fill tiny holes in wall lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed


def extract_room_contours(
    binary_wall_img: np.ndarray,
    min_area_px: int = 500,
) -> List[np.ndarray]:
    """
    Extract room-interior contours from a binary wall image.

    Strategy:
      1. Invert wall mask → room interiors are foreground
      2. Find external contours
      3. Filter by minimum area

    Returns list of contour arrays (N×1×2, int32)
    """
    # Invert: rooms are bright, walls are dark
    room_mask = cv2.bitwise_not(binary_wall_img)

    # Remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter small artefacts
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    logger.debug(f"Found {len(filtered)} room contours (min_area={min_area_px}px)")
    return filtered


def contour_to_polygon(
    contour: np.ndarray,
    epsilon_fraction: float = 0.02,
    scale_m_per_px: float = 1.0,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> Optional[Polygon]:
    """
    Convert OpenCV contour to a Shapely Polygon in world coordinates.

    Applies Douglas-Peucker simplification then scales to metres.
    """
    arc_len = cv2.arcLength(contour, True)
    epsilon = epsilon_fraction * arc_len
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < 3:
        return None

    pts = [
        (
            float(pt[0][0]) * scale_m_per_px + origin[0],
            float(pt[0][1]) * scale_m_per_px + origin[1],
        )
        for pt in approx
    ]

    try:
        poly = Polygon(pts)
        poly = make_valid(poly)
        if not poly.is_valid or poly.area <= 0:
            return None
        return poly
    except Exception:
        return None


def detect_wall_lines(
    binary_wall_img: np.ndarray,
    min_line_length: int = 30,
    max_line_gap: int = 5,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect straight wall lines using probabilistic Hough transform.

    Returns list of (x1, y1, x2, y2) line segments in pixels.
    """
    edges = cv2.Canny(binary_wall_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []
    return [(int(l[0][0]), int(l[0][1]), int(l[0][2]), int(l[0][3])) for l in lines]


def detect_door_gaps(
    wall_skeleton: np.ndarray,
    min_gap_px: int = 10,
    max_gap_px: int = 60,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect potential door gaps in skeletonised wall image.

    Gaps are short breaks in wall line continuity.
    Returns list of (mid_row, mid_col, gap_size_px, orientation) tuples.
    """
    gaps = []
    # Horizontal scan — look for breaks in horizontal runs
    for r in range(wall_skeleton.shape[0]):
        row = wall_skeleton[r, :]
        in_gap = False
        gap_start = 0
        for c in range(wall_skeleton.shape[1]):
            if row[c] == 0 and not in_gap:
                in_gap = True
                gap_start = c
            elif row[c] > 0 and in_gap:
                gap_size = c - gap_start
                if min_gap_px <= gap_size <= max_gap_px:
                    gaps.append((r, gap_start + gap_size // 2, gap_size, "H"))
                in_gap = False
    # Vertical scan
    for c in range(wall_skeleton.shape[1]):
        col = wall_skeleton[:, c]
        in_gap = False
        gap_start = 0
        for r in range(wall_skeleton.shape[0]):
            if col[r] == 0 and not in_gap:
                in_gap = True
                gap_start = r
            elif col[r] > 0 and in_gap:
                gap_size = r - gap_start
                if min_gap_px <= gap_size <= max_gap_px:
                    gaps.append((gap_start + gap_size // 2, c, gap_size, "V"))
                in_gap = False
    return gaps


def skeletonise(binary_img: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonisation of a binary image.
    Returns the skeleton (uint8, 255 = skeleton pixel).
    """
    skeleton = np.zeros_like(binary_img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp = binary_img.copy()

    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.dilate(eroded, element)
        diff = cv2.subtract(temp, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    return skeleton


def estimate_scale_from_image(img: np.ndarray) -> float:
    """
    Heuristic scale estimation.
    If no scale bar is detected, assume 1 pixel = 0.05 m (typical for A4 blueprint at 300dpi).
    A more robust implementation would detect the scale bar using OCR or template matching.
    """
    # Default fallback: typical CubiCasa floor plan scale
    assumed_m_per_px = 0.05
    logger.warning(
        f"Using default scale: {assumed_m_per_px} m/px. "
        "Provide scale bar detection for accurate results."
    )
    return assumed_m_per_px


def annotate_image(
    img: np.ndarray,
    zones: list,
    exits: list,
    openings: list,
) -> np.ndarray:
    """
    Draw zone outlines, exits, and openings on a copy of the image for debugging.
    Returns a colour BGR image.
    """
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()

    for zone in zones:
        coords = list(zone.polygon.exterior.coords)
        pts = np.array([[int(x), int(y)] for x, y in coords], dtype=np.int32)
        colour = (0, 200, 255) if zone.is_exit else (0, 255, 100)
        cv2.polylines(vis, [pts], True, colour, 2)

    return vis
