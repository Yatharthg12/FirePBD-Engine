"""
FirePBD Engine — Geometry & Graph Validation
=============================================
Validates zone polygons, graph connectivity, and building model integrity.
Produces structured ValidationReport used by the API and simulation engine.
"""
from __future__ import annotations

from typing import List

from shapely.validation import explain_validity

from backend.core.geometry import BuildingModel, Zone
from backend.core.graph_model import SpatialGraph
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum zone area in m² to be considered valid
MIN_ZONE_AREA_M2 = 1.0
# Minimum number of exits in a building
MIN_EXITS = 1


class ValidationReport:
    """Structured container for validation results."""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        logger.error(f"VALIDATION ERROR: {msg}")

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(f"VALIDATION WARNING: {msg}")

    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> dict:
        return {
            "valid": self.is_valid(),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def merge(self, other: "ValidationReport") -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def __repr__(self) -> str:
        return f"ValidationReport(errors={len(self.errors)}, warnings={len(self.warnings)})"


class GeometryValidator:
    """Validates individual zones and polygon integrity."""

    @staticmethod
    def validate_zones(zones: List[Zone]) -> ValidationReport:
        report = ValidationReport()

        if not zones:
            report.add_error("No zones defined — building model is empty")
            return report

        for zone in zones:
            poly = zone.polygon

            if not poly.is_valid:
                report.add_error(
                    f"Zone '{zone.id}' invalid polygon: {explain_validity(poly)}"
                )

            if poly.area < MIN_ZONE_AREA_M2:
                report.add_warning(
                    f"Zone '{zone.id}' very small area: {poly.area:.2f} m² "
                    f"(min {MIN_ZONE_AREA_M2} m²)"
                )

            if poly.is_empty:
                report.add_error(f"Zone '{zone.id}' has empty geometry")

            if zone.confidence < 0.5:
                report.add_warning(
                    f"Zone '{zone.id}' low confidence: {zone.confidence:.2f}"
                )

        return report


class GraphValidator:
    """Validates navigation graph structure."""

    @staticmethod
    def validate_graph(graph: SpatialGraph) -> ValidationReport:
        report = ValidationReport()

        if len(graph.graph.nodes) == 0:
            report.add_error("Spatial graph has no nodes — nothing to simulate")
            return report

        if len(graph.graph.edges) == 0:
            report.add_error(
                "Spatial graph has no edges — zones are isolated (no connectivity)"
            )

        if not graph.is_fully_connected():
            components = graph.get_components()
            report.add_error(
                f"Graph is NOT fully connected — {len(components)} disconnected components"
            )
            for i, comp in enumerate(components):
                report.add_warning(f"  Component {i + 1}: {sorted(comp)}")

        # Check for isolated nodes (degree 0)
        isolated = [n for n in graph.graph.nodes if graph.graph.degree(n) == 0]
        if isolated:
            report.add_warning(f"Isolated nodes (no connections): {isolated}")

        return report


class BuildingModelValidator:
    """Validates overall BuildingModel integrity."""

    @staticmethod
    def validate(model: BuildingModel, graph: SpatialGraph) -> ValidationReport:
        report = ValidationReport()

        # Geometry check
        geo_report = GeometryValidator.validate_zones(list(model.zones.values()))
        report.merge(geo_report)

        # Graph check
        graph_report = GraphValidator.validate_graph(graph)
        report.merge(graph_report)

        # Exit check
        if not model.exit_zones:
            report.add_error(
                "No exit zones defined — evacuation impossible. "
                "Mark at least one zone as is_exit=True."
            )

        if len(model.exit_zones) < MIN_EXITS:
            report.add_warning(
                f"Only {len(model.exit_zones)} exit(s) defined. "
                "BS 9999 recommends at least 2 independent means of escape."
            )

        # Opening check
        if not model.openings:
            report.add_error(
                "No openings (doors) defined — zones cannot be traversed"
            )

        # Scale check
        if model.scale_m_per_px <= 0:
            report.add_error(f"Invalid scale factor: {model.scale_m_per_px}")

        if model.total_area_m2 < 10:
            report.add_warning(
                f"Very small total area: {model.total_area_m2:.1f} m²"
            )

        status = "✓ VALID" if report.is_valid() else "✗ INVALID"
        logger.info(
            f"Validation {status} — {len(report.errors)} errors, "
            f"{len(report.warnings)} warnings"
        )
        return report


# Backward-compatible aliases used in tests
class SystemValidator:
    @staticmethod
    def validate(zones, graph) -> ValidationReport:
        report = ValidationReport()
        geo_report = GeometryValidator.validate_zones(zones)
        graph_report = GraphValidator.validate_graph(graph)
        report.merge(geo_report)
        report.merge(graph_report)
        return report