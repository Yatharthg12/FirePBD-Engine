"""
FirePBD Engine — Evacuation Optimization Agent
================================================
Analyses risk report findings and generates ranked structural improvement
recommendations that engineering teams can act upon.

Improvement categories:
  1. EXIT ADDITION   — add new exit where none exists near high-risk zone
  2. EXIT WIDENING   — widen bottleneck doors to reduce queue time
  3. LAYOUT CHANGE   — suggest new opening between adjacent zones
  4. OCCUPANCY LOAD  — recommend reducing occupancy in overloaded zones
  5. FIRE LOAD       — recommend fuel load reduction (storage changes)

Each recommendation includes:
  - Priority (CRITICAL / HIGH / MEDIUM / LOW)
  - Estimated RSET improvement (seconds)
  - Affected zones
  - Engineering rationale
  - Applicable standard reference
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.core.constants import (
    ASET_SAFETY_MARGIN_S,
    DOOR_FLOW_RATE_P_M_S,
    MIN_OPENING_WIDTH_M,
    SIMULATION_TIMESTEP_S,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum clear exit width per NFPA 101 §7.2.1.2
NFPA_MIN_EXIT_WIDTH_M = 0.81
# BS 9999 recommended minimum corridor width
BS9999_MIN_CORRIDOR_WIDTH_M = 1.0


@dataclass
class Recommendation:
    """A single improvement recommendation."""
    rec_id: str
    category: str               # EXIT_ADDITION / EXIT_WIDENING / LAYOUT_CHANGE / etc.
    priority: str               # CRITICAL / HIGH / MEDIUM / LOW
    title: str
    description: str
    affected_zones: List[str]
    estimated_rset_reduction_s: float = 0.0
    estimated_risk_reduction: float = 0.0
    standard_reference: str = ""
    implementation_cost: str = "MEDIUM"   # LOW / MEDIUM / HIGH

    def to_dict(self) -> dict:
        return {
            "rec_id": self.rec_id,
            "category": self.category,
            "priority": self.priority,
            "title": self.title,
            "description": self.description,
            "affected_zones": self.affected_zones,
            "estimated_rset_reduction_s": round(self.estimated_rset_reduction_s, 0),
            "estimated_risk_reduction": round(self.estimated_risk_reduction, 1),
            "standard_reference": self.standard_reference,
            "implementation_cost": self.implementation_cost,
        }


class EvacuationOptimizer:
    """
    Generates prioritised improvement recommendations from risk analysis results.

    Parameters
    ----------
    model : BuildingModel
    graph : SpatialGraph
    risk_report : dict (output from RiskAnalyzer.generate_report)
    evac_summary : dict (output from EvacuationSimulator.summary)
    """

    def __init__(
        self,
        model,
        graph,
        risk_report: dict,
        evac_summary: dict,
    ) -> None:
        self.model = model
        self.graph = graph
        self.risk_report = risk_report
        self.evac_summary = evac_summary
        self._rec_counter = 0

    def generate(self) -> List[Recommendation]:
        """
        Run all checks and return sorted recommendation list.
        Higher priority and larger RSET improvement → ranked first.
        """
        recs: List[Recommendation] = []

        recs.extend(self._check_dead_zones())
        recs.extend(self._check_bottleneck_openings())
        recs.extend(self._check_exit_count())
        recs.extend(self._check_exit_width())
        recs.extend(self._check_occupancy_overload())
        recs.extend(self._check_fuel_load())
        recs.extend(self._check_isolated_zones())

        # Sort: CRITICAL first, then by estimated RSET reduction descending
        priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        recs_sorted = sorted(
            recs,
            key=lambda r: (
                priority_order.get(r.priority, 4),
                -r.estimated_rset_reduction_s,
            ),
        )

        logger.info(f"Generated {len(recs_sorted)} recommendations")
        for r in recs_sorted[:5]:
            logger.info(f"  [{r.priority}] {r.title}")

        return recs_sorted

    # ─── Dead Zone Analysis ───────────────────────────────────────────────────

    def _check_dead_zones(self) -> List[Recommendation]:
        recs = []
        dead_zones = self.risk_report.get("dead_zones", [])

        for dz in dead_zones:
            zone_id = dz["zone_id"]
            time_s = dz.get("first_untenable_time_s", 0)

            if time_s < 120:
                priority = "CRITICAL"
                rset_reduction = 90.0
            elif time_s < 300:
                priority = "HIGH"
                rset_reduction = 45.0
            else:
                priority = "MEDIUM"
                rset_reduction = 20.0

            zone = self.model.zones.get(zone_id)
            area = zone.area if zone else 0

            recs.append(Recommendation(
                rec_id=self._next_id(),
                category="DEAD_ZONE_MITIGATION",
                priority=priority,
                title=f"Zone {zone_id} became untenable at {time_s:.0f}s",
                description=(
                    f"Zone '{zone_id}' (area={area:.1f}m\u00b2) exceeded tenability limits "
                    "(ISO 13571)"
                    f" at t={time_s:.0f}s. "
                    f"Actions: (1) Install suppression system; "
                    f"(2) Add direct exit from this zone; "
                    f"(3) Reduce fuel load density; "
                    f"(4) Install fire door between this zone and neighbours."
                ),
                affected_zones=[zone_id],
                estimated_rset_reduction_s=rset_reduction,
                estimated_risk_reduction=15.0,
                standard_reference="ISO 13571:2012 §5; NFPA 101 §7.2",
                implementation_cost="HIGH",
            ))
        return recs

    # ─── Bottleneck Opening Check ─────────────────────────────────────────────

    def _check_bottleneck_openings(self) -> List[Recommendation]:
        recs = []
        bottlenecks = self.risk_report.get("bottlenecks", [])

        for b in bottlenecks:
            if b["type"] == "opening" and b["severity"] in {"HIGH", "MEDIUM"}:
                z1 = b["zone_a"]
                z2 = b["zone_b"]
                edge_data = self.graph.get_opening_between(z1, z2)
                current_width = edge_data.get("width", 1.0) if edge_data else 1.0
                recommended_width = current_width * 1.5

                current_flow = DOOR_FLOW_RATE_P_M_S * current_width * SIMULATION_TIMESTEP_S
                new_flow = DOOR_FLOW_RATE_P_M_S * recommended_width * SIMULATION_TIMESTEP_S
                persons_helped = int(new_flow - current_flow)
                rset_reduction = persons_helped * 2.5  # rough estimate

                recs.append(Recommendation(
                    rec_id=self._next_id(),
                    category="EXIT_WIDENING",
                    priority="HIGH" if b["severity"] == "HIGH" else "MEDIUM",
                    title=f"Widen passage between {z1} and {z2}",
                    description=(
                        f"Opening between '{z1}'↔'{z2}' has high betweenness centrality "
                        f"(score={b['centrality_score']:.2f}), indicating severe congestion. "
                        f"Current width: {current_width:.1f}m. "
                        f"Recommended: ≥{recommended_width:.1f}m clear width. "
                        f"This increases flow capacity from {current_flow:.0f} to "
                        f"{new_flow:.0f} persons per {SIMULATION_TIMESTEP_S:.0f}s step."
                    ),
                    affected_zones=[z1, z2],
                    estimated_rset_reduction_s=rset_reduction,
                    estimated_risk_reduction=10.0,
                    standard_reference="NFPA 101 §7.2.1; Nelson & MacLennan SFPE 3-13.19",
                    implementation_cost="MEDIUM",
                ))
        return recs

    # ─── Exit Count Check ─────────────────────────────────────────────────────

    def _check_exit_count(self) -> List[Recommendation]:
        recs = []
        n_exits = len(self.model.exit_zones)

        if n_exits < 2:
            recs.append(Recommendation(
                rec_id=self._next_id(),
                category="EXIT_ADDITION",
                priority="CRITICAL",
                title=f"Insufficient exits: only {n_exits} exit(s) detected",
                description=(
                    f"BS 9999:2017 and NFPA 101 require a minimum of 2 independent "
                    f"means of escape for buildings with occupancy > 60 persons. "
                    f"Current building has {n_exits} exit(s). "
                    f"Add a minimum of {2 - n_exits} additional exit(s) on the "
                    f"opposite side of the building from the existing exit."
                ),
                affected_zones=[z.id for z in self.model.exit_zones],
                estimated_rset_reduction_s=120.0,
                estimated_risk_reduction=25.0,
                standard_reference="BS 9999:2017 §13; NFPA 101 §7.4",
                implementation_cost="HIGH",
            ))

        # Check for zones with only one path to exit (single point of failure)
        for zone_id in self.model.zones:
            if self.model.zones[zone_id].is_exit:
                continue
            paths = 0
            for exit_id in self.model.exit_zone_ids:
                if self.graph.shortest_path(zone_id, exit_id):
                    paths += 1
            if paths <= 1:
                recs.append(Recommendation(
                    rec_id=self._next_id(),
                    category="EXIT_ADDITION",
                    priority="HIGH",
                    title=f"Zone {zone_id} has single evacuation path",
                    description=(
                        f"Zone '{zone_id}' has only one available evacuation path. "
                        f"If this path becomes untenable, occupants are trapped. "
                        f"Add an additional opening or emergency exit from this zone."
                    ),
                    affected_zones=[zone_id],
                    estimated_rset_reduction_s=60.0,
                    estimated_risk_reduction=12.0,
                    standard_reference="BS 9999:2017 §13.2; IS 16009:2010",
                    implementation_cost="MEDIUM",
                ))
        return recs

    # ─── Exit Width Check ─────────────────────────────────────────────────────

    def _check_exit_width(self) -> List[Recommendation]:
        recs = []
        for zone in self.model.exit_zones:
            # Check all openings leading to this zone
            for opening in self.model.openings.values():
                if zone.id in {opening.zone_a, opening.zone_b}:
                    if opening.width < NFPA_MIN_EXIT_WIDTH_M:
                        recs.append(Recommendation(
                            rec_id=self._next_id(),
                            category="EXIT_WIDENING",
                            priority="CRITICAL",
                            title=f"Exit opening {opening.id} below minimum width",
                            description=(
                                f"Opening '{opening.id}' serving exit zone '{zone.id}' "
                                f"has width {opening.width:.2f}m, below NFPA 101 minimum "
                                f"of {NFPA_MIN_EXIT_WIDTH_M}m. "
                                f"Widen to minimum {NFPA_MIN_EXIT_WIDTH_M}m clear width."
                            ),
                            affected_zones=[opening.zone_a, opening.zone_b],
                            estimated_rset_reduction_s=30.0,
                            estimated_risk_reduction=8.0,
                            standard_reference="NFPA 101 §7.2.1.2",
                            implementation_cost="MEDIUM",
                        ))
        return recs

    # ─── Occupancy Overload ───────────────────────────────────────────────────

    def _check_occupancy_overload(self) -> List[Recommendation]:
        recs = []
        for zone in self.model.zones.values():
            if zone.is_exit:
                continue
            # Check if zone density is very high
            if zone.occupancy_load > 0.5:  # > 0.5 persons/m²
                recs.append(Recommendation(
                    rec_id=self._next_id(),
                    category="OCCUPANCY_LOAD",
                    priority="MEDIUM",
                    title=f"High occupancy density in zone {zone.id}",
                    description=(
                        f"Zone '{zone.id}' has occupancy density {zone.occupancy_load:.2f} "
                        f"persons/m² (max occupants: {zone.max_occupants}). "
                        f"High density increases congestion at exits. "
                        f"Consider reducing occupant load or increasing floor area."
                    ),
                    affected_zones=[zone.id],
                    estimated_rset_reduction_s=20.0,
                    estimated_risk_reduction=5.0,
                    standard_reference="NFPA 101 Table 7.3.1.2",
                    implementation_cost="LOW",
                ))
        return recs

    # ─── Fuel Load ────────────────────────────────────────────────────────────

    def _check_fuel_load(self) -> List[Recommendation]:
        recs = []
        dead_zone_ids = {dz["zone_id"] for dz in self.risk_report.get("dead_zones", [])}

        for zone_id, zone in self.model.zones.items():
            if zone_id in dead_zone_ids and zone.fuel_load_density > 600:
                recs.append(Recommendation(
                    rec_id=self._next_id(),
                    category="FIRE_LOAD",
                    priority="HIGH",
                    title=f"High fuel load in untenable zone {zone_id}",
                    description=(
                        f"Zone '{zone_id}' has fuel load density {zone.fuel_load_density:.0f} "
                        f"MJ/m² and became untenable during simulation. "
                        f"Recommended actions: "
                        f"(1) Reduce combustible materials to < 420 MJ/m² (office default); "
                        f"(2) Install automatic suppression (sprinklers); "
                        f"(3) Increase fire compartment separation."
                    ),
                    affected_zones=[zone_id],
                    estimated_rset_reduction_s=45.0,
                    estimated_risk_reduction=12.0,
                    standard_reference="NFPA 557; BS 9999:2017 §7",
                    implementation_cost="MEDIUM",
                ))
        return recs

    # ─── Isolated Zones ───────────────────────────────────────────────────────

    def _check_isolated_zones(self) -> List[Recommendation]:
        recs = []
        if not self.graph.is_fully_connected():
            components = self.graph.get_components()
            for i, comp in enumerate(components):
                has_exit = any(
                    self.model.zones.get(z, type('', (), {'is_exit': False})).is_exit
                    for z in comp
                )
                if not has_exit:
                    recs.append(Recommendation(
                        rec_id=self._next_id(),
                        category="LAYOUT_CHANGE",
                        priority="CRITICAL",
                        title=f"Isolated zone group with no exit access",
                        description=(
                            f"Zone group {sorted(comp)} has no connection to any exit. "
                            f"Occupants in these zones cannot evacuate. "
                            f"Add openings connecting this group to the main evacuation route."
                        ),
                        affected_zones=sorted(comp),
                        estimated_rset_reduction_s=300.0,
                        estimated_risk_reduction=30.0,
                        standard_reference="BS 9999:2017 §13; NFPA 101 §7.5",
                        implementation_cost="HIGH",
                    ))
        return recs

    def _next_id(self) -> str:
        self._rec_counter += 1
        return f"REC{self._rec_counter:03d}"
