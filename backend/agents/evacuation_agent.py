"""
FirePBD Engine — Multi-Agent Evacuation Simulator
===================================================
Simulates realistic human evacuation under dynamic fire hazard.

Evacuation Model (SFPE Handbook, 5th Ed.):
  ─ Speed model: v = v_free × f_smoke × f_density
    where:
      v_free  ~ N(1.2, 0.25) m/s (Fruin 1971, SFPE Table 3-13.1)
      f_smoke = speed fraction from visibility curve (SFPE 3-13.10)
      f_density = max(0, 1 − 0.266 × D) where D = persons/m² (Fruin)

  ─ Pre-movement delay: T_react ~ N(45s, 15s) (BS 9999 Annex D)

  ─ FED accumulation per timestep (ISO 13571):
      FED = FED_CO + FED_heat + FED_O2
      Incapacitation when FED ≥ 1.0

  ─ Dynamic hazard rerouting:
      Path recomputed every N steps using hazard-weighted Dijkstra.
      Zones with danger="UNTENABLE" are avoided if alternate path exists.

  ─ Door congestion model (Nelson & MacLennan, SFPE 3-13.19):
      Flow = 1.3 × W (persons/s per door)
      Queue buildup at narrow openings; blocked agents wait.

  ─ RSET = max(time_taken for all evacuated agents) + pre-movement delays
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from backend.core.constants import (
    DOOR_FLOW_RATE_P_M_S,
    FED_INCAPACITATION_THRESHOLD,
    FRUIN_SPEED_K,
    HEAT_FED_TABLE,
    MAX_CROWD_DENSITY_P_M2,
    PERSON_BODY_RADIUS_M,
    SIMULATION_TIMESTEP_S,
    T_DETECTION_S,
    T_REACTION_MEAN_S,
    T_REACTION_STD_S,
    T_WARNING_S,
    TENABILITY_CO_MAX_PPM,
    TENABILITY_TEMP_MAX_C,
    TENABILITY_VISIBILITY_MIN_M,
    VISIBILITY_SPEED_CURVE,
    WALK_SPEED_MAX_M_S,
    WALK_SPEED_MEAN_M_S,
    WALK_SPEED_MIN_M_S,
    WALK_SPEED_STD_M_S,
)
from backend.core.geometry import BuildingModel
from backend.core.graph_model import SpatialGraph
from backend.utils.logger import get_logger
from backend.utils.math_utils import (
    fed_increment_co,
    fed_increment_heat,
    fed_increment_o2,
    interpolate_table,
    clamp,
)

logger = get_logger(__name__)

# How often (steps) each agent recomputes their evacuation path
REROUTE_INTERVAL_STEPS = 10


# ─── Agent Status ─────────────────────────────────────────────────────────────

class AgentStatus(str, Enum):
    REACTING = "REACTING"           # pre-movement delay phase
    MOVING = "MOVING"               # actively moving toward exit
    WAITING = "WAITING"             # blocked at congested door
    EVACUATED = "EVACUATED"         # reached exit zone
    INCAPACITATED = "INCAPACITATED" # FED ≥ 1.0 — cannot move
    DEAD = "DEAD"                   # extreme FED or lethal conditions


# ─── Person Agent ─────────────────────────────────────────────────────────────

@dataclass
class Person:
    """
    Individual evacuee agent.

    All physical quantities in SI units (metres, seconds).
    """
    id: str
    current_zone: str
    start_zone: str = field(init=False)

    # Physical attributes (sampled from distributions at creation)
    speed_base_m_s: float = field(default_factory=lambda: clamp(
        random.gauss(WALK_SPEED_MEAN_M_S, WALK_SPEED_STD_M_S),
        WALK_SPEED_MIN_M_S,
        WALK_SPEED_MAX_M_S,
    ))
    mobility_factor: float = 1.0        # 1.0 = fully mobile; 0.5 = impaired
    body_radius_m: float = PERSON_BODY_RADIUS_M

    # State
    status: AgentStatus = AgentStatus.REACTING
    path: List[str] = field(default_factory=list)
    time_taken_s: float = 0.0
    fed_accumulated: float = 0.0
    pre_move_delay_s: float = field(default_factory=lambda: max(
        0.0,
        random.gauss(T_REACTION_MEAN_S, T_REACTION_STD_S)
        + T_DETECTION_S + T_WARNING_S
    ))

    # Metrics
    speed_history: List[float] = field(default_factory=list, repr=False)
    reroute_count: int = 0
    steps_waited: int = 0

    def __post_init__(self) -> None:
        self.start_zone = self.current_zone

    @property
    def is_active(self) -> bool:
        return self.status in (AgentStatus.REACTING, AgentStatus.MOVING, AgentStatus.WAITING)

    @property
    def evacuated(self) -> bool:
        return self.status == AgentStatus.EVACUATED

    @property
    def alive(self) -> bool:
        return self.status != AgentStatus.DEAD

    @property
    def effective_speed(self) -> float:
        return self.speed_base_m_s * self.mobility_factor

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start_zone": self.start_zone,
            "current_zone": self.current_zone,
            "status": self.status.value,
            "time_taken_s": round(self.time_taken_s, 1),
            "fed_accumulated": round(self.fed_accumulated, 3),
            "speed_base_m_s": round(self.speed_base_m_s, 2),
            "reroute_count": self.reroute_count,
        }


# ─── Door Flow Controller ─────────────────────────────────────────────────────

class DoorFlowController:
    """
    Manages queuing and flow rate at each opening (door).
    Prevents unrealistic simultaneous passage of many agents.

    Capacity = DOOR_FLOW_RATE_P_M_S × door_width × dt_s persons per step.
    """

    def __init__(self, openings: dict, dt_s: float) -> None:
        # opening_id → max_persons_per_step
        self.capacity: Dict[str, float] = {}
        self.used: Dict[str, float] = {}

        for oid, opening in openings.items():
            cap = DOOR_FLOW_RATE_P_M_S * opening.clear_width * dt_s
            self.capacity[oid] = max(1.0, cap)

    def reset_step(self) -> None:
        self.used = {k: 0.0 for k in self.capacity}

    def try_pass(self, opening_id: Optional[str]) -> bool:
        """Return True if an agent can pass through this opening this step."""
        if opening_id is None:
            return True  # no specific opening — free movement
        cap = self.capacity.get(opening_id, 1.0)
        used = self.used.get(opening_id, 0.0)
        if used < cap:
            self.used[opening_id] = used + 1.0
            return True
        return False


# ─── Evacuation Simulator ─────────────────────────────────────────────────────

class EvacuationSimulator:
    """
    Multi-agent evacuation simulator.

    Parameters
    ----------
    graph : SpatialGraph
    model : BuildingModel
    dt_s : float
        Simulation timestep in seconds
    """

    def __init__(
        self,
        graph: SpatialGraph,
        model: BuildingModel,
        dt_s: float = SIMULATION_TIMESTEP_S,
    ) -> None:
        self.graph = graph
        self.model = model
        self.dt_s = dt_s
        self.people: List[Person] = []
        self.exit_zone_ids = model.exit_zone_ids

        # Door flow controller (populated once people are added)
        self._door_flow = DoorFlowController(model.openings, dt_s)

        # Zone-level state (populated each step from FireAnalyzer)
        self.zone_status: Dict[str, dict] = {}
        self.zone_hazard_weights: Dict[str, float] = {}
        self._path_cache: Dict[tuple, List[str]] = {}

        # Crowd density tracker: zone_id → current occupancy count
        self._zone_occupancy: Dict[str, int] = {}

        # Step counter
        self._step = 0

        logger.info(
            f"EvacuationSimulator init: graph={graph}, "
            f"exits={self.exit_zone_ids}"
        )

    def populate_from_model(self, seed: Optional[int] = None) -> int:
        """
        Auto-populate agents based on zone occupancy loads.
        Returns total agent count.
        """
        if seed is not None:
            random.seed(seed)

        count = 0
        occupied_zones = [zone for zone in self.model.zones.values() if not zone.is_exit]
        if not occupied_zones:
            occupied_zones = list(self.model.zones.values())

        for zone in occupied_zones:
            n = zone.max_occupants
            for i in range(n):
                p = Person(
                    id=f"P{count:04d}",
                    current_zone=zone.id,
                )
                self.add_person(p)
                count += 1

        logger.info(f"Populated {count} agents from building occupancy model")
        return count

    def add_person(self, person: Person) -> None:
        self.people.append(person)

    # ─── Main Step ────────────────────────────────────────────────────────────

    def step(self, zone_status: Optional[Dict[str, dict]] = None) -> dict:
        """
        Advance evacuation simulation by one timestep.

        Parameters
        ----------
        zone_status : dict (optional)
            Zone hazard status from FireAnalyzer.analyze_zones().
            If None, no hazard rerouting occurs.

        Returns
        -------
        dict of step metrics
        """
        self._step += 1

        if zone_status:
            self.zone_status = zone_status
            self.zone_hazard_weights = self._compute_hazard_weights(zone_status)
            self._path_cache.clear()

        self._door_flow.reset_step()
        self._update_zone_occupancy()

        evacuated_this_step = 0
        incapacitated_this_step = 0

        for person in self.people:
            if not person.is_active:
                continue

            person.time_taken_s += self.dt_s

            # ── Pre-movement delay ─────────────────────────────────────────────
            if person.status == AgentStatus.REACTING:
                person.pre_move_delay_s -= self.dt_s
                if person.pre_move_delay_s <= 0:
                    person.status = AgentStatus.MOVING
                    person.pre_move_delay_s = 0
                    # Compute initial path
                    self._compute_path(person)
                continue

            # ── Already evacuated (shouldn't be active) ──────────────────────
            if person.evacuated:
                continue

            # ── FED accumulation ──────────────────────────────────────────────
            self._accumulate_fed(person)

            if person.fed_accumulated >= FED_INCAPACITATION_THRESHOLD:
                person.status = AgentStatus.INCAPACITATED
                incapacitated_this_step += 1
                logger.debug(f"Agent {person.id} incapacitated (FED={person.fed_accumulated:.2f})")
                continue

            # Check lethal conditions (immediate death)
            z_status = self.zone_status.get(person.current_zone, {})
            if (
                z_status.get("avg_temp_c", 0) > 130 or
                z_status.get("avg_co_ppm", 0) > 5000 or
                person.fed_accumulated > 3.0
            ):
                person.status = AgentStatus.DEAD
                logger.debug(f"Agent {person.id} died in zone {person.current_zone}")
                continue

            # ── Path planning / rerouting ──────────────────────────────────────
            if not person.path or self._step % REROUTE_INTERVAL_STEPS == 0:
                self._compute_path(person)

            if not person.path:
                # No path — wait
                person.status = AgentStatus.WAITING
                person.steps_waited += 1
                continue

            # ── Movement ──────────────────────────────────────────────────────
            person.status = AgentStatus.MOVING
            self._move_person(person)

            # ── Evacuation check ──────────────────────────────────────────────
            if person.current_zone in self.exit_zone_ids:
                person.status = AgentStatus.EVACUATED
                evacuated_this_step += 1
                logger.debug(
                    f"Agent {person.id} evacuated at t={person.time_taken_s:.0f}s "
                    f"(reroutes={person.reroute_count})"
                )

        return self._step_metrics(evacuated_this_step, incapacitated_this_step)

    # ─── Path Computation ─────────────────────────────────────────────────────

    def _compute_path(self, person: Person) -> None:
        """Compute or recompute evacuation path for one agent."""
        if person.current_zone in self.exit_zone_ids:
            person.status = AgentStatus.EVACUATED
            return

        # Identify zones too dangerous to traverse
        blocked = [
            zid for zid, s in self.zone_status.items()
            if s.get("danger") == "UNTENABLE"
        ]

        cache_key = (
            person.current_zone,
            tuple(sorted(blocked)),
            tuple(sorted((z, round(w, 1)) for z, w in self.zone_hazard_weights.items())),
            tuple(self.exit_zone_ids),
        )
        cached = self._path_cache.get(cache_key)
        if cached is not None:
            person.path = list(cached)
            return

        # Try hazard-weighted routing first
        if self.zone_hazard_weights:
            path = self.graph.shortest_path_hazard_weighted(
                person.current_zone,
                self.exit_zone_ids,
                self.zone_hazard_weights,
            )
        else:
            path = self.graph.shortest_path_to_any_exit(
                person.current_zone,
                self.exit_zone_ids,
                blocked_zones=blocked,
            )

        # Fallback: ignore hazards
        if path is None:
            path = self.graph.shortest_path_to_any_exit(
                person.current_zone,
                self.exit_zone_ids,
                blocked_zones=None,
            )

        if path and path != person.path:
            person.reroute_count += 1

        person.path = path or []
        self._path_cache[cache_key] = list(person.path)

    # ─── Movement ─────────────────────────────────────────────────────────────

    def _move_person(self, person: Person) -> None:
        """Advance person one zone along their path (subject to speed + flow)."""
        if len(person.path) <= 1:
            return

        next_zone_id = person.path[1]

        # Speed calculation
        z_status = self.zone_status.get(person.current_zone, {})
        visibility = z_status.get("min_visibility_m", 10.0)
        f_smoke = interpolate_table(visibility, VISIBILITY_SPEED_CURVE)
        density = self._zone_density(next_zone_id)
        f_density = max(0.0, 1.0 - FRUIN_SPEED_K * density)
        actual_speed = person.effective_speed * f_smoke * f_density
        actual_speed = max(actual_speed, 0.05)  # minimum shuffle speed

        person.speed_history.append(actual_speed)

        # Door flow check
        opening_data = self.graph.get_opening_between(
            person.current_zone, next_zone_id
        )
        opening_id = opening_data.get("opening_id") if opening_data else None
        can_pass = self._door_flow.try_pass(opening_id)

        if not can_pass:
            person.status = AgentStatus.WAITING
            person.steps_waited += 1
            return

        # Time to traverse depends on zone centroid distance / speed
        zone_curr = self.model.zones.get(person.current_zone)
        zone_next = self.model.zones.get(next_zone_id)

        if zone_curr and zone_next:
            dist = zone_curr.centroid.distance(zone_next.centroid)
            traverse_time = dist / max(actual_speed, 0.05)
            if traverse_time > self.dt_s:
                # Agent hasn't had enough time to cross this step — stay
                # (simplified: move every step regardless for graph-level sim)
                pass  # for graph-level, one step = one zone move

        person.current_zone = next_zone_id
        person.path.pop(0)

    # ─── FED Accumulation ─────────────────────────────────────────────────────

    def _accumulate_fed(self, person: Person) -> None:
        """Add FED increment from current zone hazard."""
        z_status = self.zone_status.get(person.current_zone, {})
        if not z_status:
            return

        fed_co = fed_increment_co(
            z_status.get("avg_co_ppm", 0.0), self.dt_s
        )
        fed_heat = fed_increment_heat(
            z_status.get("avg_temp_c", 20.0), self.dt_s, HEAT_FED_TABLE
        )
        fed_o2 = fed_increment_o2(
            z_status.get("avg_oxygen_pct", 20.9), self.dt_s
        )
        person.fed_accumulated += fed_co + fed_heat + fed_o2

    # ─── Zone Utilities ───────────────────────────────────────────────────────

    def _update_zone_occupancy(self) -> None:
        self._zone_occupancy = {}
        for person in self.people:
            if person.is_active:
                zid = person.current_zone
                self._zone_occupancy[zid] = self._zone_occupancy.get(zid, 0) + 1

    def _zone_density(self, zone_id: str) -> float:
        """Persons per m² in a zone (for speed reduction model)."""
        count = self._zone_occupancy.get(zone_id, 0)
        zone = self.model.zones.get(zone_id)
        if zone and zone.area > 0:
            return count / zone.area
        return 0.0

    def _compute_hazard_weights(
        self, zone_status: Dict[str, dict]
    ) -> Dict[str, float]:
        scale = {"LOW": 1.0, "MEDIUM": 2.5, "HIGH": 7.0, "UNTENABLE": 40.0}
        return {
            zid: scale.get(s.get("danger", "LOW"), 1.0)
            for zid, s in zone_status.items()
        }

    # ─── Metrics ──────────────────────────────────────────────────────────────

    def _step_metrics(
        self,
        evacuated_this_step: int,
        incapacitated_this_step: int,
    ) -> dict:
        total = len(self.people)
        evacuated = sum(1 for p in self.people if p.evacuated)
        incapacitated = sum(1 for p in self.people if p.status == AgentStatus.INCAPACITATED)
        dead = sum(1 for p in self.people if p.status == AgentStatus.DEAD)
        moving = sum(1 for p in self.people if p.status == AgentStatus.MOVING)
        waiting = sum(1 for p in self.people if p.status == AgentStatus.WAITING)
        reacting = sum(1 for p in self.people if p.status == AgentStatus.REACTING)

        return {
            "step": self._step,
            "total_agents": total,
            "evacuated": evacuated,
            "moving": moving,
            "waiting": waiting,
            "reacting": reacting,
            "incapacitated": incapacitated,
            "dead": dead,
            "evacuation_pct": round(100 * evacuated / max(total, 1), 1),
            "evacuated_this_step": evacuated_this_step,
            "all_resolved": (evacuated + incapacitated + dead) >= total,
        }

    def compute_rset(self) -> Optional[float]:
        """
        Compute RSET = max evacuation time across all successfully evacuated agents.
        Returns None if no one has evacuated.
        """
        times = [p.time_taken_s for p in self.people if p.evacuated]
        return max(times) if times else None

    def summary(self) -> dict:
        total = len(self.people)
        evacuated = [p for p in self.people if p.evacuated]
        incapacitated = [p for p in self.people if p.status == AgentStatus.INCAPACITATED]
        dead = [p for p in self.people if p.status == AgentStatus.DEAD]

        evac_times = [p.time_taken_s for p in evacuated]
        rset = max(evac_times) if evac_times else None

        avg_fed = np.mean([p.fed_accumulated for p in self.people]) if self.people else 0.0
        max_reroutes = max((p.reroute_count for p in self.people), default=0)

        return {
            "total_agents": total,
            "evacuated": len(evacuated),
            "incapacitated": len(incapacitated),
            "dead": len(dead),
            "evacuation_success_pct": round(100 * len(evacuated) / max(total, 1), 1),
            "rset_s": round(rset, 1) if rset else None,
            "mean_evac_time_s": round(float(np.mean(evac_times)), 1) if evac_times else None,
            "max_reroutes": max_reroutes,
            "avg_fed": round(float(avg_fed), 3),
            "persons": [p.to_dict() for p in self.people],
        }
