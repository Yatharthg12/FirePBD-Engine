"""
FirePBD Engine — Simulation State Container
============================================
Central state object for one simulation run.
Tracks timestep, manages snapshot history, and provides export utilities.
One SimulationState is created per simulation run and passed between all agents.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.core.constants import SIMULATION_TIMESTEP_S


@dataclass
class PersonSnapshot:
    """Lightweight per-step snapshot of one agent."""
    person_id: str
    zone_id: str
    x: float
    y: float
    status: str         # REACTING | MOVING | EVACUATED | INCAPACITATED | DEAD
    fed_accumulated: float
    time_taken_s: float


@dataclass
class StepSnapshot:
    """Complete state snapshot for one simulation timestep — used for playback and streaming."""
    step: int
    sim_time_s: float
    fire_snapshot: Dict[str, Any]           # compact grid snapshot
    zone_status: Dict[str, Dict]            # per-zone: temp, smoke, danger, fed
    persons: List[PersonSnapshot] = field(default_factory=list)
    events: List[str] = field(default_factory=list)  # e.g. "Flashover in Z003"

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "sim_time_s": self.sim_time_s,
            "fire": self.fire_snapshot,
            "zones": self.zone_status,
            "persons": [
                {
                    "id": p.person_id,
                    "zone": p.zone_id,
                    "x": round(p.x, 2),
                    "y": round(p.y, 2),
                    "status": p.status,
                    "fed": round(p.fed_accumulated, 3),
                    "time_s": round(p.time_taken_s, 1),
                }
                for p in self.persons
            ],
            "events": self.events,
        }


class SimulationState:
    """
    Master state container for a FirePBD simulation run.

    Lifecycle:
        created → running → converged / timeout → complete
    """

    STATUS_CREATED = "created"
    STATUS_RUNNING = "running"
    STATUS_COMPLETE = "complete"
    STATUS_ERROR = "error"

    def __init__(
        self,
        simulation_id: Optional[str] = None,
        building_id: str = "unknown",
        ignition_zone: str = "unknown",
        total_steps: int = 240,
        dt_s: float = SIMULATION_TIMESTEP_S,
    ) -> None:
        self.simulation_id: str = simulation_id or str(uuid.uuid4())[:12]
        self.building_id: str = building_id
        self.ignition_zone: str = ignition_zone

        self.total_steps: int = total_steps
        self.dt_s: float = dt_s
        self.current_step: int = 0
        self.sim_time_s: float = 0.0

        self.status: str = self.STATUS_CREATED
        self.wall_time_start: float = time.time()
        self.wall_time_end: Optional[float] = None

        # History (one entry per step)
        self.snapshots: List[StepSnapshot] = []

        # Key milestones
        self.aset_s: Optional[float] = None          # Available Safe Egress Time
        self.rset_s: Optional[float] = None          # Required Safe Egress Time
        self.flashover_step: Optional[int] = None
        self.evacuation_complete_step: Optional[int] = None

        # Events log (human-readable)
        self.events: List[str] = []

        # Final results (populated after simulation)
        self.results: Dict[str, Any] = {}

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self.status = self.STATUS_RUNNING
        self.wall_time_start = time.time()

    def complete(self) -> None:
        self.status = self.STATUS_COMPLETE
        self.wall_time_end = time.time()

    def error(self, msg: str) -> None:
        self.status = self.STATUS_ERROR
        self.events.append(f"ERROR: {msg}")

    # ─── Time Advance ─────────────────────────────────────────────────────────

    def advance(self) -> None:
        """Increment simulation clock by one timestep."""
        self.current_step += 1
        self.sim_time_s = self.current_step * self.dt_s

    @property
    def is_running(self) -> bool:
        return self.status == self.STATUS_RUNNING and self.current_step < self.total_steps

    @property
    def elapsed_wall_time_s(self) -> float:
        end = self.wall_time_end or time.time()
        return end - self.wall_time_start

    # ─── Snapshots ────────────────────────────────────────────────────────────

    def record_snapshot(self, snapshot: StepSnapshot) -> None:
        self.snapshots.append(snapshot)

    def latest_snapshot(self) -> Optional[StepSnapshot]:
        return self.snapshots[-1] if self.snapshots else None

    # ─── Events ───────────────────────────────────────────────────────────────

    def log_event(self, message: str) -> None:
        ts = f"[{self.sim_time_s:.0f}s]"
        self.events.append(f"{ts} {message}")

    def log_flashover(self, zone_id: str) -> None:
        if self.flashover_step is None:
            self.flashover_step = self.current_step
        self.log_event(f"FLASHOVER detected in zone {zone_id}")

    def log_aset(self) -> None:
        self.aset_s = self.sim_time_s
        self.log_event(f"ASET reached — tenability threshold breached at {self.aset_s:.0f}s")

    def log_rset(self) -> None:
        self.rset_s = self.sim_time_s
        self.evacuation_complete_step = self.current_step
        self.log_event(f"RSET — last occupant evacuated at {self.rset_s:.0f}s")

    # ─── Export ───────────────────────────────────────────────────────────────

    def summary_dict(self) -> dict:
        margin_s = None
        if self.aset_s is not None and self.rset_s is not None:
            margin_s = round(self.aset_s - self.rset_s, 1)

        return {
            "simulation_id": self.simulation_id,
            "building_id": self.building_id,
            "status": self.status,
            "total_steps_run": self.current_step,
            "sim_time_s": round(self.sim_time_s, 1),
            "wall_time_s": round(self.elapsed_wall_time_s, 2),
            "aset_s": self.aset_s,
            "rset_s": self.rset_s,
            "aset_rset_margin_s": margin_s,
            "flashover_step": self.flashover_step,
            "flashover_time_s": (
                self.flashover_step * self.dt_s if self.flashover_step else None
            ),
            "events": self.events,
            "results": self.results,
        }

    def to_json(self) -> str:
        return json.dumps(self.summary_dict(), indent=2, default=str)

    def __repr__(self) -> str:
        return (
            f"SimulationState(id={self.simulation_id!r}, "
            f"step={self.current_step}/{self.total_steps}, "
            f"t={self.sim_time_s:.0f}s, status={self.status!r})"
        )
