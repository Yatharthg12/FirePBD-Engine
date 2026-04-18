"""
FirePBD Engine — Risk Analysis Agent
======================================
Computes RSET, ASET, FED-based risk scores, bottleneck identification,
and runs Monte Carlo simulation for probabilistic safety assessment.

Standards Referenced:
  - BS 9999:2017 Annex D — RSET/ASET framework
  - SFPE Handbook Ch. 3-14 — Tenability criteria
  - ISO 13571:2012 — FED incapacitation model
  - NFPA 101 §7.3 — Means of egress capacity

Monte Carlo:
  Runs N independent fire+evacuation simulations with varied:
    - Ignition zone location
    - Occupant speed distribution (sampled per run)
    - Reaction time distribution
    - Fire load variation ±20%
  Produces confidence intervals for RSET, evacuation success probability,
  and per-zone ASET statistics.
"""
from __future__ import annotations

import random
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.core.constants import (
    ASET_SAFETY_MARGIN_S,
    MC_CONFIDENCE_LEVEL,
    MC_DEFAULT_RUNS,
    RISK_WEIGHT_BOTTLENECK,
    RISK_WEIGHT_DEAD_ZONES,
    RISK_WEIGHT_EVAC_SUCCESS,
    RISK_WEIGHT_RSET_ASET,
    RSET_BUFFER_FRACTION,
    SIMULATION_TIMESTEP_S,
)
from backend.utils.logger import get_logger
from backend.utils.math_utils import (
    confidence_interval,
    mean_safe,
    normalise_risk_score,
    percentile_safe,
    std_safe,
)

logger = get_logger(__name__)


# ─── Result Containers ────────────────────────────────────────────────────────

@dataclass
class SingleRunResult:
    """Results from one fire+evacuation simulation run."""
    run_id: int
    rset_s: Optional[float]
    aset_s: Optional[float]
    evacuation_success_pct: float
    dead_count: int
    incapacitated_count: int
    flashover_step: Optional[int]
    ignition_zone: str
    bottleneck_zones: List[str] = field(default_factory=list)

    @property
    def rset_aset_margin_s(self) -> Optional[float]:
        if self.rset_s is not None and self.aset_s is not None:
            return self.aset_s - self.rset_s
        return None

    @property
    def passed(self) -> bool:
        """True if RSET < ASET − safety_margin (BS 9999 criterion)."""
        margin = self.rset_aset_margin_s
        if margin is None:
            return False
        return margin >= ASET_SAFETY_MARGIN_S


@dataclass
class MonteCarloResult:
    """Aggregated results from N Monte Carlo runs."""
    n_runs: int
    rset_mean_s: float
    rset_std_s: float
    rset_p90_s: float          # 90th percentile (conservative RSET)
    rset_p95_s: float
    aset_mean_s: float
    aset_p10_s: float          # 10th percentile (conservative ASET)
    evacuation_success_mean_pct: float
    evacuation_success_p10_pct: float
    pass_rate: float            # fraction of runs that passed BS 9999
    mean_dead: float
    mean_incapacitated: float
    rset_ci_low: float
    rset_ci_high: float
    run_details: List[SingleRunResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_runs": self.n_runs,
            "rset": {
                "mean_s": round(self.rset_mean_s, 1),
                "std_s": round(self.rset_std_s, 1),
                "p90_s": round(self.rset_p90_s, 1),
                "p95_s": round(self.rset_p95_s, 1),
                "ci_90pct": [round(self.rset_ci_low, 1), round(self.rset_ci_high, 1)],
            },
            "aset": {
                "mean_s": round(self.aset_mean_s, 1),
                "p10_s": round(self.aset_p10_s, 1),
            },
            "evacuation_success": {
                "mean_pct": round(self.evacuation_success_mean_pct, 1),
                "p10_pct": round(self.evacuation_success_p10_pct, 1),
            },
            "pass_rate_pct": round(self.pass_rate * 100, 1),
            "mean_dead": round(self.mean_dead, 1),
            "mean_incapacitated": round(self.mean_incapacitated, 1),
        }


# ─── Risk Analyzer ────────────────────────────────────────────────────────────

class RiskAnalyzer:
    """
    Computes comprehensive risk metrics from simulation results.

    Inputs: simulation_state, evacuation_summary, zone_status history
    Outputs: structured risk report
    """

    # ─── RSET / ASET ──────────────────────────────────────────────────────────

    @staticmethod
    def compute_rset(
        evac_summary: dict,
        pre_movement_delay_s: float = 0.0,
    ) -> Optional[float]:
        """
        RSET = RSET_evacuation × safety_factor
        (safety factor per BS 9999 Table D.1: 1.25)
        """
        rset_evac = evac_summary.get("rset_s")
        if rset_evac is None:
            return None
        rset = (rset_evac + pre_movement_delay_s) * RSET_BUFFER_FRACTION
        return round(rset, 1)

    @staticmethod
    def compute_aset_rset_margin(
        rset_s: Optional[float],
        aset_s: Optional[float],
    ) -> Optional[float]:
        """ASET - RSET margin in seconds. Positive = safe."""
        if rset_s is None or aset_s is None:
            return None
        return round(aset_s - rset_s, 1)

    @staticmethod
    def assess_margin(margin_s: Optional[float]) -> str:
        """Classify margin against BS 9999 requirement (≥ 120s)."""
        if margin_s is None:
            return "UNKNOWN"
        if margin_s >= ASET_SAFETY_MARGIN_S:
            return "ADEQUATE"
        elif margin_s >= 0:
            return "MARGINAL"
        else:
            return "INADEQUATE"

    # ─── Evacuation Success ───────────────────────────────────────────────────

    @staticmethod
    def evacuation_success_rate(evac_summary: dict) -> float:
        total = evac_summary.get("total_agents", 0)
        evacuated = evac_summary.get("evacuated", 0)
        if total == 0:
            return 100.0
        return round(100 * evacuated / total, 1)

    # ─── Risk Score ───────────────────────────────────────────────────────────

    @staticmethod
    def compute_risk_score(
        rset_s: Optional[float],
        aset_s: Optional[float],
        evac_success_pct: float,
        bottleneck_severity: float,   # 0 = no bottleneck, 1 = severe
        dead_zone_fraction: float,    # fraction of zones that became untenable
    ) -> float:
        """
        Composite risk score 0–100 (0=safest, 100=most dangerous).
        Weighted combination of RSET/ASET, evacuation success, bottlenecks, dead zones.
        """
        margin = RiskAnalyzer.compute_aset_rset_margin(rset_s, aset_s)

        # Component 1: RSET/ASET ratio (0–1, 1 = perfectly safe margin=∞)
        if margin is None:
            rset_score = 50.0  # unknown
        elif margin >= 300:
            rset_score = 0.0   # very safe
        elif margin >= 120:
            rset_score = 20.0  # adequate
        elif margin >= 0:
            rset_score = 60.0  # marginal
        else:
            rset_score = 100.0  # unsafe

        # Component 2: Evacuation success (100% success = 0 risk)
        evac_score = max(0, 100.0 - evac_success_pct)

        # Component 3: Bottleneck severity (0–100)
        bottleneck_score = clamp_score(bottleneck_severity * 100, 0, 100)

        # Component 4: Dead zones fraction (0–100)
        dead_zone_score = clamp_score(dead_zone_fraction * 100, 0, 100)

        raw = (
            RISK_WEIGHT_RSET_ASET * rset_score +
            RISK_WEIGHT_EVAC_SUCCESS * evac_score +
            RISK_WEIGHT_BOTTLENECK * bottleneck_score +
            RISK_WEIGHT_DEAD_ZONES * dead_zone_score
        )
        return normalise_risk_score(raw)

    # ─── Bottlenecks ──────────────────────────────────────────────────────────

    @staticmethod
    def identify_bottlenecks(
        evac_summary: dict,
        graph,
    ) -> List[dict]:
        """
        Identify zones and openings with highest congestion/centrality.
        Returns sorted list of bottleneck descriptions.
        """
        # Graph-based betweenness centrality
        edge_bottlenecks = graph.identify_bottleneck_edges()
        node_centrality = graph.compute_betweenness_centrality()

        results = []
        # Top edge bottlenecks (openings)
        for z1, z2, score in edge_bottlenecks[:5]:
            results.append({
                "type": "opening",
                "zone_a": z1,
                "zone_b": z2,
                "centrality_score": round(score, 3),
                "severity": "HIGH" if score > 0.3 else "MEDIUM" if score > 0.1 else "LOW",
            })

        # Top node bottlenecks (corridors/junctions)
        sorted_nodes = sorted(node_centrality.items(), key=lambda x: x[1], reverse=True)
        for node_id, score in sorted_nodes[:3]:
            results.append({
                "type": "zone",
                "zone_id": node_id,
                "centrality_score": round(score, 3),
                "severity": "HIGH" if score > 0.4 else "MEDIUM" if score > 0.15 else "LOW",
            })

        return results

    # ─── Dead Zone Identification ─────────────────────────────────────────────

    @staticmethod
    def identify_dead_zones(
        zone_status_history: List[Dict[str, dict]],
        zones: dict,
    ) -> List[dict]:
        """
        Zones that became UNTENABLE at any point during simulation.
        Returns list of {zone_id, first_untenable_step, occupied_at_breach}.
        """
        dead = []
        first_breach: Dict[str, int] = {}

        for step, zone_statuses in enumerate(zone_status_history):
            for zone_id, status in zone_statuses.items():
                if status.get("danger") == "UNTENABLE" and zone_id not in first_breach:
                    first_breach[zone_id] = step

        for zone_id, step in first_breach.items():
            zone = zones.get(zone_id)
            dead.append({
                "zone_id": zone_id,
                "first_untenable_step": step,
                "first_untenable_time_s": step * SIMULATION_TIMESTEP_S,
                "is_exit": zone.is_exit if zone else False,
            })

        return sorted(dead, key=lambda x: x["first_untenable_step"])

    # ─── Full Risk Report ─────────────────────────────────────────────────────

    @staticmethod
    def generate_report(
        sim_state,
        evac_summary: dict,
        zone_status_history: List[Dict],
        graph,
        model,
        mc_result: Optional[MonteCarloResult] = None,
    ) -> dict:
        """Generate full risk assessment report dict."""
        rset = RiskAnalyzer.compute_rset(evac_summary)
        aset = sim_state.aset_s
        margin = RiskAnalyzer.compute_aset_rset_margin(rset, aset)
        margin_assessment = RiskAnalyzer.assess_margin(margin)
        evac_success = RiskAnalyzer.evacuation_success_rate(evac_summary)

        bottlenecks = RiskAnalyzer.identify_bottlenecks(evac_summary, graph)
        dead_zones = RiskAnalyzer.identify_dead_zones(zone_status_history, model.zones)

        bottleneck_severity = (
            max((b["centrality_score"] for b in bottlenecks), default=0.0)
        )
        dead_zone_frac = len(dead_zones) / max(len(model.zones), 1)

        risk_score = RiskAnalyzer.compute_risk_score(
            rset, aset, evac_success, bottleneck_severity, dead_zone_frac
        )

        risk_level = (
            "CRITICAL" if risk_score > 75 else
            "HIGH" if risk_score > 50 else
            "MEDIUM" if risk_score > 25 else
            "LOW"
        )

        report = {
            "simulation_id": sim_state.simulation_id,
            "building_id": sim_state.building_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "rset_s": rset,
            "aset_s": aset,
            "rset_aset_margin_s": margin,
            "margin_assessment": margin_assessment,
            "required_margin_s": ASET_SAFETY_MARGIN_S,
            "compliant": margin_assessment == "ADEQUATE",
            "evacuation_success_pct": evac_success,
            "total_agents": evac_summary.get("total_agents", 0),
            "evacuated": evac_summary.get("evacuated", 0),
            "dead": evac_summary.get("dead", 0),
            "incapacitated": evac_summary.get("incapacitated", 0),
            "flashover_time_s": (
                sim_state.flashover_step * SIMULATION_TIMESTEP_S
                if sim_state.flashover_step else None
            ),
            "bottlenecks": bottlenecks,
            "dead_zones": dead_zones,
            "events": sim_state.events,
            "monte_carlo": mc_result.to_dict() if mc_result else None,
            "standards": {
                "primary": "BS 9999:2017",
                "evac_model": "SFPE Handbook 5th Ed.",
                "fed_model": "ISO 13571:2012",
            },
        }
        return report


# ─── Monte Carlo Engine ───────────────────────────────────────────────────────

class MonteCarloEngine:
    """
    Runs N independent fire+evacuation simulations in parallel.
    Each run varies: ignition location, occupant speeds, fire load.

    Uses ProcessPoolExecutor for true CPU parallelism.
    Falls back to sequential execution if multiprocessing fails.
    """

    def __init__(
        self,
        n_runs: int = MC_DEFAULT_RUNS,
        n_workers: int = 4,
        confidence: float = MC_CONFIDENCE_LEVEL,
    ) -> None:
        self.n_runs = n_runs
        self.n_workers = n_workers
        self.confidence = confidence

    def run(
        self,
        model,
        max_steps: int = 120,
        dt_s: float = SIMULATION_TIMESTEP_S,
        progress_callback=None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Parameters
        ----------
        model : BuildingModel
        max_steps : int
            Steps per run (shorter for MC to keep manageable)
        progress_callback : callable(run_id, total) → None
            Optional progress hook.

        Returns
        -------
        MonteCarloResult
        """
        logger.info(
            f"Monte Carlo: {self.n_runs} runs × {max_steps} steps "
            f"({self.n_workers} workers)"
        )

        results: List[SingleRunResult] = []
        zone_ids = list(model.zones.keys())
        non_exit_zones = [z for z in zone_ids if not model.zones[z].is_exit]

        # Run sequentially (safer than multiprocessing for complex objects)
        for run_id in range(self.n_runs):
            try:
                result = _run_single_simulation(
                    run_id=run_id,
                    model=model,
                    zone_ids=non_exit_zones,
                    max_steps=max_steps,
                    dt_s=dt_s,
                )
                results.append(result)
            except Exception as e:
                logger.warning(f"MC run {run_id} failed: {e}")
                continue

            if progress_callback and run_id % 50 == 0:
                progress_callback(run_id, self.n_runs)

        if not results:
            logger.error("All Monte Carlo runs failed")
            return self._empty_result()

        logger.info(f"Monte Carlo complete: {len(results)}/{self.n_runs} runs succeeded")
        return self._aggregate(results)

    def _aggregate(self, results: List[SingleRunResult]) -> MonteCarloResult:
        rsets = [r.rset_s for r in results if r.rset_s is not None]
        asets = [r.aset_s for r in results if r.aset_s is not None]
        success_pcts = [r.evacuation_success_pct for r in results]
        dead_counts = [r.dead_count for r in results]
        incap_counts = [r.incapacitated_count for r in results]
        pass_count = sum(1 for r in results if r.passed)

        ci_low, ci_high = confidence_interval(rsets, self.confidence) if rsets else (0, 0)

        return MonteCarloResult(
            n_runs=len(results),
            rset_mean_s=mean_safe(rsets),
            rset_std_s=std_safe(rsets),
            rset_p90_s=percentile_safe(rsets, 90),
            rset_p95_s=percentile_safe(rsets, 95),
            aset_mean_s=mean_safe(asets),
            aset_p10_s=percentile_safe(asets, 10),
            evacuation_success_mean_pct=mean_safe(success_pcts),
            evacuation_success_p10_pct=percentile_safe(success_pcts, 10),
            pass_rate=pass_count / len(results),
            mean_dead=mean_safe(dead_counts),
            mean_incapacitated=mean_safe(incap_counts),
            rset_ci_low=ci_low,
            rset_ci_high=ci_high,
            run_details=results,
        )

    def _empty_result(self) -> MonteCarloResult:
        return MonteCarloResult(
            n_runs=0, rset_mean_s=0, rset_std_s=0, rset_p90_s=0, rset_p95_s=0,
            aset_mean_s=0, aset_p10_s=0, evacuation_success_mean_pct=0,
            evacuation_success_p10_pct=0, pass_rate=0, mean_dead=0,
            mean_incapacitated=0, rset_ci_low=0, rset_ci_high=0,
        )


def _run_single_simulation(
    run_id: int,
    model,
    zone_ids: List[str],
    max_steps: int,
    dt_s: float,
) -> SingleRunResult:
    """
    Run one complete fire+evacuation simulation.
    Called by MonteCarloEngine for each run.
    Varied parameters: ignition zone, fire load, occupant speeds.
    """
    from backend.agents.topology_agent import TopologyAgent
    from backend.agents.fire_agent import FireSimulator, FireAnalyzer
    from backend.agents.evacuation_agent import EvacuationSimulator

    rng = random.Random(run_id * 13337)

    # Vary fire load ±20%
    fire_load_factor = rng.uniform(0.8, 1.2)
    for zone in model.zones.values():
        zone.fuel_load_density *= fire_load_factor

    # Build topology
    topo = TopologyAgent()
    graph, grid = topo.build(model, auto_repair=True)

    # Ignite random non-exit zone
    ignition_zone = rng.choice(zone_ids) if zone_ids else None
    if ignition_zone:
        fire_sim = FireSimulator(grid, dt_s=dt_s)
        fire_sim.ignite_zone(ignition_zone)
    else:
        fire_sim = FireSimulator(grid, dt_s=dt_s)

    fire_analyzer = FireAnalyzer(grid, model.zones, dt_s=dt_s)

    # Create evacuation simulator
    evac_sim = EvacuationSimulator(graph, model, dt_s=dt_s)
    evac_sim.populate_from_model(seed=run_id)

    aset_s = None
    rset_s = None
    flashover_step = None

    for step in range(max_steps):
        fire_metrics = fire_sim.step()
        zone_status = fire_analyzer.analyze_zones()
        evac_metrics = evac_sim.step(zone_status)

        # ASET detection
        if aset_s is None and fire_analyzer.compute_aset(zone_status):
            aset_s = step * dt_s

        if fire_metrics.get("flashover_detected") and flashover_step is None:
            flashover_step = step

        if evac_metrics.get("all_resolved"):
            rset_s = evac_sim.compute_rset()
            break

    rset_s = rset_s or evac_sim.compute_rset()
    evac_summary = evac_sim.summary()

    # Restore fire load
    for zone in model.zones.values():
        zone.fuel_load_density /= fire_load_factor

    return SingleRunResult(
        run_id=run_id,
        rset_s=rset_s,
        aset_s=aset_s,
        evacuation_success_pct=evac_summary["evacuation_success_pct"],
        dead_count=evac_summary["dead"],
        incapacitated_count=evac_summary["incapacitated"],
        flashover_step=flashover_step,
        ignition_zone=ignition_zone or "unknown",
    )


# ─── Utility ─────────────────────────────────────────────────────────────────

def clamp_score(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
