"""
FirePBD Engine — End-to-End Smoke Test
Constructs a synthetic 3-zone building, runs fire + evacuation simulation,
and prints key metrics. No blueprint file needed.
"""
import sys
sys.path.insert(0, 'e:\\FirePBD_Engine')
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.core.geometry import BuildingModel, Zone, Opening
from backend.agents.topology_agent import TopologyAgent
from backend.agents.fire_agent import FireSimulator, FireAnalyzer
from backend.agents.evacuation_agent import EvacuationSimulator
from backend.agents.risk_agent import RiskAnalyzer
from backend.agents.optimization_agent import EvacuationOptimizer
from backend.core.simulation_state import SimulationState

print("=" * 60)
print("FirePBD Engine — Smoke Test")
print("=" * 60)

# ── 1. Build synthetic model ─────────────────────────────────────────────────
model = BuildingModel(building_id="SMOKE_TEST", scale_m_per_px=0.05)

# Room 1: Office (10×8m = 80m²)
z1 = Zone("Z0001", [(0,0),(10,0),(10,8),(0,8)], label="zone", is_exit=False)
# Room 2: Corridor (10×3m = 30m²)
z2 = Zone("Z0002", [(0,8),(10,8),(10,11),(0,11)], label="zone", is_exit=False)
# Room 3: Exit lobby (10×5m = 50m²)
z3 = Zone("Z0003", [(0,11),(10,11),(10,16),(0,16)], label="zone", is_exit=True)

model.add_zone(z1)
model.add_zone(z2)
model.add_zone(z3)

o1 = Opening("O001", "Z0001", "Z0002", width=1.5, opening_type="door", midpoint=(5.0, 8.0))
o2 = Opening("O002", "Z0002", "Z0003", width=2.0, opening_type="double_door", is_exit_door=True, midpoint=(5.0, 11.0))
model.add_opening(o1)
model.add_opening(o2)

print(f"Model: {model}")
print(f"  Exits: {model.exit_zone_ids}")
print(f"  Total area: {model.total_area_m2:.0f}m²")
print(f"  Occupants: {model.total_occupants}")

# ── 2. Build topology ────────────────────────────────────────────────────────
topo = TopologyAgent(cell_size_m=1.0)   # coarser for speed in smoke test
graph, grid = topo.build(model, auto_repair=True)
print(f"\nGraph: {graph}")
print(f"Grid: {grid}")
print(f"  Wall cells: {grid.wall_mask.sum()} / {grid.wall_mask.size}")

# ── 3. Ignite ────────────────────────────────────────────────────────────────
fire_sim = FireSimulator(grid, dt_s=5.0)
ignited = fire_sim.ignite_zone("Z0001")
print(f"\nIgnition: {ignited} cells ignited in Z0001")

fire_analyzer = FireAnalyzer(grid, model.zones, dt_s=5.0)

# ── 4. Evacuation ────────────────────────────────────────────────────────────
import random; random.seed(42)
evac = EvacuationSimulator(graph, model, dt_s=5.0)
n_agents = evac.populate_from_model(seed=42)
print(f"Evacuation: {n_agents} agents placed")

# ── 5. Simulation loop ───────────────────────────────────────────────────────
aset_s = None
zone_status_history = []
for step in range(30):   # 30 steps × 5s = 150s
    fire_metrics = fire_sim.step()
    zone_status = fire_analyzer.analyze_zones()
    zone_status_history.append(zone_status)
    evac_metrics = evac.step(zone_status)

    if aset_s is None and fire_analyzer.compute_aset(zone_status):
        aset_s = step * 5.0
        print(f"  [t={aset_s}s] ASET triggered!")

    if evac_metrics.get("all_resolved"):
        print(f"  [step {step}] All agents resolved")
        break

    if step % 5 == 0:
        print(f"  [step {step}] Burning: {fire_metrics['burning_cells']}, "
              f"MaxT: {fire_metrics['max_temp_c']:.0f}°C, "
              f"Evac: {evac_metrics['evacuated']}/{evac_metrics['total_agents']}")

# ── 6. Results ───────────────────────────────────────────────────────────────
evac_summary = evac.summary()
print(f"\nEvacuation Summary:")
print(f"  RSET: {evac_summary.get('rset_s')}s")
print(f"  Evacuated: {evac_summary['evacuated']}/{evac_summary['total_agents']}")
print(f"  Dead: {evac_summary['dead']}")
print(f"  Incapacitated: {evac_summary['incapacitated']}")
print(f"  Avg FED: {evac_summary['avg_fed']}")

# ── 7. Risk Analysis ─────────────────────────────────────────────────────────
from backend.core.simulation_state import SimulationState
state = SimulationState(building_id="SMOKE_TEST", ignition_zone="Z0001")
state.aset_s = aset_s
state.rset_s = evac_summary.get("rset_s")
state.current_step = 30
state.sim_time_s = 150.0
state.complete()

risk = RiskAnalyzer.generate_report(state, evac_summary, zone_status_history, graph, model)
print(f"\nRisk Score: {risk['risk_score']}/100 ({risk['risk_level']})")
print(f"RSET/ASET Margin: {risk['rset_aset_margin_s']}s → {risk['margin_assessment']}")
print(f"Compliant: {risk['compliant']}")

# ── 8. Recommendations ───────────────────────────────────────────────────────
optimizer = EvacuationOptimizer(model, graph, risk, evac_summary)
recs = optimizer.generate()
print(f"\nOptimization: {len(recs)} recommendations generated")
for r in recs[:3]:
    print(f"  [{r.priority}] {r.title}")

print("\n" + "=" * 60)
print("SMOKE TEST PASSED ✓")
print("=" * 60)
