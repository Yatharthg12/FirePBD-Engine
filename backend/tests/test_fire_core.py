import numpy as np

from backend.agents.fire_agent import FireAnalyzer, FireSimulator
from backend.core.constants import CELL_BURNING
from backend.core.grid_model import Grid


def test_fire_ignites_zone_and_generates_heat(simple_topology):
    model, _, grid = simple_topology
    simulator = FireSimulator(grid, dt_s=5.0)

    ignited = simulator.ignite_zone("Z0001")
    metrics = simulator.step()

    assert ignited > 0
    assert metrics["burning_cells"] >= ignited
    assert metrics["max_temp_c"] > 20.0

    analyzer = FireAnalyzer(grid, model.zones, dt_s=5.0)
    zone_status = analyzer.analyze_zones()
    assert "Z0001" in zone_status
    assert zone_status["Z0001"]["avg_temp_c"] > 20.0


def test_fire_does_not_jump_a_solid_wall():
    np.random.seed(0)
    grid = Grid(width_m=3.0, height_m=1.0, cell_size_m=1.0)
    grid.ignite_cell(0, 0)
    grid.set_wall(0, 1)

    simulator = FireSimulator(grid, dt_s=5.0, base_spread_prob=1.0)
    for _ in range(3):
        simulator.step()

    assert grid.state[0, 2] != CELL_BURNING


def test_fire_analyzer_triggers_aset_when_zone_is_untenable(simple_topology):
    model, _, grid = simple_topology
    simulator = FireSimulator(grid, dt_s=5.0)
    simulator.ignite_zone("Z0001")
    simulator.step()

    analyzer = FireAnalyzer(grid, model.zones, dt_s=5.0)
    zone_status = analyzer.analyze_zones()

    assert zone_status["Z0001"]["tenable"] is False
    assert analyzer.compute_aset(zone_status) is True
