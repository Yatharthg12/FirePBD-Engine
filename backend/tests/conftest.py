from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.agents.topology_agent import TopologyAgent
from backend.core.geometry import BuildingModel, Opening, Zone
from backend.main import _blueprints, _simulations, app

collect_ignore_glob = [
    "test_geometry.py",
    "test_graph.py",
    "test_fire.py",
    "test_evacuation.py",
    "test_validation.py",
]


@pytest.fixture(autouse=True)
def reset_api_state():
    _blueprints.clear()
    _simulations.clear()
    yield
    _blueprints.clear()
    _simulations.clear()


@pytest.fixture
def sample_svg_path() -> Path:
    path = Path(
        "datasets/CubiCasa5K/cubicasa5k/high_quality_architectural/10000/model.svg"
    )
    assert path.exists()
    return path


@pytest.fixture
def simple_building_model() -> BuildingModel:
    model = BuildingModel(building_id="TEST_BUILDING", scale_m_per_px=0.05)

    z1 = Zone("Z0001", [(0, 0), (10, 0), (10, 8), (0, 8)], label="zone")
    z2 = Zone("Z0002", [(0, 8), (10, 8), (10, 11), (0, 11)], label="corridor")
    z3 = Zone(
        "Z0003",
        [(0, 11), (10, 11), (10, 16), (0, 16)],
        label="outdoor",
        is_exit=True,
    )

    model.add_zone(z1)
    model.add_zone(z2)
    model.add_zone(z3)

    model.add_opening(
        Opening(
            "O001",
            "Z0001",
            "Z0002",
            width=1.5,
            opening_type="door",
            midpoint=(5.0, 8.0),
        )
    )
    model.add_opening(
        Opening(
            "O002",
            "Z0002",
            "Z0003",
            width=2.0,
            opening_type="double_door",
            is_exit_door=True,
            midpoint=(5.0, 11.0),
        )
    )
    return model


@pytest.fixture
def simple_topology(simple_building_model):
    graph, grid = TopologyAgent(cell_size_m=1.0).build(
        simple_building_model,
        auto_repair=True,
    )
    return simple_building_model, graph, grid


@pytest.fixture
def api_client():
    with TestClient(app) as client:
        yield client
