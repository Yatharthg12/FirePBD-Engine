from backend.core.geometry import BuildingModel, Zone
from backend.core.graph_model import SpatialGraph
from backend.utils.validation import BuildingModelValidator, GraphValidator


def test_valid_model_passes_validation(simple_topology):
    model, graph, _ = simple_topology
    report = BuildingModelValidator.validate(model, graph)
    assert report.is_valid()


def test_disconnected_graph_fails_validation():
    model = BuildingModel(building_id="DISCONNECTED")
    graph = SpatialGraph()

    for zone in (
        Zone("Z1", [(0, 0), (2, 0), (2, 2), (0, 2)]),
        Zone("Z2", [(5, 0), (7, 0), (7, 2), (5, 2)]),
    ):
        model.add_zone(zone)
        graph.add_zone(zone)

    report = GraphValidator.validate_graph(graph)
    assert report.is_valid() is False
    assert report.errors
