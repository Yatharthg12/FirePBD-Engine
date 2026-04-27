from backend.agents.blueprint_agent import BlueprintAgent


def test_svg_blueprint_agent_extracts_real_sample(sample_svg_path):
    model = BlueprintAgent().process(str(sample_svg_path))

    assert model.building_id == "10000"
    assert len(model.zones) >= 3
    assert len(model.openings) >= 1
    assert len(model.walls) >= 10
    assert model.exit_zone_ids
