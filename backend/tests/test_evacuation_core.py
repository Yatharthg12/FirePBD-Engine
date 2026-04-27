from backend.agents.evacuation_agent import AgentStatus, EvacuationSimulator, Person


def _safe_zone_status(model):
    return {
        zone_id: {
            "avg_temp_c": 20.0,
            "avg_smoke": 0.0,
            "avg_co_ppm": 0.0,
            "min_visibility_m": 10.0,
            "avg_oxygen_pct": 20.9,
            "danger": "LOW",
            "tenable": True,
        }
        for zone_id in model.zones
    }


def test_agents_can_evacuate_under_tenable_conditions(simple_topology):
    model, graph, _ = simple_topology
    simulator = EvacuationSimulator(graph, model, dt_s=5.0)
    simulator.populate_from_model(seed=42)

    for person in simulator.people:
        person.pre_move_delay_s = 0.0

    zone_status = _safe_zone_status(model)
    for _ in range(5):
        metrics = simulator.step(zone_status)
        if metrics["all_resolved"]:
            break

    summary = simulator.summary()
    assert summary["evacuated"] == summary["total_agents"]
    assert summary["dead"] == 0
    assert summary["rset_s"] is not None


def test_person_incapacitation_under_severe_conditions(simple_topology):
    model, graph, _ = simple_topology
    simulator = EvacuationSimulator(graph, model, dt_s=5.0)
    person = Person(id="P0001", current_zone="Z0001")
    person.pre_move_delay_s = 0.0
    simulator.add_person(person)

    severe_status = _safe_zone_status(model)
    severe_status["Z0001"].update(
        {
            "avg_temp_c": 200.0,
            "avg_co_ppm": 50000.0,
            "avg_oxygen_pct": 8.0,
            "min_visibility_m": 1.0,
            "danger": "UNTENABLE",
            "tenable": False,
        }
    )

    simulator.step(severe_status)
    simulator.step(severe_status)

    assert person.status in {AgentStatus.INCAPACITATED, AgentStatus.DEAD}
