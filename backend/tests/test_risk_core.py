from backend.agents.risk_agent import MonteCarloEngine, RiskAnalyzer
from backend.core.simulation_state import SimulationState


def test_margin_assessment_levels():
    assert RiskAnalyzer.assess_margin(150.0) == "ADEQUATE"
    assert RiskAnalyzer.assess_margin(10.0) == "MARGINAL"
    assert RiskAnalyzer.assess_margin(-5.0) == "INADEQUATE"
    assert RiskAnalyzer.assess_margin(None) == "UNKNOWN"


def test_generate_risk_report(simple_topology):
    model, graph, _ = simple_topology
    state = SimulationState(simulation_id="SIM001", building_id=model.building_id)
    state.aset_s = 120.0
    state.rset_s = 40.0

    evacuation = {
        "rset_s": 40.0,
        "total_agents": 10,
        "evacuated": 8,
        "dead": 1,
        "incapacitated": 1,
        "evacuation_success_pct": 80.0,
    }
    history = [
        {
            "Z0001": {"danger": "LOW", "tenable": True},
            "Z0002": {"danger": "LOW", "tenable": True},
            "Z0003": {"danger": "LOW", "tenable": True},
        },
        {
            "Z0001": {"danger": "UNTENABLE", "tenable": False},
            "Z0002": {"danger": "LOW", "tenable": True},
            "Z0003": {"danger": "LOW", "tenable": True},
        },
    ]

    report = RiskAnalyzer.generate_report(state, evacuation, history, graph, model)
    assert report["risk_score"] >= 0
    assert report["margin_assessment"] == "MARGINAL"
    assert report["dead_zones"]


def test_monte_carlo_engine_runs_small_sample(simple_building_model):
    engine = MonteCarloEngine(n_runs=2, n_workers=1)
    result = engine.run(simple_building_model, max_steps=5, dt_s=5.0)

    assert result.n_runs >= 1
    assert result.rset_p90_s >= 0
    assert result.evacuation_success_mean_pct >= 0
