import time


def test_health_endpoint(api_client):
    response = api_client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_blueprint_upload_and_zone_query(api_client, sample_svg_path):
    with sample_svg_path.open("rb") as handle:
        response = api_client.post(
            "/api/blueprint/upload",
            files={"file": ("model.svg", handle, "image/svg+xml")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["zones"] >= 3
    assert payload["building_id"] != "input_blueprints"

    zones_response = api_client.get(f"/api/blueprint/{payload['building_id']}/zones")
    assert zones_response.status_code == 200
    zones_payload = zones_response.json()
    assert len(zones_payload["zones"]) >= 3
    assert len(zones_payload["openings"]) >= 1


def test_simulation_run_endpoint(api_client, sample_svg_path):
    with sample_svg_path.open("rb") as handle:
        upload = api_client.post(
            "/api/blueprint/upload",
            files={"file": ("model.svg", handle, "image/svg+xml")},
        )

    building_id = upload.json()["building_id"]
    run = api_client.post(
        "/api/simulation/run",
        json={
            "building_id": building_id,
            "n_steps": 3,
            "run_monte_carlo": False,
            "run_optimization": False,
            "generate_report": False,
        },
    )
    assert run.status_code == 200
    sim_id = run.json()["simulation_id"]

    status_payload = None
    for _ in range(30):
        status = api_client.get(f"/api/simulation/{sim_id}/status")
        assert status.status_code == 200
        status_payload = status.json()
        if status_payload["status"] in {"complete", "error"}:
            break
        time.sleep(0.05)

    assert status_payload is not None
    assert status_payload["status"] == "complete"

    results = api_client.get(f"/api/simulation/{sim_id}/results")
    assert results.status_code == 200
    payload = results.json()
    assert "simulation" in payload
    assert "risk_report" in payload
