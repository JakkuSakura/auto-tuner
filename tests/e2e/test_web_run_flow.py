from __future__ import annotations


def test_web_run_flow(client) -> None:
    create_response = client.post("/api/runs", params={"config_path": "examples/sample_experiment.toml"})
    assert create_response.status_code == 200
    payload = create_response.json()
    assert payload["status"] == "completed"

    list_response = client.get("/api/runs")
    assert list_response.status_code == 200
    listed_runs = list_response.json()
    assert any(item["run_id"] == payload["run_id"] for item in listed_runs)

    run_response = client.get(f"/api/runs/{payload['run_id']}")
    assert run_response.status_code == 200
    detail = run_response.json()
    assert detail["report"]["backend"] == "fake"
    assert detail["report"]["demo"]["input_prompt"]
    assert detail["prompts"]["generation_prompt"]
    assert detail["prompts"]["grading_prompt"]
    assert detail["training"]["mode"] == "simulated"

    download_response = client.get(f"/api/runs/{payload['run_id']}/download/report.json")
    assert download_response.status_code == 200

    export_response = client.get(f"/api/runs/{payload['run_id']}/export")
    assert export_response.status_code == 200

    delete_response = client.delete(f"/api/runs/{payload['run_id']}")
    assert delete_response.status_code == 200

    missing_response = client.get(f"/api/runs/{payload['run_id']}")
    assert missing_response.status_code == 404
