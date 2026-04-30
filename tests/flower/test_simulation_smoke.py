import json, pathlib, subprocess
import pytest

@pytest.mark.slow
def test_one_round_iot_simulation(tmp_path, monkeypatch):
    repo = pathlib.Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    
    # Use an absolute path for the results file in the tmp_path
    results_file = str(tmp_path / "results_smoke.json")
    
    cmd = ["flwr", "run", str(repo), "local-simulation",
           "--run-config",
           f'dataset="iot" num-server-rounds=1 results-file="{results_file}"']
    
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    
    # Just find ANY json file in tmp_path or repo
    out = list(tmp_path.glob("*.json")) + list(repo.glob("results_smoke.json"))
    assert out, "No results JSON produced"
    results_path = out[0]
    
    payload = json.loads(results_path.read_text())
    assert len(payload["rounds"]) == 1
    
    # Check if val_accuracy is > 0
    found_acc = False
    for round_data in payload["rounds"]:
        acc = round_data.get("val_accuracy", 0.0)
        if acc > 0:
            found_acc = True
            break
    assert found_acc, f"No accuracy found in {results_file}"
