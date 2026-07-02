"""Run a single Flower simulation and pickle the History for offline analysis."""
import argparse, pickle, pathlib, json
from flwr.simulation import run_simulation
from qpriviot_fl.client_app import app as client_app
from qpriviot_fl.server_app import app as server_app

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-supernodes", type=int, default=10)
    p.add_argument("--out", required=True, help="path/to/history.pkl")
    args = p.parse_args()

    history = run_simulation(
        server_app=server_app, client_app=client_app,
        num_supernodes=args.num_supernodes,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(history, f)

    # Also dump a JSON view for human readers
    json_view = {
        "losses_distributed":   list(history.losses_distributed),
        "losses_centralized":   list(history.losses_centralized),
        "metrics_distributed_fit": {k: list(v) for k, v in history.metrics_distributed_fit.items()},
        "metrics_distributed":     {k: list(v) for k, v in history.metrics_distributed.items()},
        "metrics_centralized":     {k: list(v) for k, v in history.metrics_centralized.items()},
    }
    out.with_suffix(".json").write_text(json.dumps(json_view, indent=2))

if __name__ == "__main__":
    main()
