# scripts/archive/ — FROZEN legacy code

Code from the **QFL / AdaPriv-FL** direction (Direction A: VQC-vs-CNN + d-governs-DP-crossover),
which was scooped and superseded by the federated agent-memory paper. Kept for provenance
(git history preserved via `git mv`). **Not maintained; not expected to run as-is** — the
relative `sys.path`/`ROOT` paths were correct at the old depth and are stale here.

```
archive/
  python/       QFL experiment drivers, probes, plotting, stats
                  _qfl_*.py            MNIST/CNN/VQC FL runs + d-sweep + tuning
                  _femnist_probe.py    (imports the archived qpriviot_fl.task)
                  run_with_history.py  (imports the archived Flower app)
                  generate_*.py, plot.py, extract_metrics.py, merge_per_client.py,
                  visualize_eval.py, stat_*.py   figures/metrics for FL results_*.json
  shell/        flwr-run experiment suites: run_*.sh, _cifar/_femnist/_skellam_*.sh,
                  quick_*.sh, clean_outputs.sh, ...
  flower_app/   the Flower FL app moved out of qpriviot_fl/:
                  client_app.py server_app.py task.py device_profile.py config.py
                  tests_flower/  (load_data partitioner test — depends on task.py)
```

## Restoring the Flower app
To `flwr run` again: move `flower_app/{client_app,server_app,task,device_profile,config}.py`
back into `qpriviot_fl/`, restore the app imports/`__all__` in `qpriviot_fl/__init__.py`,
and uncomment `[tool.flwr.app.components]` in `pyproject.toml`.
