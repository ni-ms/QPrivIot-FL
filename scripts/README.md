# scripts/

Organized after the pivot from the (scooped) QFL/AdaPriv direction to **federated
agent-memory** (D1+D5). Active paper code is separated from archived legacy code.

```
scripts/
  agentmem/   ACTIVE — agent-memory paper (Python; modules cross-import, keep co-located)
  shell/      ACTIVE — shell drivers
  archive/    FROZEN legacy (see archive/README.md) — not maintained, paths may be stale
```

## Active — `agentmem/` (run with `.venv/bin/python scripts/agentmem/<script>.py`)

| script | role |
|---|---|
| `_longmemeval_data.py` | LongMemEval loader + cached ST embeddings (`experiment_results/_st_lme_*.npz`) |
| `_longmemeval_probe.py` | real-data UTILITY (evidence-recall@k) over an ε sweep |
| `_longmemeval_leakage.py` | real-data LEAKAGE (MIA / low-count tail-AUC / extraction gap) |
| `_agentmem_probe.py` | synthetic + real-ST utility; also the **shared helper module** (buckets, clipping, SecAgg+Skellam release, σ) |
| `_agentmem_leakage.py` | synthetic + real-ST leakage; shared attack helpers |
| `_longmemeval_distill.py` | LLM-distill turns → notes (local Ollama) |
| `_longmemeval_distilled_utility.py` | distilled answer-retrieval utility (vector-only release) |
| `_longmemeval_distilled_analysis.py` | distilled-vs-raw leakage comparison |
| `_skellam_accounting.py` | rigorous Skellam RDP ε accounting (Agarwal NeurIPS'21) |
| `rerun_tables.py` | aggregate `experiment_results/rerun_grid/*.json` → CSV + `SUMMARY_TABLE.md` |
| `rerun_figures.py` | regenerate `figures/fig_*.{png,pdf}` |

All `agentmem/` modules import each other by bare name and reach `qpriviot_fl.privacy_utils`
via a relative `sys.path` insert (`parents[2]` = repo root). **Keep them in one directory.**

## Active — `shell/`

| script | role |
|---|---|
| `rerun_grid.sh` | multi-seed (0-4) final-metric rerun of the 33-config oracle/small grid → JSON. Idempotent. `bash scripts/shell/rerun_grid.sh` |

## Shared dependency

`qpriviot_fl/` was slimmed to `privacy_utils.py` (SecAgg+Skellam crypto) + `__init__.py`.
The Flower FL app was archived to `archive/flower_app/`.
