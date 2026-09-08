# Private Federated Aggregation of LLM-Agent Memory

This repository contains the code, experiments, and manuscript for the paper: **Private Federated Aggregation of LLM-Agent Memory: Usable Shared Memory with a Measurable Leakage-Drop Guarantee under Secure Aggregation and the Skellam Mechanism**.

## Directory Structure

- `qpriviot_fl/`: Core privacy utilities and mechanism implementation.
- `scripts/agentmem/`: Experiment scripts (utility, leakage, LiRA, and dataset distillation).
- `paper/`: LaTeX source for the manuscript and build scripts.
- `experiment_results/` & `figures/`: Experiment outputs and generated plots.

## Getting Started

Install the project dependencies:

```bash
pip install -e .
```

To compile the paper (requires LaTeX):

```powershell
cd paper
.\build.ps1
```