# Robust ML Lab (`rmlab`)

Domain-agnostic experiments on imbalanced/limited data, anomaly detection, OOD detection and trustworthy ML.
Primary domain: telecom (RF signals, network KPI time series). Secondary domain: banking (tabular fraud data).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -c "import rmlab; print(rmlab.__version__)"
```

See `.claude/PLAN.md` for the roadmap.
