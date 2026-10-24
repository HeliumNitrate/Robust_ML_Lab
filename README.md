# Robust ML Lab (`rmlab`)

Domain-agnostic experiments on imbalanced/limited data, anomaly detection, OOD detection and trustworthy ML.
Primary domain: telecom (RF signals, network KPI time series). Secondary domain: banking (tabular fraud data).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -c "import rmlab; print(rmlab.__version__)"
```

On Windows PowerShell, activate with `.venv\Scripts\activate`.

## Development

```bash
python -m pytest           # run tests
ruff check .               # lint
ruff format .              # format code (use --check to only verify)
```

See `.claude/PLAN.md` for the roadmap.
