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

## Run an experiment

```bash
python -m rmlab.train --config configs/smoke.yaml            # defaults from configs/base.yaml
python -m rmlab.train --config configs/smoke.yaml seed=1 train.epochs=10
```

Each run writes `experiments/<YYYYMMDD-HHMMSS>_<name>_s<seed>/` with the resolved
`config.yaml`, `run_info.json` (git commit, versions, resolved values), `metrics.json`,
`model.pt` and `train.log`. Set `RMLAB_RUNS_DIR` to write runs elsewhere.

## Development

```bash
python -m pytest           # run tests
ruff check .               # lint
ruff format .              # format code (use --check to only verify)
```

See `.claude/PLAN.md` for the roadmap.
