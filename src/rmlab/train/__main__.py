"""Command line: ``python -m rmlab.train --config configs/smoke.yaml [key=value ...]``."""

import argparse
import json

import torch
from torch.utils.data import DataLoader

import rmlab.data  # noqa: F401  (registers datasets)
import rmlab.models  # noqa: F401  (registers models)
from rmlab.data.base import SampleDataset
from rmlab.registry import DATASETS, MODELS
from rmlab.train.loop import build_optimizer, evaluate, train_one_epoch
from rmlab.utils.config import load_config
from rmlab.utils.run import Run, log
from rmlab.utils.seed import set_seed, worker_init_fn


def main(argv: list[str] | None = None) -> Run:
    parser = argparse.ArgumentParser(description="Train a model from an experiment config.")
    parser.add_argument("--config", required=True, help="experiment YAML file")
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. seed=1")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.seed)
    run = Run(cfg)
    try:
        train(cfg, run)
    finally:
        run.close()
    return run


def train(cfg, run: Run) -> None:
    data = DATASETS.build(cfg.data.name, **cfg.data.params)
    train_ds = SampleDataset(data.load("train"))
    val_ds = SampleDataset(data.load("val"))  # the test split is only used by evaluation

    loader_args = {"batch_size": cfg.data.batch_size, "num_workers": cfg.data.num_workers}
    shuffle_gen = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader(
        train_ds, shuffle=True, generator=shuffle_gen, worker_init_fn=worker_init_fn, **loader_args
    )
    val_dl = DataLoader(val_ds, shuffle=False, **loader_args)

    in_features = train_ds[0][0].numel()
    model = MODELS.build(
        cfg.model.name, in_features=in_features, num_classes=data.num_classes, **cfg.model.params
    ).to(run.device)
    optimizer = build_optimizer(model.parameters(), **cfg.train.optimizer)
    run.record(
        in_features=in_features,
        num_classes=data.num_classes,
        n_train=len(train_ds),
        n_val=len(val_ds),
        n_parameters=sum(p.numel() for p in model.parameters()),
    )

    history = []
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_one_epoch(model, train_dl, optimizer, run.device)
        val = evaluate(model, val_dl, run.device)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}}
        )
        log.info(
            "epoch %d  train_loss %.4f  val_loss %.4f  val_acc %.4f",
            epoch,
            train_loss,
            val["loss"],
            val["accuracy"],
        )

    metrics = {"final": history[-1], "history": history}
    (run.dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), run.dir / "model.pt")
    log.info("Done: %s", run.dir)


if __name__ == "__main__":
    main()
