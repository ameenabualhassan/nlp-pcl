#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pcl_exercise.config import load_config
from pcl_exercise.data import load_datasets
from pcl_exercise.metrics import compute_prf, confusion
from pcl_exercise.training import train_cv, ensemble_predict
from pcl_exercise.utils import set_seed, setup_logging, save_json, timestamped_run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    run_dir = timestamped_run_dir(cfg.paths.output_dir, cfg.run_name)
    logger = setup_logging(run_dir, name="train")
    logger.info(f"Run dir: {run_dir}")
    save_json(run_dir / "config.resolved.json", cfg.raw)

    ds = load_datasets(
        train_csv=cfg.paths.train_csv,
        dev_csv=cfg.paths.dev_csv,
        test_tsv=cfg.paths.test_tsv,
        category_labels_train_csv=cfg.paths.category_labels_train_csv,
        category_labels_dev_csv=cfg.paths.category_labels_dev_csv,
        span_categories_tsv=cfg.paths.span_categories_tsv,
        span_min_annotators=cfg.paths.span_min_annotators,
    )

    logger.info(f"Loaded train={len(ds.train)} dev={len(ds.dev)}")
    logger.info(f"Train positives (y=1): {int(ds.train['y'].sum())} / {len(ds.train)}")
    logger.info(f"Dev positives (y=1): {int(ds.dev['y'].sum())} / {len(ds.dev)}")
    logger.info(f"Aux labels available: train={bool(ds.train['has_aux_labels'].any())} dev={bool(ds.dev['has_aux_labels'].any())}")

    fold_artifacts, oof_probs, t_star = train_cv(
        ds.train,
        run_dir=run_dir,
        backbone_name=cfg.raw["model"]["backbone_name"],
        pooling=cfg.raw["model"]["pooling"],
        max_length=int(cfg.raw["model"]["max_length"]),
        dropout=float(cfg.raw["model"].get("dropout", 0.1)),
        folds=int(cfg.raw["train"]["folds"]),
        epochs=int(cfg.raw["train"]["epochs"]),
        batch_size=int(cfg.raw["train"]["batch_size"]),
        grad_accum_steps=int(cfg.raw["train"]["grad_accum_steps"]),
        lr=float(cfg.raw["train"]["lr"]),
        weight_decay=float(cfg.raw["train"]["weight_decay"]),
        warmup_ratio=float(cfg.raw["train"]["warmup_ratio"]),
        max_grad_norm=float(cfg.raw["train"]["max_grad_norm"]),
        mixed_precision=str(cfg.raw["train"].get("mixed_precision", "bf16")),
        use_pos_weight=bool(cfg.raw["loss"].get("use_pos_weight", True)),
        aux_lambda=float(cfg.raw["loss"].get("aux_lambda", 0.1)),
        aux_on_positives_only=bool(cfg.raw["loss"].get("aux_on_positives_only", True)),
        use_label_scale_weights=bool(cfg.raw["loss"].get("use_label_scale_weights", False)),
        label_scale_weights=cfg.raw["loss"].get("label_scale_weights", {}),
        threshold_step=float(cfg.raw["inference"].get("threshold_search_step", 0.01)),
        log_every_steps=int(cfg.raw["logging"].get("log_every_steps", 50)),
        logger=logger,
    )

    np.save(run_dir / "oof_probs.npy", oof_probs)
    save_json(run_dir / "threshold.json", {"threshold": float(t_star)})

    # OOF summary at t*
    y_train = ds.train["y"].astype(int).to_numpy()
    yhat_oof = (oof_probs >= t_star).astype(int)
    oof_m = compute_prf(y_train, yhat_oof)
    oof_c = confusion(y_train, yhat_oof)
    save_json(run_dir / "oof_metrics.json", {**oof_m, **oof_c})
    logger.info(f"OOF metrics @t*: {oof_m} conf={oof_c}")

    # Ensemble on dev
    dev_probs = ensemble_predict(
        ds.dev,
        fold_artifacts,
        backbone_name=cfg.raw["model"]["backbone_name"],
        pooling=cfg.raw["model"]["pooling"],
        max_length=int(cfg.raw["model"]["max_length"]),
        dropout=float(cfg.raw["model"].get("dropout", 0.1)),
        batch_size=int(cfg.raw["train"]["batch_size"]),
        mixed_precision=str(cfg.raw["train"].get("mixed_precision", "bf16")),
    )
    np.save(run_dir / "dev_probs.npy", dev_probs)

    y_dev = ds.dev["y"].astype(int).to_numpy()
    yhat_dev = (dev_probs >= t_star).astype(int)
    dev_m = compute_prf(y_dev, yhat_dev)
    dev_c = confusion(y_dev, yhat_dev)
    save_json(run_dir / "dev_metrics.json", {**dev_m, **dev_c})
    logger.info(f"DEV metrics (binary label) @t*: {dev_m} conf={dev_c}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
