#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pcl_exercise.config import load_config
from pcl_exercise.data import load_datasets
from pcl_exercise.training import FoldArtifacts, ensemble_predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True, help="outputs/<timestamp>_<run_name>")
    ap.add_argument("--split", type=str, choices=["dev", "test", "both"], default="both")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)

    models_dir = run_dir / "models"
    fold_dirs = sorted([p for p in models_dir.iterdir() if p.is_dir() and p.name.startswith("fold")])
    if not fold_dirs:
        raise FileNotFoundError(f"No fold dirs in {models_dir}")

    fold_artifacts = [
        FoldArtifacts(
            fold=int(p.name.replace("fold", "")),
            best_epoch=-1,
            best_f1=-1.0,
            model_path=p / "model.pt",
        )
        for p in fold_dirs
    ]

    th = json.loads((run_dir / "threshold.json").read_text(encoding="utf-8"))["threshold"]
    threshold = float(th)

    ds = load_datasets(
        train_csv=cfg.paths.train_csv,
        dev_csv=cfg.paths.dev_csv,
        test_tsv=cfg.paths.test_tsv,
        category_labels_train_csv=cfg.paths.category_labels_train_csv,
        category_labels_dev_csv=cfg.paths.category_labels_dev_csv,
        span_categories_tsv=cfg.paths.span_categories_tsv,
        span_min_annotators=cfg.paths.span_min_annotators,
    )

    if args.split in ("dev", "both"):
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
        dev_pred = (dev_probs >= threshold).astype(int)
        out_path = run_dir / "dev.txt"
        np.savetxt(out_path, dev_pred, fmt="%d")
        print(f"Wrote {out_path} (n={len(dev_pred)})")

    if args.split in ("test", "both"):
        if ds.test is None:
            raise FileNotFoundError("No test file loaded. Set paths.test_tsv in config and ensure it exists.")
        test_probs = ensemble_predict(
            ds.test,
            fold_artifacts,
            backbone_name=cfg.raw["model"]["backbone_name"],
            pooling=cfg.raw["model"]["pooling"],
            max_length=int(cfg.raw["model"]["max_length"]),
            dropout=float(cfg.raw["model"].get("dropout", 0.1)),
            batch_size=int(cfg.raw["train"]["batch_size"]),
            mixed_precision=str(cfg.raw["train"].get("mixed_precision", "bf16")),
        )
        test_pred = (test_probs >= threshold).astype(int)
        out_path = run_dir / "test.txt"
        np.savetxt(out_path, test_pred, fmt="%d")
        print(f"Wrote {out_path} (n={len(test_pred)})")


if __name__ == "__main__":
    main()
