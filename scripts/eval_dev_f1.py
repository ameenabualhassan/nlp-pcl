#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev_csv", type=str, required=True, help="Path to data/dev_df_2.csv (tab-separated)")
    ap.add_argument("--pred", type=str, required=True, help="Path to dev.txt (one 0/1 per line)")
    args = ap.parse_args()

    dev = pd.read_csv(args.dev_csv, sep="\t")
    if "target_flag" not in dev.columns:
        raise ValueError(f"dev_csv missing target_flag column: {dev.columns.tolist()}")

    y_true = dev["target_flag"].astype(int).to_numpy()
    y_pred = np.loadtxt(args.pred, dtype=int)

    print("len(y_true) =", len(y_true))
    print("len(y_pred) =", len(y_pred))
    print("positives true =", int(y_true.sum()))
    print("positives pred =", int(y_pred.sum()))
    print("DEV F1 =", f1_score(y_true, y_pred))


if __name__ == "__main__":
    main()
