from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


CANONICAL_CATEGORIES = [
    "Unbalanced_power_relations",
    "Compassion",
    "Presupposition",
    "Authority_voice",
    "Shallow_solution",
    "Metaphors",
    "The_poorer_the_merrier",
]

# Normalize common variants to canonical names
CATEGORY_ALIASES = {
    "metaphor": "Metaphors",
    "metaphors": "Metaphors",
    "metaphors ": "Metaphors",
    "metaphor ": "Metaphors",
    "authority voice": "Authority_voice",
    "authority_voice": "Authority_voice",
    "unbalanced power relations": "Unbalanced_power_relations",
    "unbalanced_power_relations": "Unbalanced_power_relations",
    "shallow solution": "Shallow_solution",
    "shallow_solution": "Shallow_solution",
    "the poorer, the merrier": "The_poorer_the_merrier",
    "the_poorer_the_merrier": "The_poorer_the_merrier",
}


def label_scale_to_binary(label_scale: int) -> int:
    # labels 0–1 -> 0 (No PCL), labels 2–4 -> 1 (PCL)
    return 1 if int(label_scale) >= 2 else 0


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    dev: pd.DataFrame
    test: Optional[pd.DataFrame]


def _read_category_labels_csv(path: Path) -> pd.DataFrame:
    """SemEval-style par_id -> 7-dim list string (e.g. "[1,0,...]")."""
    df = pd.read_csv(path)
    if "par_id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}: {df.columns.tolist()}")

    def parse_list(x: str) -> List[int]:
        arr = ast.literal_eval(x)
        if not isinstance(arr, list) or len(arr) != 7:
            raise ValueError(f"Bad label list in {path}: {x}")
        return [int(v) for v in arr]

    y = np.stack(df["label"].apply(parse_list).to_numpy())
    out = pd.DataFrame(y, columns=[f"cat_{c}" for c in CANONICAL_CATEGORIES])
    out.insert(0, "par_id", df["par_id"].astype(int))
    return out


def _read_span_categories_tsv(path: Path, min_annotators: Optional[int]) -> pd.DataFrame:
    """Aggregate dontpatronizeme_categories.tsv (span-level) into paragraph-level multi-hot labels."""
    # File has 3-line disclaimer header, then blank line, then data.
    # Columns in the provided file: par_id, art_id, paragraph, keyword, country_code, span_start, span_end, span_text, pcl_category, n_annotators
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("-"):
                continue
            if line.startswith("This file"):
                continue
            # data lines begin with an integer par_id
            if not line[0].isdigit():
                continue
            rows.append(line.split("\t"))

    if not rows:
        raise ValueError(f"No data rows found in {path}")

    df = pd.DataFrame(
        rows,
        columns=[
            "par_id",
            "art_id",
            "paragraph",
            "keyword",
            "country_code",
            "span_start",
            "span_end",
            "span_text",
            "pcl_category",
            "n_annotators",
        ],
    )
    df["par_id"] = df["par_id"].astype(int)
    df["n_annotators"] = df["n_annotators"].astype(int)

    if min_annotators is not None:
        df = df[df["n_annotators"] >= int(min_annotators)].copy()

    def norm_cat(s: str) -> str:
        k = str(s).strip()
        k2 = k.lower()
        if k2 in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[k2]
        # Canonical already?
        for c in CANONICAL_CATEGORIES:
            if k == c:
                return c
        # Try replacing spaces with underscores
        k3 = k.replace(" ", "_")
        for c in CANONICAL_CATEGORIES:
            if k3 == c:
                return c
        raise ValueError(f"Unknown pcl_category '{s}' in {path}")

    df["cat_norm"] = df["pcl_category"].apply(norm_cat)

    # Aggregate spans -> paragraph multi-hot
    out = df.groupby("par_id")["cat_norm"].apply(lambda xs: sorted(set(xs))).reset_index()

    for c in CANONICAL_CATEGORIES:
        out[f"cat_{c}"] = out["cat_norm"].apply(lambda xs, c=c: int(c in xs))

    out.drop(columns=["cat_norm"], inplace=True)
    return out


def _ensure_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df has integer column y in {0,1}. Prefer existing target_flag if present."""
    if "target_flag" in df.columns:
        df["y"] = df["target_flag"].astype(int)
    else:
        df["y"] = df["label"].astype(int).apply(label_scale_to_binary).astype(int)
    return df


def _attach_aux(
    df: pd.DataFrame,
    *,
    category_labels_csv: Optional[Path],
    span_categories_tsv: Optional[Path],
    span_min_annotators: Optional[int],
) -> pd.DataFrame:
    cat_cols = [f"cat_{c}" for c in CANONICAL_CATEGORIES]

    aux_df: Optional[pd.DataFrame] = None
    if category_labels_csv is not None and Path(category_labels_csv).exists():
        aux_df = _read_category_labels_csv(Path(category_labels_csv))
    elif span_categories_tsv is not None and Path(span_categories_tsv).exists():
        aux_df = _read_span_categories_tsv(Path(span_categories_tsv), span_min_annotators)

    if aux_df is None:
        for c in cat_cols:
            df[c] = 0
        df["has_aux_labels"] = False
        return df

    merged = df.merge(aux_df, on="par_id", how="left", indicator=True)
    merged["has_aux_labels"] = merged["_merge"] == "both"
    merged.drop(columns=["_merge"], inplace=True)
    for c in cat_cols:
        merged[c] = merged[c].fillna(0).astype(int)

    # For No-PCL paragraphs, force aux labels to all-zeros
    neg_mask = merged["y"].astype(int) == 0
    if neg_mask.any():
        merged.loc[neg_mask, cat_cols] = 0

    return merged


def load_datasets(
    train_csv: Path,
    dev_csv: Path,
    test_tsv: Optional[Path] = None,
    category_labels_train_csv: Optional[Path] = None,
    category_labels_dev_csv: Optional[Path] = None,
    span_categories_tsv: Optional[Path] = None,
    span_min_annotators: Optional[int] = None,
) -> DatasetBundle:
    """Load train/dev (and optional test) and attach binary + auxiliary labels if available."""

    train = pd.read_csv(train_csv, sep="\t")
    dev = pd.read_csv(dev_csv, sep="\t")

    required_cols = {"par_id", "text", "label"}
    for name, df in [("train", train), ("dev", dev)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns {missing} in {train_csv if name=='train' else dev_csv}")

    train["par_id"] = train["par_id"].astype(int)
    dev["par_id"] = dev["par_id"].astype(int)
    train["label"] = train["label"].astype(int)
    dev["label"] = dev["label"].astype(int)

    train = _ensure_binary_label(train)
    dev = _ensure_binary_label(dev)

    # weight column if present
    train["w_scale"] = train["weights"].astype(float) if "weights" in train.columns else 1.0
    dev["w_scale"] = dev["weights"].astype(float) if "weights" in dev.columns else 1.0

    train = _attach_aux(
        train,
        category_labels_csv=category_labels_train_csv,
        span_categories_tsv=span_categories_tsv,
        span_min_annotators=span_min_annotators,
    )
    dev = _attach_aux(
        dev,
        category_labels_csv=category_labels_dev_csv,
        span_categories_tsv=span_categories_tsv,
        span_min_annotators=span_min_annotators,
    )

    test: Optional[pd.DataFrame] = None
    if test_tsv is not None and Path(test_tsv).exists():
        test = pd.read_csv(
            test_tsv,
            sep="\t",
            header=None,
            names=["example_id", "art_id", "keyword", "country_code", "text"],
        )

    return DatasetBundle(train=train, dev=dev, test=test)


def compute_pos_weight(y: np.ndarray) -> float:
    y = y.astype(int)
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    if n_pos == 0:
        raise ValueError("No positive examples in y; cannot compute pos_weight")
    return float(n_neg / n_pos)


def make_category_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [f"cat_{c}" for c in CANONICAL_CATEGORIES]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df[cols].astype(int).to_numpy()
