from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass
class RunPaths:
    data_dir: Path
    train_csv: Path
    dev_csv: Path
    test_tsv: Optional[Path]
    category_labels_train_csv: Optional[Path]
    category_labels_dev_csv: Optional[Path]
    span_categories_tsv: Optional[Path]
    span_min_annotators: Optional[int]
    output_dir: Path


@dataclass
class Config:
    run_name: str
    seed: int
    paths: RunPaths
    raw: Dict[str, Any]


def load_config(path: str | Path) -> Config:
    raw = load_yaml(path)
    base = Path(path).resolve().parent

    def _p(rel: Optional[str]) -> Optional[Path]:
        if rel is None:
            return None
        p = Path(rel)
        return p if p.is_absolute() else (base / p).resolve()

    span_min = deep_get(raw, "paths.span_min_annotators", None)
    span_min = None if span_min is None else int(span_min)

    paths = RunPaths(
        data_dir=_p(deep_get(raw, "paths.data_dir", "../data")) or Path("data"),
        train_csv=_p(deep_get(raw, "paths.train_csv")) or (base / "../data/train_df.csv").resolve(),
        dev_csv=_p(deep_get(raw, "paths.dev_csv")) or (base / "../data/dev_df_2.csv").resolve(),
        test_tsv=_p(deep_get(raw, "paths.test_tsv")),
        category_labels_train_csv=_p(deep_get(raw, "paths.category_labels_train_csv")),
        category_labels_dev_csv=_p(deep_get(raw, "paths.category_labels_dev_csv")),
        span_categories_tsv=_p(deep_get(raw, "paths.span_categories_tsv")),
        span_min_annotators=span_min,
        output_dir=_p(deep_get(raw, "paths.output_dir", "../outputs")) or (base / "../outputs").resolve(),
    )

    return Config(
        run_name=str(raw.get("run_name", "run")),
        seed=int(raw.get("seed", 42)),
        paths=paths,
        raw=raw,
    )
