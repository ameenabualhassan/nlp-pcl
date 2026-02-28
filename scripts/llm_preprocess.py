#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm


# -----------------------------
# Prompt + schema constraints
# -----------------------------
TRAIT_KEYS = [
    "unbalanced_power_relations",
    "shallow_solution",
    "presupposition",
    "authority_voice",
    "metaphor_or_euphemism",
    "compassion_pity_framing",
    "poorer_the_merrier",
    "us_vs_them_framing",
    "saviour_framing",
]

TRAIT_ABBR = {
    "unbalanced_power_relations": "upr",
    "shallow_solution": "shal",
    "presupposition": "pres",
    "authority_voice": "auth",
    "metaphor_or_euphemism": "meta",
    "compassion_pity_framing": "comp",
    "poorer_the_merrier": "merr",
    "us_vs_them_framing": "us",
    "saviour_framing": "sav",
}

FORBIDDEN_KEY_PATTERNS = [
    r"\bis_pcl\b",
    r"\bpcl\b",
    r"\bverdict\b",
    r"\blabel\b",
    r"\btarget\b",
    r"\btarget_flag\b",
    r"\bclassification\b",
]

def ollama_json_schema() -> Dict[str, Any]:
    # Strongly constrains output shape; Ollama will try to enforce this.
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["traits", "evidence", "notes"],
        "properties": {
            "traits": {
                "type": "object",
                "additionalProperties": False,
                "required": TRAIT_KEYS,
                "properties": {k: {"type": "integer", "enum": [0, 1]} for k in TRAIT_KEYS},
            },
            "evidence": {
                "type": "array",
                "minItems": 0,
                "maxItems": 3,
                "items": {"type": "string"},
            },
            "notes": {"type": "string"},
        },
    }


def output_json_schema() -> Dict[str, Any]:
    # strict schema that matches parse_and_validate expectations
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["traits", "evidence", "notes"],
        "properties": {
            "traits": {
                "type": "object",
                "additionalProperties": False,
                "required": TRAIT_KEYS,
                "properties": {k: {"type": "integer", "enum": [0, 1]} for k in TRAIT_KEYS},
            },
            "evidence": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string"},
            },
            "notes": {"type": "string"},
        },
    }


def build_system_prompt() -> str:
    return (
        "You are a careful annotation assistant.\n"
        "You must NOT classify PCL vs No PCL.\n"
        "You only extract rhetorical traits and cite short evidence snippets copied from the input.\n"
        "Return ONLY valid JSON matching the schema. No extra keys. No prose."
    )


def build_user_prompt(paragraph: str) -> str:
    keys = ", ".join(TRAIT_KEYS)
    return (
        "Task: Extract rhetorical traits (NOT a PCL verdict).\n"
        "Return ONLY a single JSON object that matches the required schema.\n\n"
        "Constraints:\n"
        f"- traits must include exactly these keys: {keys}\n"
        "- each traits value must be 0 or 1 (integer)\n"
        "- evidence: 0–3 short phrases copied verbatim from the paragraph (<=12 words each)\n"
        "- notes: <=12 words, neutral, no verdict\n"
        "- Do NOT include any keys like pcl, label, verdict, target.\n\n"
        "PARAGRAPH:\n"
        f"{paragraph}"
    )


def extract_json_object(text: str) -> str:
    """
    If the model returns extra wrapping text, extract the first {...} JSON object.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return m.group(0)


def validate_no_forbidden_keys(obj: Dict[str, Any]) -> None:
    flat = json.dumps(obj, ensure_ascii=False).lower()
    for pat in FORBIDDEN_KEY_PATTERNS:
        if re.search(pat, flat):
            raise ValueError(f"Forbidden key/pattern detected in output: {pat}")


def parse_and_validate(output_text: str) -> Dict[str, Any]:
    raw_json = extract_json_object(output_text)
    obj = json.loads(raw_json)

    if not isinstance(obj, dict):
        raise ValueError("Top-level output is not a JSON object.")

    # exact keys only
    expected_top = {"traits", "evidence", "notes"}
    if set(obj.keys()) != expected_top:
        raise ValueError(f"Top-level keys must be exactly {sorted(expected_top)}; got {sorted(obj.keys())}")

    validate_no_forbidden_keys(obj)

    traits = obj["traits"]
    evidence = obj["evidence"]
    notes = obj["notes"]

    if not isinstance(traits, dict):
        raise ValueError("traits must be an object")
    if not isinstance(evidence, list):
        raise ValueError("evidence must be a list")
    if not isinstance(notes, str):
        raise ValueError("notes must be a string")

    # trait keys must match exactly
    if set(traits.keys()) != set(TRAIT_KEYS):
        raise ValueError(f"traits keys mismatch. Expected {sorted(TRAIT_KEYS)} got {sorted(traits.keys())}")

    # values must be 0/1 ints (or strings convertible)
    norm_traits: Dict[str, int] = {}
    for k in TRAIT_KEYS:
        v = traits.get(k)
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)) and v in (0, 1):
            norm_traits[k] = int(v)
        elif isinstance(v, str) and v.strip() in ("0", "1"):
            norm_traits[k] = int(v.strip())
        else:
            raise ValueError(f"Trait {k} must be 0/1; got {v!r}")

    # evidence constraints (soft)
    norm_evidence: List[str] = []
    for e in evidence[:3]:
        if not isinstance(e, str):
            continue
        e = " ".join(e.strip().split())
        if not e:
            continue
        # truncate to 12 words max if model violates
        words = e.split()
        if len(words) > 12:
            e = " ".join(words[:12])
        norm_evidence.append(e)
    norm_evidence = norm_evidence[:3]

    # notes: keep short-ish (soft)
    notes = " ".join(notes.strip().split())
    if len(notes.split()) > 12:
        notes = " ".join(notes.split()[:12])

    return {"traits": norm_traits, "evidence": norm_evidence, "notes": notes, "raw": obj}


def make_prefix(traits: Dict[str, int], evidence: List[str]) -> str:
    """
    Put the trait cues FIRST so they aren't truncated away by max_length tokenization.
    Keep it compact to preserve token budget.
    """
    parts = []
    for k in TRAIT_KEYS:
        parts.append(f"{TRAIT_ABBR[k]}={traits[k]}")
    trait_str = " ".join(parts)
    evid_str = " | ".join(evidence)
    return f"[PCL_TRAITS] {trait_str}\n[PCL_EVIDENCE] {evid_str}\n\n"


def call_ollama_generate(
    ollama_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    num_predict: int,
    timeout_s: int,
    retries: int,
    sleep_s: float,
) -> str:
    url = ollama_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout_s)
            r.raise_for_status()
            data = r.json()

            # chat response shape
            msg = data.get("message", {})
            resp = msg.get("content", "")

            if not isinstance(resp, str) or not resp.strip():
                raise ValueError(
                    "Empty response from Ollama(chat): "
                    f"done={data.get('done')} done_reason={data.get('done_reason')} "
                    f"eval_count={data.get('eval_count')} prompt_eval_count={data.get('prompt_eval_count')} "
                    f"keys={list(data.keys())}"
                )

            if sleep_s > 0:
                time.sleep(sleep_s)
            return resp

        except Exception as e:
            last_err = e
            backoff = min(8.0, 0.5 * (2 ** attempt))
            time.sleep(backoff)

    raise RuntimeError(f"Ollama call failed after retries. Last error: {last_err}")



def load_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    if not cache_path.exists():
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = str(rec.get("par_id"))
            if pid and pid != "None":
                cache[pid] = rec
    return cache


def append_cache(cache_path: Path, rec: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def infer_paths(repo_root: Path, split: str) -> Tuple[Path, Path, Path]:
    data_dir = repo_root / "data"
    cache_dir = data_dir / "llm_cache"
    if split == "train":
        in_path = data_dir / "train_df.csv"
        out_path = data_dir / "train_df_llm.csv"
        cache_path = cache_dir / "train.jsonl"
    elif split == "dev":
        in_path = data_dir / "dev_df_2.csv"
        out_path = data_dir / "dev_df_2_llm.csv"
        cache_path = cache_dir / "dev.jsonl"
    elif split == "test":
        in_path = data_dir / "task4_test.tsv"
        out_path = data_dir / "task4_test_llm.tsv"
        cache_path = cache_dir / "test.jsonl"
    else:
        raise ValueError("split must be one of: train, dev, test")
    return in_path, out_path, cache_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, required=True, help="Repo root path (can be \\\\wsl.localhost\\... UNC)")
    ap.add_argument("--split", type=str, choices=["train", "dev", "test"], required=True)
    ap.add_argument("--model", type=str, default="gpt-oss:20b")
    ap.add_argument("--ollama_url", type=str, default="http://localhost:11434")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N rows (0 = all)")
    ap.add_argument("--resume", action="store_true", help="Skip rows already in cache")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output file if exists")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--num_predict", type=int, default=400)
    ap.add_argument("--timeout_s", type=int, default=180)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--sleep_s", type=float, default=0.2)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("llm_preprocess")

    repo_root = Path(args.repo_root)
    in_path, out_path, cache_path = infer_paths(repo_root, args.split)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path}. Use --overwrite to replace.")

    # Load input (these are tab-separated even if extension says .csv)
    df = pd.read_csv(in_path, sep="\t", dtype={"par_id": "int64"}, keep_default_na=False)
    if "text" not in df.columns:
        raise ValueError(f"'text' column not found in {in_path}. Columns: {df.columns.tolist()}")
    if "par_id" not in df.columns:
        raise ValueError(f"'par_id' column not found in {in_path}. Columns: {df.columns.tolist()}")

    n_total = len(df)
    n_use = args.limit if args.limit and args.limit > 0 else n_total
    df = df.iloc[:n_use].copy()

    cache = load_cache(cache_path) if args.resume else {}
    processed = 0
    skipped = 0
    failed = 0

    system_prompt = build_system_prompt()

    augmented_texts: List[str] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{args.split}"):
        par_id = int(row["par_id"])
        text = str(row["text"])

        pid = str(par_id)
        rec = cache.get(pid)

        if rec and rec.get("ok") is True:
            # Reuse cached
            traits = rec["traits"]
            evidence = rec.get("evidence", [])
            prefix = make_prefix(traits, evidence)
            augmented_texts.append(prefix + text)
            skipped += 1
            continue

        try:
            user_prompt = build_user_prompt(text)
            resp = call_ollama_generate(
                ollama_url=args.ollama_url,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
                num_predict=args.num_predict,
                timeout_s=args.timeout_s,
                retries=args.retries,
                sleep_s=args.sleep_s,
            )

            try:
                parsed = parse_and_validate(resp)
            except Exception as pe:
                preview = resp[:400].replace("\n", " ")
                raise RuntimeError(f"{pe} | resp_preview={preview!r}") from pe

            traits = parsed["traits"]
            evidence = parsed["evidence"]
            notes = parsed["notes"]

            prefix = make_prefix(traits, evidence)
            augmented_texts.append(prefix + text)

            append_cache(cache_path, {
                "par_id": par_id,
                "ok": True,
                "traits": traits,
                "evidence": evidence,
                "notes": notes,
            })
            processed += 1

        except Exception as e:
            failed += 1
            # fallback: no prefix, but still keep row count/order identical
            augmented_texts.append(text)
            append_cache(cache_path, {
                "par_id": par_id,
                "ok": False,
                "error": str(e)[:500],
            })
            log.warning(f"par_id={par_id} failed: {e}")

    # Write output: same schema, only text replaced
    out_df = df.copy()
    out_df["text"] = augmented_texts

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)

    # Summary sanity checks
    log.info(f"Input:  {in_path} (rows used={len(df)}/{n_total})")
    log.info(f"Output: {out_path} (rows={len(out_df)})")
    log.info(f"Cache:  {cache_path}")
    log.info(f"Processed new={processed} skipped(cache)={skipped} failed(fallback)={failed}")

    # Basic content checks
    sample = out_df["text"].iloc[0]
    # log.info(f"Sample augmented text (first 300 chars): {sample[:300].replace('\\n',' | ')}")


    preview = sample[:300].replace("\n", " | ")
    log.info(f"Sample augmented text (first 300 chars): {preview}")

if __name__ == "__main__":
    main()
