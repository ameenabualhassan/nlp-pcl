from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import compute_pos_weight, make_category_matrix
from .metrics import compute_prf, tune_threshold_for_f1
from .modeling import MultiTaskClassifier


@dataclass
class FoldArtifacts:
    fold: int
    best_epoch: int
    best_f1: float
    model_path: Path


class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        y: np.ndarray,
        cat: np.ndarray,
        w_scale: np.ndarray,
        has_aux: np.ndarray,
    ) -> None:
        self.texts = ["" if t is None else str(t) for t in texts]
        self.y = y.astype(np.int64)
        self.cat = cat.astype(np.int64)
        self.w_scale = w_scale.astype(np.float32)
        self.has_aux = has_aux.astype(np.int64)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "text": self.texts[idx],
            "y": int(self.y[idx]),
            "cat": self.cat[idx],
            "w_scale": float(self.w_scale[idx]),
            "has_aux": int(self.has_aux[idx]),
        }


def make_collate(tokenizer, max_length: int):
    def collate(batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        # Force *real* Python strings (handles None, NaN, numpy scalars, etc.)
        texts = ["" if b.get("text") is None else str(b.get("text")) for b in batch]

        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        y = torch.tensor([b["y"] for b in batch], dtype=torch.float32)
        cat = torch.tensor(np.stack([b["cat"] for b in batch]), dtype=torch.float32)
        w_scale = torch.tensor([b["w_scale"] for b in batch], dtype=torch.float32)
        has_aux = torch.tensor([b["has_aux"] for b in batch], dtype=torch.int64)
        return {
            **enc,
            "y": y,
            "cat": cat,
            "w_scale": w_scale,
            "has_aux": has_aux,
        }

    return collate


def _autocast_ctx(mixed_precision: str, device: torch.device):
    if device.type != "cuda" or mixed_precision == "no":
        return torch.autocast(device_type=device.type, enabled=False)
    if mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unknown mixed_precision={mixed_precision}")


@torch.no_grad()
def predict_probs(
    model: MultiTaskClassifier,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: str,
) -> np.ndarray:
    model.eval()
    probs: List[np.ndarray] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with _autocast_ctx(mixed_precision, device):
            logit_bin, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            p = torch.sigmoid(logit_bin).detach().float().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def train_cv(
    df_train,
    *,
    run_dir: Path,
    backbone_name: str,
    pooling: str,
    max_length: int,
    dropout: float,
    folds: int,
    epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float,
    max_grad_norm: float,
    mixed_precision: str,
    use_pos_weight: bool,
    aux_lambda: float,
    aux_on_positives_only: bool,
    use_label_scale_weights: bool,
    label_scale_weights: Dict[int, float],
    threshold_step: float,
    log_every_steps: int,
    logger,
) -> Tuple[List[FoldArtifacts], np.ndarray, float]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(backbone_name, use_fast=True)

    texts = df_train["text"].astype(str).tolist()
    y = df_train["y"].astype(int).to_numpy()
    cat = make_category_matrix(df_train)

    has_aux_col = df_train.get("has_aux_labels", False)
    if isinstance(has_aux_col, bool):
        has_aux = np.full(len(df_train), int(has_aux_col), dtype=np.int64)
    else:
        has_aux = df_train["has_aux_labels"].astype(int).to_numpy()

    # Optional per-example weights from label scale
    if use_label_scale_weights:
        w = (
            df_train["label"].astype(int).map(lambda v: float(label_scale_weights.get(int(v), 1.0))).to_numpy(dtype=np.float32)
        )
    else:
        w = np.ones(len(df_train), dtype=np.float32)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    oof_probs = np.zeros(len(df_train), dtype=np.float32)
    fold_artifacts: List[FoldArtifacts] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        logger.info(f"Fold {fold+1}/{folds}: train={len(tr_idx)} val={len(va_idx)}")

        ds_tr = TextDataset([texts[i] for i in tr_idx], y[tr_idx], cat[tr_idx], w[tr_idx], has_aux[tr_idx])
        ds_va = TextDataset([texts[i] for i in va_idx], y[va_idx], cat[va_idx], w[va_idx], has_aux[va_idx])

        collate = make_collate(tokenizer, max_length)
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate)
        dl_va = DataLoader(ds_va, batch_size=batch_size * 2, shuffle=False, collate_fn=collate)

        model = MultiTaskClassifier(backbone_name=backbone_name, dropout=dropout, pooling=pooling)
        model.to(device)

        # Loss objects
        if use_pos_weight:
            pos_w = compute_pos_weight(y[tr_idx])
            logger.info(f"Fold {fold}: pos_weight={pos_w:.4f} (neg/pos)")
            bce_bin = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=device), reduction="none")
        else:
            bce_bin = torch.nn.BCEWithLogitsLoss(reduction="none")

        bce_cat = torch.nn.BCEWithLogitsLoss(reduction="none")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        total_steps = math.ceil(len(dl_tr) / grad_accum_steps) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and mixed_precision == "fp16"))

        best_f1 = -1.0
        best_epoch = -1
        best_state = None

        global_step = 0
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(dl_tr, desc=f"fold{fold} epoch{epoch}", leave=False)
            for step, batch in enumerate(pbar, start=1):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                y_b = batch["y"].to(device)
                cat_b = batch["cat"].to(device)
                w_b = batch["w_scale"].to(device)
                has_aux_b = batch["has_aux"].to(device)

                with _autocast_ctx(mixed_precision, device):
                    logit_bin, logit_cat = model(input_ids=input_ids, attention_mask=attention_mask)

                    loss_bin = bce_bin(logit_bin, y_b)  # [B]
                    loss_bin = (loss_bin * w_b).mean()

                    if aux_lambda > 0:
                        if aux_on_positives_only:
                            mask = (y_b >= 0.5) & (has_aux_b > 0)
                        else:
                            mask = has_aux_b > 0

                        if mask.any():
                            loss_cat = bce_cat(logit_cat[mask], cat_b[mask]).mean()
                        else:
                            loss_cat = torch.tensor(0.0, device=device)
                    else:
                        loss_cat = torch.tensor(0.0, device=device)

                    loss = loss_bin + aux_lambda * loss_cat

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step % grad_accum_steps == 0 or step == len(dl_tr):
                    if max_grad_norm and max_grad_norm > 0:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    if global_step % log_every_steps == 0:
                        pbar.set_postfix({"loss": float(loss.detach().cpu())})

            # Evaluate epoch
            probs_va = predict_probs(model, dl_va, device, mixed_precision)
            y_va = y[va_idx]
            y_pred = (probs_va >= 0.5).astype(int)
            m = compute_prf(y_va, y_pred)
            logger.info(
                f"Fold {fold} epoch {epoch}: f1_pos@0.5={m['f1_pos']:.4f} p={m['precision_pos']:.4f} r={m['recall_pos']:.4f}"
            )

            if m["f1_pos"] > best_f1:
                best_f1 = float(m["f1_pos"])
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Save best
        fold_dir = models_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_path = fold_dir / "model.pt"
        torch.save(best_state, model_path)
        tokenizer.save_pretrained(fold_dir / "tokenizer")

        # Reload best for OOF
        model.load_state_dict(best_state)
        probs_va = predict_probs(model, dl_va, device, mixed_precision)
        oof_probs[va_idx] = probs_va.astype(np.float32)

        fold_artifacts.append(FoldArtifacts(fold=fold, best_epoch=best_epoch, best_f1=best_f1, model_path=model_path))

    # Tune threshold on OOF
    t_star, f1_star = tune_threshold_for_f1(y.astype(int), oof_probs, step=float(threshold_step))
    logger.info(f"OOF threshold*: {t_star:.2f} (F1={f1_star:.4f})")

    return fold_artifacts, oof_probs, t_star


def ensemble_predict(
    df,
    fold_artifacts: List[FoldArtifacts],
    *,
    backbone_name: str,
    pooling: str,
    max_length: int,
    dropout: float,
    batch_size: int,
    mixed_precision: str,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(backbone_name, use_fast=True)

    texts = ["" if t is None else str(t) for t in df["text"].tolist()]
    y = np.zeros(len(df), dtype=np.int64)
    cat = np.zeros((len(df), 7), dtype=np.int64)
    w = np.ones(len(df), dtype=np.float32)
    has_aux = np.zeros(len(df), dtype=np.int64)

    ds = TextDataset(texts, y, cat, w, has_aux)
    collate = make_collate(tokenizer, max_length)
    dl = DataLoader(ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate)

    probs_all = []
    for fa in fold_artifacts:
        model = MultiTaskClassifier(backbone_name=backbone_name, dropout=dropout, pooling=pooling)
        state = torch.load(fa.model_path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        probs = predict_probs(model, dl, device, mixed_precision)
        probs_all.append(probs)

    probs_ens = np.mean(np.stack(probs_all, axis=0), axis=0)
    return probs_ens.astype(np.float32)
