from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


Pooling = Literal["cls", "mean"]


class MultiTaskClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        dropout: float = 0.1,
        pooling: Pooling = "cls",
        n_categories: int = 7,
    ) -> None:
        super().__init__()
        self.pooling: Pooling = pooling

        cfg = AutoConfig.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name, config=cfg)
        hidden = cfg.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.bin_head = nn.Linear(hidden, 1)
        self.cat_head = nn.Linear(hidden, n_categories)

    def pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return last_hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = self.pool(out.last_hidden_state, attention_mask)
        h = self.dropout(h)
        logit_bin = self.bin_head(h).squeeze(-1)
        logit_cat = self.cat_head(h)
        return logit_bin, logit_cat
