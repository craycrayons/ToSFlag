"""Lap 2: a legal-domain transformer for ToSFlag, behind the same interface.

Why this model, stated as a decision not a default (per the syllabus's
"fine-tune vs prompt vs RAG" framing):

  Lap 1's error analysis showed the residual misses are SEMANTIC -- waiver-of-
  rights and responsibility-shifting clauses written in calm boilerplate, unfair
  because of legal meaning rather than surface vocabulary. A bag-of-words model
  cannot represent that. The fix is therefore not "a bigger model" but "a model
  whose pretraining already encodes legal language." legal-bert
  (nlpaueb/legal-bert-base-uncased) is pretrained on legislation, contracts, and
  court cases, so the waiver/liability register is in-distribution for it.

Design contract with Lap 1:
  - Same binary task (unfair = 1).
  - Emits a probability per clause via `predict_proba`-shaped output, so the
    EXACT SAME threshold-selection and metric code in model.py / run.py works
    unchanged. The transformer is a drop-in scorer, not a parallel pipeline.
  - The comparison against TF-IDF is therefore apples-to-apples: same data,
    same operating-point logic, same cost ratio.

Honesty: fine-tuning needs a GPU to be quick; on CPU it runs but slowly. The
script supports a frozen-encoder + linear-probe mode (fast, no GPU) as the
reproducible default, and full fine-tuning as an opt-in flag. Both are real;
the linear probe is the "runs anywhere" artifact, full fine-tune is the ceiling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEGAL_BERT = "nlpaueb/legal-bert-base-uncased"
MAX_LEN = 256  # clauses are short; 256 covers the long tail in this dataset


@dataclass
class EncoderConfig:
    model_name: str = LEGAL_BERT
    max_len: int = MAX_LEN
    mode: str = "probe"  # "probe" (frozen + logreg, fast) or "finetune" (full)
    epochs: int = 3
    batch_size: int = 16
    lr: float = 2e-5


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class TransformerScorer:
    """Wraps legal-bert to emit P(unfair) per clause, matching the TF-IDF
    pipeline's `predict_proba(...)[:, 1]` contract used by run.py."""

    def __init__(self, cfg: EncoderConfig | None = None):
        self.cfg = cfg or EncoderConfig()
        self._tok = None
        self._model = None
        self._probe = None  # logistic head when in probe mode

    # ---- probe mode: freeze encoder, embed, fit a linear head (fast, CPU-ok) ----
    def _embed(self, texts: list[str]) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._tok is None:
            self._tok = AutoTokenizer.from_pretrained(self.cfg.model_name)
            self._model = AutoModel.from_pretrained(self.cfg.model_name).to(_device())
            self._model.eval()
        device = _device()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.cfg.batch_size):
                batch = list(texts[i : i + self.cfg.batch_size])
                enc = self._tok(
                    batch,
                    truncation=True,
                    max_length=self.cfg.max_len,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                hidden = self._model(**enc).last_hidden_state
                # mean-pool over real tokens (mask-aware)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(pooled.cpu().numpy())
        return np.vstack(out)

    def fit(self, texts, y):
        texts = list(texts)
        if self.cfg.mode == "probe":
            from sklearn.linear_model import LogisticRegression

            X = self._embed(texts)
            self._probe = LogisticRegression(
                max_iter=2000, class_weight="balanced", C=1.0
            ).fit(X, np.asarray(y))
        else:
            self._fit_finetune(texts, np.asarray(y))
        return self

    def predict_proba_unfair(self, texts) -> np.ndarray:
        texts = list(texts)
        if self.cfg.mode == "probe":
            X = self._embed(texts)
            return self._probe.predict_proba(X)[:, 1]
        return self._finetune_scores(texts)

    # ---- full fine-tune mode (opt-in; GPU recommended) ----
    def _fit_finetune(self, texts, y):
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = _device()
        self._tok = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name, num_labels=2
        ).to(device)

        # class weights for the ~4:1 imbalance, mirrored from the baseline's
        # class_weight="balanced" so the comparison stays fair.
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        w = torch.tensor([1.0, neg / max(pos, 1.0)], device=device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=w)

        tok, max_len = self._tok, self.cfg.max_len

        class _DS(Dataset):
            def __len__(self):
                return len(texts)

            def __getitem__(self, i):
                enc = tok(
                    texts[i],
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                return {k: v.squeeze(0) for k, v in enc.items()}, int(y[i])

        def collate(b):
            xs = {k: torch.stack([d[0][k] for d in b]) for k in b[0][0]}
            ys = torch.tensor([d[1] for d in b])
            return xs, ys

        dl = DataLoader(
            _DS(), batch_size=self.cfg.batch_size, shuffle=True, collate_fn=collate
        )
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.cfg.lr)
        self._model.train()
        for _ in range(self.cfg.epochs):
            for xs, ys in dl:
                xs = {k: v.to(device) for k, v in xs.items()}
                ys = ys.to(device)
                opt.zero_grad()
                logits = self._model(**xs).logits
                loss_fn(logits, ys).backward()
                opt.step()

    def _finetune_scores(self, texts) -> np.ndarray:
        import torch

        device = _device()
        self._model.eval()
        scores = []
        with torch.no_grad():
            for i in range(0, len(texts), self.cfg.batch_size):
                batch = texts[i : i + self.cfg.batch_size]
                enc = self._tok(
                    batch,
                    truncation=True,
                    max_length=self.cfg.max_len,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                probs = torch.softmax(self._model(**enc).logits, dim=-1)
                scores.append(probs[:, 1].cpu().numpy())
        return np.concatenate(scores)
