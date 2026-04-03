"""
MetaSelector — Loop C meta-learner.
Fixes:
  - update() method signature now matches the pipeline call:
    update(meta_features, model_key, reward, epsilon)
  - select() uses both 'num_rows'/'n_rows' key aliases
  - Returns MetaDecision with confident=True only after MIN_SAMPLES runs
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

MIN_SAMPLES = 3


@dataclass
class MetaDecision:
    confident:  bool
    model_key:  str   = ""
    epsilon:    float = 1.0
    reason:     str   = ""


class MetaSelector:
    def __init__(self):
        self._history: list[dict] = []

    def select(self, meta_features: dict) -> MetaDecision:
        if len(self._history) < MIN_SAMPLES:
            print(f"  [MetaSelector] Low confidence → falling back to rules "
                  f"({len(self._history)}/{MIN_SAMPLES} runs so far)")
            return MetaDecision(confident=False, reason="insufficient history")

        n_rows = meta_features.get("num_rows", meta_features.get("n_rows", 0))

        candidates = [
            h for h in self._history
            if 0.5 <= h.get("n_rows", n_rows) / max(n_rows, 1) <= 2.0
        ]
        if not candidates:
            return MetaDecision(confident=False, reason="no similar historical runs")

        from collections import defaultdict
        by_model: dict = defaultdict(list)
        for c in candidates:
            by_model[c["model_key"]].append(c["reward"])
        best_model = max(by_model, key=lambda k: np.mean(by_model[k]))
        best_eps   = float(np.mean([
            c.get("epsilon", 1.0) for c in candidates
            if c["model_key"] == best_model
        ]))

        print(f"  [MetaSelector] Recommending {best_model.upper()} "
              f"(mean reward {np.mean(by_model[best_model]):.3f} "
              f"from {len(by_model[best_model])} similar runs)")
        return MetaDecision(
            confident  = True,
            model_key  = best_model,
            epsilon    = float(np.clip(best_eps, 0.1, 10.0)),
            reason     = f"best on similar data (n={len(candidates)} runs)",
        )

    def update(self, meta_features: dict, model_key: str,
               reward: float, epsilon: float = 1.0):
        """Called by pipeline after every run (Loop C)."""
        self._history.append({
            "n_rows":    meta_features.get("num_rows", meta_features.get("n_rows", 0)),
            "n_cols":    meta_features.get("num_cols", meta_features.get("n_cols", 0)),
            "model_key": model_key,
            "reward":    reward,
            "epsilon":   epsilon,
        })
        print(f"  [MetaSelector] History updated — {len(self._history)} runs recorded")
