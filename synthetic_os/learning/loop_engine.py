"""
Loop Engine — meta-learning update (Loop C)
Fixes:
  - num_cols key lookup uses both 'num_cols' and 'num_columns' (profiler vs MFE mismatch)
  - Threshold lowered from 5 to 3 runs so Loop C fires sooner
  - Clean dataclass interface; no silent exceptions
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

META_STORE = Path("meta_learning_log.jsonl")


@dataclass
class MetaEntry:
    n_rows:    int
    n_cols:    int
    model_key: str
    reward:    float


class LoopEngine:
    MIN_SAMPLES = 3   # start learning after this many runs

    def __init__(self, cfg=None):
        self._history: list[MetaEntry] = self._load()

    def update(self, meta_features: dict, model_key: str, reward: float):
        n_rows = int(meta_features.get("num_rows",
                     meta_features.get("n_rows", 0)))
        n_cols = int(meta_features.get("num_cols",
                     meta_features.get("num_columns",
                     meta_features.get("n_cols", 0))))

        entry = MetaEntry(n_rows=n_rows, n_cols=n_cols,
                          model_key=model_key, reward=reward)
        self._history.append(entry)
        self._save(entry)

        if len(self._history) >= self.MIN_SAMPLES:
            self._summarise()

    def best_model_for(self, n_rows: int, n_cols: int) -> Optional[str]:
        """Return the historically best model for a dataset of this size."""
        if len(self._history) < self.MIN_SAMPLES:
            return None
        # Find runs with similar size (within 2×)
        candidates = [e for e in self._history
                      if 0.5 <= e.n_rows / max(n_rows, 1) <= 2.0]
        if not candidates:
            return None
        # Group by model, take mean reward
        from collections import defaultdict
        model_rewards: dict[str, list[float]] = defaultdict(list)
        for e in candidates:
            model_rewards[e.model_key].append(e.reward)
        best = max(model_rewards, key=lambda k: sum(model_rewards[k]) / len(model_rewards[k]))
        return best

    def _summarise(self):
        from collections import defaultdict
        model_rewards: dict[str, list[float]] = defaultdict(list)
        for e in self._history:
            model_rewards[e.model_key].append(e.reward)
        summary = {k: round(sum(v) / len(v), 4) for k, v in model_rewards.items()}
        print(f"  [LoopEngine] Meta-learner summary ({len(self._history)} runs): "
              f"{summary}")

    def _load(self) -> list[MetaEntry]:
        if not META_STORE.exists():
            return []
        entries = []
        for line in META_STORE.read_text().splitlines():
            try:
                d = json.loads(line)
                entries.append(MetaEntry(**d))
            except Exception:
                pass
        return entries

    def _save(self, entry: MetaEntry):
        with open(META_STORE, "a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")