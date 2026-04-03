"""
MIA Evaluator — Membership Inference Attack
Fixes:
  - Threshold derived from HOLDOUT split of real data (not the training set)
    so attack_success is not mathematically pinned to ~0.05
  - Returns a dataclass with privacy_score and attack_success
  - Works on mixed dtypes (numeric only for distance computation)
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class MIAResult:
    privacy_score:  float
    attack_success: float
    threshold:      float
    train_median:   float
    holdout_median: float


class MIAEvaluator:
    def evaluate(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> MIAResult:
        # Use numeric columns only
        num_cols = [c for c in real.columns
                    if pd.api.types.is_numeric_dtype(real[c])
                    and c in synthetic.columns]
        if not num_cols:
            return MIAResult(0.95, 0.05, 0.0, 0.0, 0.0)

        # Split real into train/holdout
        n      = len(real)
        idx    = np.random.permutation(n)
        split  = max(1, int(n * 0.6))
        train  = real.iloc[idx[:split]][num_cols].values.astype(float)
        hold   = real.iloc[idx[split:]][num_cols].values.astype(float)
        synth  = synthetic[num_cols].dropna().values.astype(float)

        # Normalise using train statistics
        scaler = StandardScaler()
        scaler.fit(train)
        train_s = scaler.transform(train)
        hold_s  = scaler.transform(hold)
        synth_s = scaler.transform(synth[:min(len(synth), 5000)])

        # Threshold: 5th percentile of min-distances from HOLDOUT to synthetic
        hold_min  = _min_distances(hold_s, synth_s)
        threshold = float(np.percentile(hold_min, 5))

        # Attack: what fraction of TRAINING records fall below the threshold?
        train_min   = _min_distances(train_s, synth_s)
        attack_succ = float(np.mean(train_min < threshold))

        # Random baseline: ~5% of holdout falls below threshold by construction
        # Privacy score: how much better does holdout do vs train?
        # If train_attack ≈ holdout_attack → no memorisation → high privacy
        hold_attack  = float(np.mean(hold_min < threshold))
        advantage    = max(0.0, attack_succ - hold_attack)
        privacy      = float(np.clip(1.0 - advantage * 5, 0.0, 1.0))

        print(f"  [MIA] threshold={threshold:.4f}  "
              f"train_median={np.median(train_min):.4f}  "
              f"holdout_median={np.median(hold_min):.4f}  "
              f"attack_success={attack_succ:.4f}  "
              f"privacy={privacy:.4f}")

        return MIAResult(
            privacy_score  = privacy,
            attack_success = attack_succ,
            threshold      = threshold,
            train_median   = float(np.median(train_min)),
            holdout_median = float(np.median(hold_min)),
        )


def _min_distances(query: np.ndarray, reference: np.ndarray,
                   chunk: int = 256) -> np.ndarray:
    """
    Compute minimum L2 distance from each query row to any reference row.
    Chunked to avoid allocating a full N×M matrix.
    """
    n   = len(query)
    out = np.full(n, np.inf)
    for start in range(0, len(reference), chunk):
        ref_chunk = reference[start:start + chunk]          # (C, D)
        # Broadcast: (N, 1, D) - (1, C, D) → (N, C, D) → (N, C)
        diffs = query[:, np.newaxis, :] - ref_chunk[np.newaxis, :, :]
        dists = np.sqrt((diffs ** 2).sum(axis=-1))         # (N, C)
        out   = np.minimum(out, dists.min(axis=1))
    return out