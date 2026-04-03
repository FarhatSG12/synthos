"""
Singling-Out Risk Evaluator
A synthetic record 'singles out' a real individual if it is uniquely close
to exactly one real record in the normalised feature space.

Safety score < 0.80 → release gate blocks export.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class SingleOutResult:
    safety_score:     float   # 1 - singling_out_rate
    singling_out_rate: float
    n_singled:        int
    n_checked:        int


class SingleOutEvaluator:
    RADIUS = 0.5    # normalised distance threshold to be "close"
    MAX_SYNTH = 2000
    MAX_REAL  = 2000

    def evaluate(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> SingleOutResult:
        num_cols = [c for c in real.columns
                    if pd.api.types.is_numeric_dtype(real[c])
                    and c in synthetic.columns]
        if not num_cols:
            return SingleOutResult(1.0, 0.0, 0, 0)

        scaler = StandardScaler()
        real_n  = scaler.fit_transform(
            real[num_cols].fillna(0).values.astype(float)
        )
        synth_n = scaler.transform(
            synthetic[num_cols].fillna(0).values.astype(float)
        )

        # Sample to avoid O(N*M) explosion
        if len(real_n) > self.MAX_REAL:
            idx    = np.random.choice(len(real_n), self.MAX_REAL, replace=False)
            real_n = real_n[idx]
        if len(synth_n) > self.MAX_SYNTH:
            idx     = np.random.choice(len(synth_n), self.MAX_SYNTH, replace=False)
            synth_n = synth_n[idx]

        n_singled = 0
        n_checked = len(synth_n)

        for s_row in synth_n:
            dists = np.sqrt(((real_n - s_row) ** 2).sum(axis=1))
            n_close = int((dists < self.RADIUS).sum())
            if n_close == 1:   # uniquely identifies exactly one real record
                n_singled += 1

        rate  = n_singled / max(n_checked, 1)
        score = float(np.clip(1.0 - rate, 0.0, 1.0))

        return SingleOutResult(
            safety_score      = score,
            singling_out_rate = rate,
            n_singled         = n_singled,
            n_checked         = n_checked,
        )