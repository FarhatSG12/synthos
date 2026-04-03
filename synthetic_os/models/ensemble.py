"""
EnsembleModel — TabDDPM + CTGAN in parallel, column-wise blending.
TabDDPM wins on continuous clinical measurements.
CTGAN wins on sparse binary/categorical medication flags.
Blending weight: inverse KS-distance (whichever model fits better wins that column).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from synthetic_os.models.tab_ddpm  import TabDDPM
from synthetic_os.models.dp_ctgan  import DPCTGAN


class EnsembleModel:
    def __init__(self):
        self._tabddpm = TabDDPM()
        self._ctgan   = DPCTGAN()
        self._fitted  = False

    def fit(self, df: pd.DataFrame, epsilon: float = 1.0):
        # Split epsilon budget equally
        eps_each = epsilon / 2.0
        print(f"  [Ensemble] Training TabDDPM + CTGAN  ε_each={eps_each:.3f}")
        self._tabddpm.fit(df, eps_each)
        self._ctgan.fit(df,   eps_each)
        self._real_df = df.copy()
        self._fitted  = True

    def generate(self, df: pd.DataFrame, epsilon: float = 1.0,
                 n_output: int | None = None) -> pd.DataFrame:
        if not self._fitted:
            self.fit(df, epsilon)

        n  = n_output if n_output is not None else len(df)
        s1 = self._tabddpm.generate(df, epsilon, n_output=n)
        s2 = self._ctgan.generate(df,   epsilon, n_output=n)

        # Column-wise blending
        blended = {}
        for col in df.columns:
            if col not in s1.columns or col not in s2.columns:
                blended[col] = s1.get(col, s2.get(col))
                continue

            real_col = self._real_df[col].dropna().values

            # Weight by inverse KS statistic (lower KS → better fit → higher weight)
            if pd.api.types.is_numeric_dtype(df[col]):
                ks1, _ = ks_2samp(real_col.astype(float),
                                   s1[col].dropna().values.astype(float))
                ks2, _ = ks_2samp(real_col.astype(float),
                                   s2[col].dropna().values.astype(float))
                w1 = 1.0 / (ks1 + 1e-6)
                w2 = 1.0 / (ks2 + 1e-6)
                w1, w2 = w1 / (w1 + w2), w2 / (w1 + w2)
                blended[col] = s1[col].values * w1 + s2[col].values * w2
            else:
                # For categoricals: pick whichever model has lower JSD
                from scipy.spatial.distance import jensenshannon
                cats = list(set(real_col) | set(s1[col].dropna()) | set(s2[col].dropna()))
                r  = (pd.Series(real_col).value_counts(normalize=True)
                                         .reindex(cats, fill_value=0).values + 1e-8)
                r /= r.sum()
                p1 = (s1[col].value_counts(normalize=True)
                             .reindex(cats, fill_value=0).values + 1e-8)
                p1 /= p1.sum()
                p2 = (s2[col].value_counts(normalize=True)
                             .reindex(cats, fill_value=0).values + 1e-8)
                p2 /= p2.sum()
                jsd1 = jensenshannon(r, p1)
                jsd2 = jensenshannon(r, p2)
                blended[col] = s1[col].values if jsd1 <= jsd2 else s2[col].values

        out = pd.DataFrame(blended)
        return out.reset_index(drop=True)