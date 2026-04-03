"""
MetaFeatureExtractor — extracts dataset signature for meta-learning (Loop C).
Replaces existing meta_features.py.
Exports: MetaFeatureExtractor class.

Features extracted:
  num_rows, num_cols, num_numeric, num_categorical,
  sparsity, imbalance_ratio, has_missing, mean_cardinality
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


class MetaFeatureExtractor:
    """Extracts a compact feature vector describing a dataset."""

    def extract(self, df: pd.DataFrame,
                target_col: Optional[str] = None) -> dict:
        n_rows, n_cols = df.shape
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]

        # Sparsity
        null_frac = float(df.isnull().values.mean())
        zero_frac = float((df[num_cols] == 0).values.mean()) if num_cols else 0.0
        sparsity  = float(min(null_frac + zero_frac * 0.5, 1.0))

        # Imbalance
        imbalance = 1.0
        if target_col and target_col in df.columns:
            vc = df[target_col].value_counts()
            if len(vc) >= 2 and vc.iloc[-1] > 0:
                imbalance = float(vc.iloc[0] / vc.iloc[-1])

        # Mean cardinality of categoricals
        mean_card = float(
            np.mean([df[c].nunique() for c in cat_cols])
        ) if cat_cols else 0.0

        return {
            "num_rows":        n_rows,
            "num_cols":        n_cols,
            "num_numeric":     len(num_cols),
            "num_categorical": len(cat_cols),
            "sparsity":        sparsity,
            "imbalance_ratio": imbalance,
            "has_missing":     float(null_frac > 0.01),
            "mean_cardinality":mean_card,
            # aliases used by some callers
            "n_rows":          n_rows,
            "n_cols":          n_cols,
        }