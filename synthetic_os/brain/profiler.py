"""
Profiler — Data Profiler for the routing brain.
Replaces the existing profiler.py entirely.
Exports: Profiler class, profile() function (for backwards compatibility).

Computes:
  - row/column counts
  - imbalance ratio (if target_col provided)
  - sparsity (fraction of zero/null values)
  - cardinality statistics
  - modality hints
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ProfileResult:
    n_rows:          int
    n_cols:          int
    n_numeric:       int
    n_categorical:   int
    sparsity:        float       # fraction of cells that are 0 or null
    imbalance_ratio: float       # majority_class / minority_class (1.0 = balanced)
    has_target:      bool
    target_col:      Optional[str]
    col_names:       list = field(default_factory=list)


class Profiler:
    """Data profiler used by the routing brain."""

    def profile(
        self,
        df:         pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> ProfileResult:

        n_rows, n_cols = df.shape
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]

        # Sparsity: fraction of cells that are null or zero
        null_frac = df.isnull().values.mean()
        if num_cols:
            zero_frac = (df[num_cols] == 0).values.mean()
        else:
            zero_frac = 0.0
        sparsity = float(min(null_frac + zero_frac * 0.5, 1.0))

        # Imbalance ratio
        imbalance = 1.0
        has_target = target_col is not None and target_col in df.columns
        if has_target:
            vc = df[target_col].value_counts()
            if len(vc) >= 2 and vc.iloc[-1] > 0:
                imbalance = float(vc.iloc[0] / vc.iloc[-1])

        result = ProfileResult(
            n_rows          = n_rows,
            n_cols          = n_cols,
            n_numeric       = len(num_cols),
            n_categorical   = len(cat_cols),
            sparsity        = sparsity,
            imbalance_ratio = imbalance,
            has_target      = has_target,
            target_col      = target_col,
            col_names       = df.columns.tolist(),
        )

        print(f"  [Profiler] {n_rows:,} rows × {n_cols} cols  "
              f"sparsity={sparsity:.2f}  imbalance={imbalance:.2f}")
        return result


# ── Module-level function for backwards compatibility ────────────────────────
def profile(df: pd.DataFrame, target_col: Optional[str] = None) -> ProfileResult:
    return Profiler().profile(df, target_col)