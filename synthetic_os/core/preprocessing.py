"""
Preprocessing — fit/transform/inverse_transform for synthetic data pipeline
Fixes:
  - OHE detection now requires ALL values in a group to be binary (0/1)
    so readmission's diag_1, metformin_up etc. are NOT treated as OHE groups
  - Target column excluded from StandardScaler
  - Original dtype recorded and enforced on inverse_transform
  - Float precision capped to original decimal places
  - Categorical inverse OHE via argmax on each group
  - Class balancing via bootstrap resampling (balance_classes=True by default)
  - Range clamping: no values outside real data min/max
  - discrete_columns parameter accepted so categorical cols are never scaled
    (fixes CTGAN crash and improves utility for mixed-type datasets)
  - balance_classes now uses min(n_per, len(subset)*4) cap to avoid
    over-replicating tiny rare classes into artificial majority
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional


class Preprocessor:
    def __init__(self):
        self._scaler         = StandardScaler()
        self._le             = LabelEncoder()
        self._ohe_groups     = {}
        self._numeric_cols   = []
        self._cat_cols       = []
        self._target_col     = None
        self._target_classes = []
        self._col_dtypes     = {}
        self._col_ranges     = {}
        self._col_decimals   = {}
        self._fitted         = False
        self._all_cols       = []
        self._discrete_cols  = set()   # NEW: explicit categorical set

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None,
            discrete_columns: Optional[list] = None):
        self._target_col    = target_col
        self._all_cols      = df.columns.tolist()
        # Build the definitive discrete set from explicit list OR dtype detection
        if discrete_columns is not None:
            self._discrete_cols = set(discrete_columns) & set(df.columns)
        else:
            self._discrete_cols = {
                c for c in df.columns
                if df[c].dtype == object
                or str(df[c].dtype).startswith("category")
                or (pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() <= 20)
            }

        for col in df.columns:
            self._col_dtypes[col] = str(df[col].dtype)
            if pd.api.types.is_numeric_dtype(df[col]):
                self._col_ranges[col] = (float(df[col].min()), float(df[col].max()))
                sample = df[col].dropna().astype(str).head(100)
                decimals = sample.apply(
                    lambda x: len(x.split('.')[1]) if '.' in x else 0
                ).max()
                self._col_decimals[col] = int(decimals)

        feature_cols = [c for c in df.columns if c != target_col]

        # FIX: OHE detection — only treat as OHE if:
        #   1. 2+ columns share the same prefix
        #   2. ALL values in those columns are strictly binary (0 or 1)
        #   This prevents readmission's diag_1, metformin_up, etc. from
        #   being grouped and corrupted by argmax reconstruction.
        ohe_detected = {}
        non_ohe      = []
        for col in feature_cols:
            m = re.match(r'^(.+)_([^_]+)$', col)
            if m:
                prefix = m.group(1)
                ohe_detected.setdefault(prefix, []).append(col)
            else:
                non_ohe.append(col)

        for prefix, cols in ohe_detected.items():
            if len(cols) >= 2 and self._is_ohe_group(df, cols):
                self._ohe_groups[prefix] = cols
            else:
                non_ohe.extend(cols)

        ohe_flat = [c for grp in self._ohe_groups.values() for c in grp]

        # FIX: exclude discrete_cols from StandardScaler — scaling a column
        # like Gender (0/1 or Male/Female) distorts it and causes CTGAN to
        # treat it as a continuous Gaussian feature.
        self._numeric_cols = [
            c for c in non_ohe
            if pd.api.types.is_numeric_dtype(df[c])
            and c not in self._discrete_cols
        ]
        self._cat_cols = [
            c for c in non_ohe
            if not pd.api.types.is_numeric_dtype(df[c])
            and c not in ohe_flat
        ]

        if self._numeric_cols:
            self._scaler.fit(df[self._numeric_cols])

        if target_col and target_col in df.columns:
            self._target_classes = sorted(df[target_col].dropna().unique().tolist())
            self._le.fit(df[target_col].astype(str))

        self._fitted = True
        return self

    def _is_ohe_group(self, df: pd.DataFrame, cols: list) -> bool:
        """Return True only if every column in the group is strictly binary."""
        for col in cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False
            unique_vals = set(df[col].dropna().unique())
            if not unique_vals.issubset({0, 1, 0.0, 1.0}):
                return False
        return True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self._numeric_cols:
            out[self._numeric_cols] = self._scaler.transform(df[self._numeric_cols])
        if self._target_col and self._target_col in out.columns:
            out[self._target_col] = self._le.transform(
                out[self._target_col].astype(str)
            )
        return out

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None,
                      discrete_columns: Optional[list] = None) -> pd.DataFrame:
        return self.fit(df, target_col, discrete_columns).transform(df)

    def inverse_transform(self, df: pd.DataFrame,
                          balance_classes: bool = True) -> pd.DataFrame:
        out = df.copy()

        if self._numeric_cols:
            present = [c for c in self._numeric_cols if c in out.columns]
            if present:
                out[present] = self._scaler.inverse_transform(
                    out[present].values.astype(float).reshape(-1, len(present))
                )

        for prefix, cols in self._ohe_groups.items():
            present_cols = [c for c in cols if c in out.columns]
            if not present_cols:
                continue
            idx      = out[present_cols].values.argmax(axis=1)
            labels   = [c.replace(f"{prefix}_", "", 1) for c in present_cols]
            out[prefix] = [labels[i] for i in idx]
            out.drop(columns=present_cols, inplace=True)

        if self._target_col and self._target_col in out.columns:
            raw       = out[self._target_col].values.astype(float)
            n_classes = len(self._le.classes_)
            clipped   = np.clip(np.round(raw), 0, n_classes - 1).astype(int)
            out[self._target_col] = self._le.inverse_transform(clipped)

        for col in out.columns:
            if col not in self._col_dtypes:
                continue
            dtype = self._col_dtypes[col]
            if "int" in dtype and pd.api.types.is_numeric_dtype(out[col]):
                lo, hi = self._col_ranges.get(col, (-1e9, 1e9))
                out[col] = out[col].astype(float).clip(lo, hi).round(0).astype(int)
            elif "float" in dtype and pd.api.types.is_numeric_dtype(out[col]):
                lo, hi   = self._col_ranges.get(col, (-1e9, 1e9))
                decimals  = self._col_decimals.get(col, 4)
                out[col]  = out[col].astype(float).clip(lo, hi).round(decimals)

        if balance_classes and self._target_col and self._target_col in out.columns:
            n_total   = len(out)
            classes   = out[self._target_col].unique()
            n_per     = max(1, n_total // len(classes))
            parts     = []
            for cls in classes:
                subset = out[out[self._target_col] == cls]
                if len(subset) == 0:
                    continue
                # FIX: cap resampling to 4× the class's natural size so rare
                # classes don't balloon into artificial majorities and distort
                # the distribution metrics (Wasserstein, JSD).
                n_draw = min(n_per, len(subset) * 4)
                if len(subset) >= n_draw:
                    parts.append(subset.sample(n_draw, random_state=42))
                else:
                    parts.append(subset.sample(n_draw, replace=True, random_state=42))
            out = pd.concat(parts, ignore_index=True).sample(
                frac=1, random_state=42
            ).reset_index(drop=True)
            print(f"  [Preprocess] Balanced classes: "
                  f"{dict(out[self._target_col].value_counts())}")

        return out

    def fit(self, df: pd.DataFrame, target_col: Optional[str] = None):
        self._target_col = target_col
        self._all_cols   = df.columns.tolist()

        for col in df.columns:
            self._col_dtypes[col] = str(df[col].dtype)
            if pd.api.types.is_numeric_dtype(df[col]):
                self._col_ranges[col] = (float(df[col].min()), float(df[col].max()))
                sample = df[col].dropna().astype(str).head(100)
                decimals = sample.apply(
                    lambda x: len(x.split('.')[1]) if '.' in x else 0
                ).max()
                self._col_decimals[col] = int(decimals)

        feature_cols = [c for c in df.columns if c != target_col]

        # FIX: OHE detection — only treat as OHE if:
        #   1. 2+ columns share the same prefix
        #   2. ALL values in those columns are strictly binary (0 or 1)
        #   This prevents readmission's diag_1, metformin_up, etc. from
        #   being grouped and corrupted by argmax reconstruction.
        ohe_detected = {}
        non_ohe      = []
        for col in feature_cols:
            m = re.match(r'^(.+)_([^_]+)$', col)
            if m:
                prefix = m.group(1)
                ohe_detected.setdefault(prefix, []).append(col)
            else:
                non_ohe.append(col)

        for prefix, cols in ohe_detected.items():
            if len(cols) >= 2 and self._is_ohe_group(df, cols):
                self._ohe_groups[prefix] = cols
            else:
                non_ohe.extend(cols)

        ohe_flat = [c for grp in self._ohe_groups.values() for c in grp]
        self._numeric_cols = [c for c in non_ohe
                              if pd.api.types.is_numeric_dtype(df[c])]
        self._cat_cols     = [c for c in non_ohe
                              if not pd.api.types.is_numeric_dtype(df[c])
                              and c not in ohe_flat]

        if self._numeric_cols:
            self._scaler.fit(df[self._numeric_cols])

        if target_col and target_col in df.columns:
            self._target_classes = sorted(df[target_col].dropna().unique().tolist())
            self._le.fit(df[target_col].astype(str))

        self._fitted = True
        return self

    def _is_ohe_group(self, df: pd.DataFrame, cols: list) -> bool:
        """Return True only if every column in the group is strictly binary."""
        for col in cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False
            unique_vals = set(df[col].dropna().unique())
            if not unique_vals.issubset({0, 1, 0.0, 1.0}):
                return False
        return True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self._numeric_cols:
            out[self._numeric_cols] = self._scaler.transform(df[self._numeric_cols])
        if self._target_col and self._target_col in out.columns:
            out[self._target_col] = self._le.transform(
                out[self._target_col].astype(str)
            )
        return out

    def fit_transform(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        return self.fit(df, target_col).transform(df)

    def inverse_transform(self, df: pd.DataFrame,
                          balance_classes: bool = True) -> pd.DataFrame:
        out = df.copy()

        if self._numeric_cols:
            present = [c for c in self._numeric_cols if c in out.columns]
            if present:
                out[present] = self._scaler.inverse_transform(
                    out[present].values.astype(float).reshape(-1, len(present))
                )

        for prefix, cols in self._ohe_groups.items():
            present_cols = [c for c in cols if c in out.columns]
            if not present_cols:
                continue
            idx      = out[present_cols].values.argmax(axis=1)
            labels   = [c.replace(f"{prefix}_", "", 1) for c in present_cols]
            out[prefix] = [labels[i] for i in idx]
            out.drop(columns=present_cols, inplace=True)

        if self._target_col and self._target_col in out.columns:
            raw       = out[self._target_col].values.astype(float)
            n_classes = len(self._le.classes_)
            clipped   = np.clip(np.round(raw), 0, n_classes - 1).astype(int)
            out[self._target_col] = self._le.inverse_transform(clipped)

        for col in out.columns:
            if col not in self._col_dtypes:
                continue
            dtype = self._col_dtypes[col]
            if "int" in dtype and pd.api.types.is_numeric_dtype(out[col]):
                lo, hi = self._col_ranges.get(col, (-1e9, 1e9))
                out[col] = out[col].astype(float).clip(lo, hi).round(0).astype(int)
            elif "float" in dtype and pd.api.types.is_numeric_dtype(out[col]):
                lo, hi   = self._col_ranges.get(col, (-1e9, 1e9))
                decimals  = self._col_decimals.get(col, 4)
                out[col]  = out[col].astype(float).clip(lo, hi).round(decimals)

        if balance_classes and self._target_col and self._target_col in out.columns:
            n_total   = len(out)
            classes   = out[self._target_col].unique()
            n_per     = max(1, n_total // len(classes))
            parts     = []
            for cls in classes:
                subset = out[out[self._target_col] == cls]
                if len(subset) == 0:
                    continue
                if len(subset) >= n_per:
                    parts.append(subset.sample(n_per, random_state=42))
                else:
                    parts.append(subset.sample(n_per, replace=True, random_state=42))
            out = pd.concat(parts, ignore_index=True).sample(
                frac=1, random_state=42
            ).reset_index(drop=True)
            print(f"  [Preprocess] Balanced classes: "
                  f"{dict(out[self._target_col].value_counts())}")

        return out