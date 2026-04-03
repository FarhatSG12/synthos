"""
Attribute Leakage Evaluator
Trains a classifier on synthetic data, tests it on real data.
Leakage advantage = accuracy_synthetic_trained - majority_class_baseline.
If the model trained on synthetic data can predict real attributes much better
than guessing the majority class, sensitive attributes are being leaked.

Safety score < 0.75 → release gate blocks export.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


@dataclass
class AttributeLeakResult:
    safety_score: float
    advantage:    float   # accuracy - baseline
    accuracy:     float
    baseline:     float


class AttributeLeakEvaluator:
    def evaluate(
        self,
        real:       pd.DataFrame,
        synthetic:  pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> AttributeLeakResult:

        # Pick a sensitive column to test
        col = self._pick_target(real, target_col)
        if col is None:
            return AttributeLeakResult(1.0, 0.0, 0.0, 0.0)

        # Feature columns: numeric, excluding the target
        feat_cols = [c for c in real.columns
                     if c != col
                     and pd.api.types.is_numeric_dtype(real[c])
                     and c in synthetic.columns]
        if not feat_cols:
            return AttributeLeakResult(1.0, 0.0, 0.0, 0.0)

        # Encode target
        le        = LabelEncoder()
        all_vals  = pd.concat([real[col], synthetic[col]]).astype(str)
        le.fit(all_vals)

        synth_clean = synthetic[feat_cols + [col]].dropna()
        real_clean  = real[feat_cols + [col]].dropna()

        if len(synth_clean) < 20 or len(real_clean) < 10:
            return AttributeLeakResult(1.0, 0.0, 0.0, 0.0)

        X_train = synth_clean[feat_cols].values.astype(float)
        y_train = le.transform(synth_clean[col].astype(str))
        X_test  = real_clean[feat_cols].values.astype(float)
        y_test  = le.transform(real_clean[col].astype(str))

        clf = RandomForestClassifier(n_estimators=50, max_depth=5,
                                     random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        accuracy = float((clf.predict(X_test) == y_test).mean())
        baseline = float(pd.Series(y_test).value_counts(normalize=True).max())
        advantage = max(0.0, accuracy - baseline)

        # Safety: 0 advantage → 1.0; advantage ≥ 0.25 → 0.0
        safety = float(np.clip(1.0 - advantage / 0.25, 0.0, 1.0))

        return AttributeLeakResult(
            safety_score = safety,
            advantage    = advantage,
            accuracy     = accuracy,
            baseline     = baseline,
        )

    def _pick_target(self, real: pd.DataFrame,
                     target_col: Optional[str]) -> Optional[str]:
        # FIX: the original code used target_col as the leakage test column.
        # This is circular — of course a model trained on synthetic data can
        # predict the target from correlated features, especially on simple
        # binary classification datasets like heart disease.
        # Leakage should be tested on a SENSITIVE column that is NOT the
        # model target — e.g. sex, age-band, race, or another demographic.
        # If no suitable sensitive column exists, return None (safe default).

        # Columns to never use as the leakage test target
        excluded = {target_col} if target_col else set()

        # Prefer columns that look like demographics / protected attributes
        SENSITIVE_HINTS = {"sex", "gender", "race", "ethnicity", "age",
                           "religion", "nationality", "marital", "zip",
                           "income", "education", "insurance"}

        candidates = [
            col for col in real.columns
            if col not in excluded
            and real[col].dtype == object
            and 2 <= real[col].nunique() <= 10
        ]

        # Also consider low-cardinality numeric columns (e.g. sex=0/1)
        candidates += [
            col for col in real.columns
            if col not in excluded
            and pd.api.types.is_numeric_dtype(real[col])
            and 2 <= real[col].nunique() <= 5
            and col not in candidates
        ]

        if not candidates:
            return None

        # Prioritise any column whose name contains a sensitive hint
        for col in candidates:
            if any(hint in col.lower() for hint in SENSITIVE_HINTS):
                return col

        # Otherwise pick the lowest-cardinality non-target column
        # (most likely to be a protected attribute)
        return min(candidates, key=lambda c: real[c].nunique())