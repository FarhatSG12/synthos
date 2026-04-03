"""
Realism evaluator
Fixes:
  - Correlation matrix capped at MAX_CORR_ROWS=200, MAX_CORR_COLS=50
    (prevents 8GB allocation on readmission with 2009 columns)
  - KS and chi-square tests unchanged (column-by-column, always fast)
  - Returns a float in [0, 1]
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy import stats

MAX_CORR_ROWS = 200
MAX_CORR_COLS = 50


class RealismEvaluator:
    def evaluate(self, real: pd.DataFrame, synthetic: pd.DataFrame) -> float:
        scores = []

        # ── 1. KS test on numeric columns ────────────────────────────────────
        # Use real df to identify numeric cols; coerce synthetic to match —
        # inverse-transform can return numeric columns as object/string dtype.
        numeric_cols = real.select_dtypes(include=[np.number]).columns.tolist()
        ks_scores    = []
        for col in numeric_cols:
            if col not in synthetic.columns:
                continue
            r = pd.to_numeric(real[col], errors="coerce").dropna().values
            s = pd.to_numeric(synthetic[col], errors="coerce").dropna().values
            if len(r) < 2 or len(s) < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, _ = stats.ks_2samp(r, s)
            ks_scores.append(1.0 - stat)
        if ks_scores:
            scores.append(float(np.mean(ks_scores)))

        # ── 2. Chi-square on categorical columns ────────────────────────────
        cat_cols   = real.select_dtypes(include=["object", "category"]).columns.tolist()
        chi_scores = []
        for col in cat_cols:
            if col not in synthetic.columns:
                continue
            all_cats = list(set(real[col].dropna()) | set(synthetic[col].dropna()))
            r_counts = real[col].value_counts().reindex(all_cats, fill_value=0).values + 1
            s_counts = synthetic[col].value_counts().reindex(all_cats, fill_value=0).values + 1
            # Normalise to same scale
            s_scaled = s_counts * (r_counts.sum() / s_counts.sum())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, _ = stats.chisquare(f_obs=s_scaled, f_exp=r_counts)
            # Convert to [0,1] score: 0 chi² → 1.0
            chi_scores.append(float(1.0 / (1.0 + stat / len(all_cats))))
        if chi_scores:
            scores.append(float(np.mean(chi_scores)))

        # ── 3. Correlation matrix similarity (sampled for large data) ────────
        if len(numeric_cols) >= 2:
            # Coerce synthetic cols to numeric (inverse-transform may stringify them)
            real_num  = real[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
            synth_num = synthetic[[c for c in numeric_cols if c in synthetic.columns]] \
                            .apply(pd.to_numeric, errors="coerce").dropna()
            # Align columns after coercion
            shared_num = [c for c in numeric_cols if c in synth_num.columns]
            r_sample = real_num[shared_num]
            s_sample = synth_num[shared_num]
            if len(r_sample) > MAX_CORR_ROWS:
                r_sample = r_sample.sample(MAX_CORR_ROWS, random_state=42)
            if len(s_sample) > MAX_CORR_ROWS:
                s_sample = s_sample.sample(MAX_CORR_ROWS, random_state=42)

            # Sample columns
            if len(shared_num) > MAX_CORR_COLS:
                sampled_cols = np.random.choice(shared_num, MAX_CORR_COLS,
                                                replace=False).tolist()
                r_sample = r_sample[sampled_cols]
                s_sample = s_sample[sampled_cols]

            if len(shared_num) >= 2 and len(r_sample) >= 2 and len(s_sample) >= 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r_corr = r_sample.corr().values
                    s_corr = s_sample.corr().values

                # Fill NaN correlations (constant columns) with 0
                r_corr = np.nan_to_num(r_corr)
                s_corr = np.nan_to_num(s_corr)

                corr_diff = np.abs(r_corr - s_corr).mean()
                scores.append(float(1.0 - min(corr_diff, 1.0)))

        if not scores:
            return 0.5

        return float(np.mean(scores))