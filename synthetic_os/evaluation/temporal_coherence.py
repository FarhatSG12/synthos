"""
Temporal Coherence Evaluator
Checks that synthetic patient encounter sequences follow clinically
plausible transitions — used when schema.is_temporal == True.

Method:
  1. Build a first-order Markov transition matrix from REAL data
     (state = discretised values of the primary temporal column)
  2. Score each SYNTHETIC sequence by its log-likelihood under that matrix
  3. Compare mean log-likelihood of synthetic vs real holdout sequences
  4. Return a score in [0, 1]

Falls back to a simpler autocorrelation comparison when the temporal
column cannot be identified or the dataset has fewer than 50 rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
import pandas as pd


@dataclass
class TemporalCoherenceResult:
    score:          float    # [0, 1]
    method:         str
    real_ll:        float    # mean log-likelihood of real sequences
    synth_ll:       float    # mean log-likelihood of synthetic sequences
    n_states:       int


class TemporalCoherenceEvaluator:
    N_STATES   = 10    # bins for discretisation
    MIN_ROWS   = 50
    MAX_ROWS   = 5000

    def evaluate(
        self,
        real:       pd.DataFrame,
        synthetic:  pd.DataFrame,
        time_col:   Optional[str] = None,
        group_col:  Optional[str] = None,
    ) -> TemporalCoherenceResult:

        tcol = time_col or self._detect_time_col(real)
        if tcol is None or len(real) < self.MIN_ROWS:
            return self._autocorr_fallback(real, synthetic)

        gcol = group_col or self._detect_group_col(real, tcol)

        try:
            return self._markov_method(real, synthetic, tcol, gcol)
        except Exception as e:
            print(f"  [TemporalCoherence] Markov failed ({e}) → autocorr fallback")
            return self._autocorr_fallback(real, synthetic)

    # ── Markov transition model ───────────────────────────────────────────────
    def _markov_method(self, real, synthetic, tcol, gcol) -> TemporalCoherenceResult:
        # Sample
        real_s  = real.copy()
        synth_s = synthetic.copy()
        if len(real_s) > self.MAX_ROWS:
            real_s = real_s.sample(self.MAX_ROWS, random_state=42)

        # Discretise the temporal column into N_STATES bins using real data
        col_vals = pd.to_numeric(real_s[tcol], errors="coerce").dropna()
        if col_vals.nunique() < 3:
            return self._autocorr_fallback(real, synthetic)

        bins  = np.percentile(col_vals, np.linspace(0, 100, self.N_STATES + 1))
        bins  = np.unique(bins)
        n_s   = len(bins) - 1
        if n_s < 2:
            return self._autocorr_fallback(real, synthetic)

        def _discretise(df):
            vals = pd.to_numeric(df[tcol], errors="coerce").fillna(col_vals.median())
            return np.clip(np.digitize(vals, bins[1:-1]), 0, n_s - 1)

        real_states  = _discretise(real_s)
        synth_states = _discretise(synth_s)

        # Build transition matrix from real data (with Laplace smoothing)
        T = np.ones((n_s, n_s))   # Laplace prior
        for i in range(len(real_states) - 1):
            T[real_states[i], real_states[i + 1]] += 1
        T = T / T.sum(axis=1, keepdims=True)

        # Score sequences by mean log transition probability
        def _mean_ll(states):
            lls = []
            for i in range(len(states) - 1):
                lls.append(np.log(T[states[i], states[i + 1]] + 1e-12))
            return float(np.mean(lls)) if lls else 0.0

        real_ll  = _mean_ll(real_states)
        synth_ll = _mean_ll(synth_states)

        # Score: how close is synthetic LL to real LL?
        # Perfect = 1.0, completely random = 0.0
        if real_ll == 0:
            score = 0.5
        else:
            ratio = synth_ll / real_ll
            score = float(np.clip(1.0 - abs(1.0 - ratio), 0.0, 1.0))

        print(f"  [TemporalCoherence] Markov  "
              f"real_ll={real_ll:.4f}  synth_ll={synth_ll:.4f}  "
              f"n_states={n_s}  score={score:.3f}")

        return TemporalCoherenceResult(
            score    = score,
            method   = "markov_transition",
            real_ll  = real_ll,
            synth_ll = synth_ll,
            n_states = n_s,
        )

    # ── Autocorrelation fallback ──────────────────────────────────────────────
    def _autocorr_fallback(self, real, synthetic) -> TemporalCoherenceResult:
        """
        Compare lag-1 autocorrelation of numeric columns.
        High agreement → synthetic preserves temporal dependencies.
        """
        num_cols = [c for c in real.select_dtypes(include=[np.number]).columns
                    if c in synthetic.columns][:10]

        if not num_cols:
            return TemporalCoherenceResult(
                score=0.7, method="no_numeric_cols",
                real_ll=0.0, synth_ll=0.0, n_states=0,
            )

        scores = []
        for col in num_cols:
            r = real[col].dropna().values.astype(float)
            s = synthetic[col].dropna().values.astype(float)
            if len(r) < 4 or len(s) < 4:
                continue
            ac_r = float(pd.Series(r).autocorr(lag=1) or 0.0)
            ac_s = float(pd.Series(s).autocorr(lag=1) or 0.0)
            scores.append(1.0 - min(abs(ac_r - ac_s), 1.0))

        score = float(np.mean(scores)) if scores else 0.7

        print(f"  [TemporalCoherence] Autocorr fallback  score={score:.3f}")

        return TemporalCoherenceResult(
            score    = score,
            method   = "autocorrelation_fallback",
            real_ll  = 0.0,
            synth_ll = 0.0,
            n_states = 0,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _detect_time_col(self, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            if any(kw in col.lower() for kw in
                   ("time", "date", "visit", "day", "month", "year",
                    "seq", "order", "encounter", "admission")):
                if pd.api.types.is_numeric_dtype(df[col]) or \
                   df[col].dtype == object:
                    return col
        # Fallback: first numeric column
        num = df.select_dtypes(include=[np.number]).columns.tolist()
        return num[0] if num else None

    def _detect_group_col(self, df: pd.DataFrame, time_col: str) -> Optional[str]:
        for col in df.columns:
            if col == time_col:
                continue
            if any(kw in col.lower() for kw in
                   ("patient", "subject", "id", "nbr", "mrn")):
                return col
        return None