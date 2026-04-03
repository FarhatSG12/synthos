"""
Utility Evaluator
Combines three signals:
  - TSTR (Train on Synthetic, Test on Real) — 50% weight
  - Wasserstein distance (marginal distribution fidelity) — 30%
  - Jensen-Shannon Divergence on categorical columns — 20%
Returns a float in [0, 1].

Fixes applied:
  1. Wasserstein: normalise by a robust range (1st–99th percentile of real data)
     and use a softer penalty curve so outliers in one column don't collapse
     the entire score to 0.
  2. TSTR: smooth the baseline so near-baseline performance isn't punished
     as heavily on small datasets; also guard against degenerate single-class
     targets.
  3. JSD neutral (no categoricals) aligned to 0.5 so all three signals
     have a consistent neutral point.
  4. Skip near-constant and ID-like columns in Wasserstein (high cardinality
     numeric columns whose range is driven by IDs drag the score down).
  5. Regression R² clipped to [0, 1] and centred so 0 = random baseline.
  6. TSTR now includes label-encoded categorical features, not just numeric ones.
     Previously ALL object-dtype feature columns were silently dropped, leaving
     the classifier with almost no signal (→ TSTR=0.5 neutral fallback every time).
  7. Text modality: when the dataset has a long-text column, TF-IDF features are
     extracted and used for TSTR so the score reflects semantic similarity rather
     than always returning the neutral 0.5.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


class UtilityEvaluator:

    # Weights
    W_TSTR        = 0.50
    W_WASSERSTEIN = 0.30
    W_JSD         = 0.20

    def evaluate(
        self,
        real:       pd.DataFrame,
        synthetic:  pd.DataFrame,
        target_col: Optional[str] = None,
        modality:   str           = "tabular",   # NEW: "tabular" | "text" | "image"
    ) -> float:

        scores = {}

        # ── 1. TSTR ──────────────────────────────────────────────────────────
        scores["tstr"] = self._tstr(real, synthetic, target_col, modality)

        # ── 2. Wasserstein (numeric columns) ─────────────────────────────────
        scores["wasserstein"] = self._wasserstein(real, synthetic)

        # ── 3. JSD (categorical columns) ─────────────────────────────────────
        scores["jsd"] = self._jsd(real, synthetic)

        utility = (
            self.W_TSTR        * scores["tstr"] +
            self.W_WASSERSTEIN * scores["wasserstein"] +
            self.W_JSD         * scores["jsd"]
        )
        print(f"  [Utility] TSTR={scores['tstr']:.3f}  "
              f"Wass={scores['wasserstein']:.3f}  "
              f"JSD={scores['jsd']:.3f}  "
              f"→ Utility={utility:.3f}")
        return float(np.clip(utility, 0.0, 1.0))

    # ── TSTR ─────────────────────────────────────────────────────────────────
    def _tstr(self, real, synthetic, target_col, modality="tabular"):
        if target_col is None or target_col not in real.columns:
            return 0.5   # neutral — no supervised signal

        # ── Text modality: TF-IDF features ───────────────────────────────────
        if modality == "text":
            return self._tstr_text(real, synthetic, target_col)

        # ── Tabular: encode ALL feature columns, not just numeric ones ────────
        feature_cols = [c for c in real.columns
                        if c != target_col and c in synthetic.columns]
        if not feature_cols:
            return 0.5

        real_clean  = real[feature_cols + [target_col]].dropna()
        synth_clean = synthetic[feature_cols + [target_col]].dropna()
        if len(synth_clean) < 20 or len(real_clean) < 10:
            return 0.5

        # Encode every feature column: numeric kept as-is, categorical label-encoded
        def _encode_features(df_real, df_synth, cols):
            r_parts, s_parts = [], []
            for c in cols:
                if pd.api.types.is_numeric_dtype(df_real[c]):
                    r_vals = pd.to_numeric(df_real[c],  errors="coerce").fillna(0).values.reshape(-1,1)
                    s_vals = pd.to_numeric(df_synth[c], errors="coerce").fillna(0).values.reshape(-1,1)
                else:
                    le2 = LabelEncoder()
                    combined = pd.concat([df_real[c], df_synth[c]]).astype(str)
                    le2.fit(combined)
                    r_vals = le2.transform(df_real[c].astype(str)).reshape(-1,1).astype(float)
                    s_vals = le2.transform(df_synth[c].astype(str)).reshape(-1,1).astype(float)
                r_parts.append(r_vals)
                s_parts.append(s_vals)
            return np.hstack(r_parts), np.hstack(s_parts)

        X_real_feat, X_synth_feat = _encode_features(real_clean, synth_clean, feature_cols)

        target_dtype = real[target_col].dtype
        is_text  = target_dtype == object or str(target_dtype) == "string"
        is_class = is_text or real[target_col].nunique() <= 20

        if is_class:
            le    = LabelEncoder()
            all_y = pd.concat([real_clean[target_col],
                               synth_clean[target_col]]).astype(str)
            le.fit(all_y)
            y_synth = le.transform(synth_clean[target_col].astype(str))
            y_real  = le.transform(real_clean[target_col].astype(str))

            if len(np.unique(y_synth)) < 2:
                return 0.5

            clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                         random_state=42, n_jobs=-1)
            clf.fit(X_synth_feat, y_synth)

            accuracy = float(
                (clf.predict(X_real_feat) == y_real).mean()
            )
            n_classes = len(np.unique(y_real))
            maj_base  = float(
                pd.Series(y_real).value_counts(normalize=True).iloc[0]
            )
            baseline  = 0.7 * maj_base + 0.3 * (1.0 / max(n_classes, 1))

            if accuracy <= baseline:
                return float(np.clip(accuracy / (baseline + 1e-6) * 0.3, 0.0, 0.3))
            return float(
                0.3 + 0.7 * (accuracy - baseline) / (1.0 - baseline + 1e-6)
            )

        else:
            from sklearn.metrics import r2_score
            y_synth_s = pd.to_numeric(synth_clean[target_col], errors="coerce")
            y_real_s  = pd.to_numeric(real_clean[target_col],  errors="coerce")
            mask_s = y_synth_s.notna()
            mask_r = y_real_s.notna()
            if mask_s.sum() < 10 or mask_r.sum() < 10:
                return 0.5
            reg = RandomForestRegressor(n_estimators=100, max_depth=8,
                                        random_state=42, n_jobs=-1)
            reg.fit(X_synth_feat[mask_s], y_synth_s[mask_s].values)
            preds = reg.predict(X_real_feat[mask_r])
            r2    = float(r2_score(y_real_s[mask_r].values, preds))
            return float(np.clip((r2 + 1.0) / 2.0, 0.0, 1.0))

    def _tstr_text(self, real, synthetic, target_col) -> float:
        """
        TSTR for text modality: extract TF-IDF features from the longest text
        column and use them to train/evaluate a classifier on target_col.
        Previously the neutral 0.5 was always returned for text because all
        object-dtype columns were excluded from feat_cols.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
        except ImportError:
            return 0.5

        # Find the text column (longest average string)
        text_col = None
        best_len = 0
        for c in real.columns:
            if c == target_col:
                continue
            if real[c].dtype == object:
                avg_len = real[c].dropna().astype(str).str.len().mean()
                if avg_len > best_len:
                    best_len = avg_len
                    text_col = c

        if text_col is None or best_len < 20:
            return 0.5

        real_clean  = real[[text_col, target_col]].dropna()
        synth_clean = synthetic[[text_col, target_col]].dropna() if target_col in synthetic.columns else pd.DataFrame()

        if len(synth_clean) < 20 or len(real_clean) < 10:
            return 0.5

        if real[target_col].nunique() > 50:
            return 0.5   # too many classes for a quick TSTR

        try:
            tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                                    sublinear_tf=True, min_df=2)
            # Fit on combined corpus so OOV is handled
            tfidf.fit(pd.concat([real_clean[text_col],
                                 synth_clean[text_col]]).astype(str))
            X_synth = tfidf.transform(synth_clean[text_col].astype(str))
            X_real  = tfidf.transform(real_clean[text_col].astype(str))

            le = LabelEncoder()
            combined_y = pd.concat([real_clean[target_col],
                                    synth_clean[target_col]]).astype(str)
            le.fit(combined_y)
            y_synth = le.transform(synth_clean[target_col].astype(str))
            y_real  = le.transform(real_clean[target_col].astype(str))

            if len(np.unique(y_synth)) < 2:
                return 0.5

            clf = LogisticRegression(max_iter=300, C=1.0, random_state=42,
                                     multi_class="auto", n_jobs=-1)
            clf.fit(X_synth, y_synth)
            accuracy  = float((clf.predict(X_real) == y_real).mean())
            n_classes = len(np.unique(y_real))
            maj_base  = float(pd.Series(y_real).value_counts(normalize=True).iloc[0])
            baseline  = 0.7 * maj_base + 0.3 * (1.0 / max(n_classes, 1))
            if accuracy <= baseline:
                return float(np.clip(accuracy / (baseline + 1e-6) * 0.3, 0.0, 0.3))
            return float(0.3 + 0.7 * (accuracy - baseline) / (1.0 - baseline + 1e-6))
        except Exception:
            return 0.5

    # ── Wasserstein ───────────────────────────────────────────────────────────
    def _wasserstein(self, real, synthetic) -> float:
        num_cols = [c for c in real.columns
                    if pd.api.types.is_numeric_dtype(real[c])
                    and c in synthetic.columns]
        if not num_cols:
            return 0.5

        scores = []
        for col in num_cols[:30]:
            r = pd.to_numeric(real[col],      errors="coerce").dropna().values
            s = pd.to_numeric(synthetic[col], errors="coerce").dropna().values
            if len(r) < 2 or len(s) < 2:
                continue

            p1, p99 = np.percentile(r, 1), np.percentile(r, 99)
            rng = float(p99 - p1)
            if rng < 1e-6:
                continue

            if (pd.api.types.is_integer_dtype(real[col])
                    and real[col].nunique() > 0.9 * len(real)):
                continue

            dist  = wasserstein_distance(r, s)
            score = float(np.clip(1.0 - (dist / rng) * 0.8, 0.0, 1.0))
            scores.append(score)

        return float(np.mean(scores)) if scores else 0.5

    # ── JSD ──────────────────────────────────────────────────────────────────
    def _jsd(self, real, synthetic) -> float:
        cat_cols = [c for c in real.columns
                    if (real[c].dtype == object or real[c].nunique() <= 20)
                    and c in synthetic.columns]
        if not cat_cols:
            return 0.5

        scores = []
        for col in cat_cols[:20]:
            cats = list(set(real[col].dropna()) | set(synthetic[col].dropna()))
            if not cats:
                continue
            r = (real[col].value_counts(normalize=True)
                          .reindex(cats, fill_value=0).values + 1e-8)
            s = (synthetic[col].value_counts(normalize=True)
                               .reindex(cats, fill_value=0).values + 1e-8)
            r /= r.sum()
            s /= s.sum()
            jsd = float(jensenshannon(r, s))
            scores.append(1.0 - jsd)
        return float(np.mean(scores)) if scores else 0.5