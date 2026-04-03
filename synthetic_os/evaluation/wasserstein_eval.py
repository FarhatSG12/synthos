"""
wasserstein_eval.py  —  Utility: JSD + Wasserstein Distance Evaluator

Workflow box: "Utility evaluator — TSTR accuracy · JSD / Wasserstein"

Measures distributional similarity between real and synthetic data
using information-theoretic and optimal-transport distances.
"""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


class WassersteinEvaluator:
    """
    Computes per-column JSD and Wasserstein distances,
    then aggregates into a single utility score.

    Score in [0, 1]. Higher = distributions more similar = better utility.
    """

    MAX_COLS = 30   # cap for speed on wide datasets

    def evaluate(self, real: pd.DataFrame,
                 synthetic: pd.DataFrame) -> dict:
        """
        Returns dict with per-metric scores and an aggregate.
        """
        try:
            numeric_cols = real.select_dtypes(
                include=[np.number]
            ).columns.tolist()

            if len(numeric_cols) > self.MAX_COLS:
                rng = np.random.default_rng(42)
                numeric_cols = list(
                    rng.choice(numeric_cols, self.MAX_COLS, replace=False)
                )

            w_scores  = []
            jsd_scores = []

            for col in numeric_cols:
                r = pd.to_numeric(real[col],      errors="coerce").dropna().values
                s = pd.to_numeric(synthetic[col], errors="coerce").dropna().values

                if len(r) == 0 or len(s) == 0:
                    continue

                # Wasserstein distance (lower = more similar)
                w_dist = wasserstein_distance(r, s)
                # Normalise by range to get a [0,1] score
                col_range = max(r.max() - r.min(), 1e-6)
                w_score   = float(np.clip(1 - w_dist / col_range, 0, 1))
                w_scores.append(w_score)

                # Jensen-Shannon divergence via histogram binning
                bins      = np.histogram_bin_edges(
                    np.concatenate([r, s]), bins=20
                )
                r_hist, _ = np.histogram(r, bins=bins, density=True)
                s_hist, _ = np.histogram(s, bins=bins, density=True)
                r_hist    = r_hist + 1e-10
                s_hist    = s_hist + 1e-10
                r_hist   /= r_hist.sum()
                s_hist   /= s_hist.sum()
                jsd       = float(jensenshannon(r_hist, s_hist))
                jsd_score = float(np.clip(1 - jsd, 0, 1))
                jsd_scores.append(jsd_score)

            avg_w   = float(np.mean(w_scores))   if w_scores   else 0.5
            avg_jsd = float(np.mean(jsd_scores)) if jsd_scores else 0.5

            # Aggregate: equal weight between W and JSD
            aggregate = 0.5 * avg_w + 0.5 * avg_jsd

            print(f"  [Wasserstein] avg_w={avg_w:.3f}  "
                  f"avg_jsd={avg_jsd:.3f}  aggregate={aggregate:.3f}")

            return {
                "wasserstein_score" : avg_w,
                "jsd_score"        : avg_jsd,
                "distributional"   : aggregate,
            }

        except Exception as e:
            print(f"[WassersteinEvaluator WARNING] {e}")
            return {
                "wasserstein_score": 0.5,
                "jsd_score"       : 0.5,
                "distributional"  : 0.5,
            }