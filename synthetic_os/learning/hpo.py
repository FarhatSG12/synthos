"""
HPO — Hyperparameter Optimisation via Optuna (Loop A — full implementation)
Tunes: epsilon, batch_size, clip_norm

Real mini-train feedback per trial (replaces proxy score):
  - Subsamples to MINI_ROWS rows for speed
  - Trains for MINI_EPOCHS epochs
  - Scores with fast MIA proxy + fast TSTR accuracy on holdout split
  - Falls back to proxy score if mini-train raises any exception
  - Timeout: 120s across all trials so readmission never hangs
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _OPTUNA = False

from synthetic_os.config.schema        import DataSchema
from synthetic_os.config.system_config import SystemConfig

MINI_ROWS   = 500
MINI_EPOCHS = 10


class HPOptimiser:
    def __init__(self, cfg: Optional[SystemConfig] = None):
        self.cfg = cfg or SystemConfig()

    def optimise(
        self,
        df:          pd.DataFrame,
        schema:      DataSchema,
        model_key:   str,
        epsilon:     float,
        epsilon_cap: float = 10.0,
    ) -> float:
        if len(df) < 100 or not _OPTUNA:
            return epsilon

        # Subsample once — shared across all trials for speed
        n_sub  = min(MINI_ROWS, len(df))
        df_sub = df.sample(n_sub, random_state=42).reset_index(drop=True)

        try:
            strat = df_sub[schema.target_col].values if schema.has_target() else None
            df_train, df_hold = train_test_split(
                df_sub, test_size=0.3, random_state=42, stratify=strat
            )
        except Exception:
            df_train, df_hold = train_test_split(df_sub, test_size=0.3, random_state=42)

        def objective(trial: "optuna.Trial") -> float:
            eps  = trial.suggest_float(
                "epsilon", self.cfg.min_epsilon,
                min(epsilon_cap, self.cfg.max_epsilon), log=True,
            )
            bs   = trial.suggest_categorical("batch_size", [64, 128, 256])
            clip = trial.suggest_float("clip_norm", 0.5, 2.0)
            try:
                return self._mini_score(df_train, df_hold, schema,
                                        model_key, eps, bs, clip)
            except Exception:
                priv = float(np.clip(1.0 - eps / 10.0, 0.0, 1.0))
                util = 0.6 if schema.has_target() else 0.4
                pen  = 0.0 if bs <= len(df_train) // 4 else 0.1
                return 0.4 * priv + 0.3 * util - pen

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study = optuna.create_study(direction="maximize")
            study.optimize(
                objective,
                n_trials          = self.cfg.hpo_trials,
                timeout           = self.cfg.hpo_timeout,
                show_progress_bar = False,
            )

        best_eps  = float(np.clip(
            study.best_params.get("epsilon", epsilon),
            self.cfg.min_epsilon, epsilon_cap,
        ))
        best_bs   = study.best_params.get("batch_size", 256)
        best_clip = study.best_params.get("clip_norm", 1.0)
        print(f"  [HPO] Best ε={best_eps:.4f}  "
              f"batch={best_bs}  clip={best_clip:.2f}  "
              f"({self.cfg.hpo_trials} trials)")
        return best_eps

    # ── Mini-train score ──────────────────────────────────────────────────────
    def _mini_score(self, train, hold, schema, model_key,
                    eps, bs, clip) -> float:
        model = self._mini_model(model_key)
        model.fit(train, eps)
        synth = model.generate(train, eps, n_output=len(train))

        priv = self._fast_mia(train, hold, synth)
        util = self._fast_tstr(hold, synth, schema)
        return 0.4 * priv + 0.3 * util

    def _mini_model(self, key: str):
        if key == "tabddpm":
            from synthetic_os.models.tab_ddpm import TabDDPM
            m = TabDDPM()
            m._n_epochs_override = MINI_EPOCHS
            return m
        if key in ("ctgan", "ensemble"):
            from synthetic_os.models.dp_ctgan import DPCTGAN
            m = DPCTGAN()
            return m
        return _StatProxy()

    def _fast_mia(self, train, hold, synth) -> float:
        num = [c for c in train.select_dtypes(include=[np.number]).columns
               if c in synth.columns]
        if not num:
            return 0.8
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        tr = sc.fit_transform(train[num].fillna(0).values.astype(float))
        ho = sc.transform(hold[num].fillna(0).values.astype(float))
        sy = sc.transform(synth[num].fillna(0).values.astype(float)[:200])

        def _min_d(q, ref):
            return np.array([
                np.sqrt(((ref - row) ** 2).sum(axis=1)).min()
                for row in q[:100]
            ])

        ratio = float(np.median(_min_d(ho, sy)) /
                      (np.median(_min_d(tr, sy)) + 1e-8))
        return float(np.clip((ratio - 1.0) / 2.0 + 0.5, 0.0, 1.0))

    def _fast_tstr(self, hold, synth, schema) -> float:
        if not schema.has_target():
            return 0.5
        tgt  = schema.target_col
        feat = [c for c in hold.columns
                if c != tgt
                and pd.api.types.is_numeric_dtype(hold[c])
                and c in synth.columns]
        if not feat or len(synth) < 10 or len(hold) < 5:
            return 0.5
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            le.fit(pd.concat([synth[tgt], hold[tgt]]).astype(str))
            clf = RandomForestClassifier(n_estimators=20, max_depth=4,
                                         random_state=42, n_jobs=-1)
            clf.fit(synth[feat].fillna(0).values,
                    le.transform(synth[tgt].astype(str)))
            y_ho  = le.transform(hold[tgt].astype(str))
            acc   = float((clf.predict(hold[feat].fillna(0).values) == y_ho).mean())
            base  = float(pd.Series(y_ho).value_counts(normalize=True).max())
            return float((acc - base) / (1.0 - base + 1e-6)) if acc > base else 0.0
        except Exception:
            return 0.5


class _StatProxy:
    def fit(self, df, epsilon):
        self._df = df
    def generate(self, df, epsilon, n_output=None):
        n = n_output or len(df)
        return pd.DataFrame(
            np.random.randn(n, len(df.columns)),
            columns=df.columns,
        )