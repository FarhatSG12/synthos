"""
TaskClassifier — determines task type and data modality
Fixes:
  - schema_modality parameter short-circuits heuristics (prevents metadata CSV → image misroute)
  - Image heuristic additionally requires values in [0, 255] integer range
  - Text heuristic checks for long string columns correctly
  - Returns modality alongside task type
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassificationResult:
    task:     str   # classification | regression | generation
    modality: str   # tabular | text | image | graph


class TaskClassifier:
    def classify(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        schema_modality: Optional[str] = None,   # ← from registry; takes priority
    ) -> ClassificationResult:

        # Registry modality always wins — prevents OHE-expanded tabular being
        # mis-classified as image just because it has many numeric columns
        if schema_modality and schema_modality in ("tabular", "text", "image", "graph"):
            modality = schema_modality
        else:
            modality = self._infer_modality(df)

        task = self._infer_task(df, target_col, modality)
        print(f"  [Classifier] task={task}  schema_modality={modality}")
        return ClassificationResult(task=task, modality=modality)

    # ── Private helpers ──────────────────────────────────────────────────────
    def _infer_modality(self, df: pd.DataFrame) -> str:
        n_cols = len(df.columns)
        n_rows = len(df)

        # Text: any column with mean string length > 50
        text_cols = [c for c in df.columns
                     if df[c].dtype == object
                     and df[c].dropna().str.len().mean() > 50]
        if text_cols:
            return "text"

        # Image: many numeric columns AND values look like pixel intensities [0,255]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if n_cols > 200 and len(numeric_cols) / n_cols > 0.95:
            sample = df[numeric_cols].values.flatten()
            sample = sample[~np.isnan(sample)][:5000]
            if (sample.min() >= 0 and sample.max() <= 255 and
                    np.median(np.abs(sample - sample.round())) < 0.01):
                return "image"

        return "tabular"

    def _infer_task(self, df: pd.DataFrame, target_col: Optional[str],
                    modality: str) -> str:
        if modality in ("text", "image"):
            return "generation"
        if target_col is None or target_col not in df.columns:
            return "generation"
        n_unique = df[target_col].nunique()
        if n_unique <= 20:
            return "classification"
        return "regression"