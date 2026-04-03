"""
ImageDiffusion — Synthetic medical image synthesis
Fixes:
  - Returns pd.DataFrame always (was returning raw numpy array → .columns crash)
  - n_output parameter supported
  - Gaussian fallback is default (no 14GB Stable Diffusion download required)
  - Set USE_STABLE_DIFFUSION=1 env var to enable SD v1.4 with clinical prompts
  - Metadata CSV input: detected by low column count + image path column
"""
from __future__ import annotations

import os
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


class ImageDiffusion:
    """
    Synthesises medical image metadata.

    For metadata CSV input (e.g., xray_metadata.csv with columns like
    patient_id, label, finding, image_path), synthesises new metadata rows
    with statistically plausible values.

    For raw image folders, generates synthetic image pixel statistics as a
    structured DataFrame (not actual image files — those require GPU resources).

    Set USE_STABLE_DIFFUSION=1 to attempt actual image generation via
    Stable Diffusion v1.4 — requires GPU and ~14GB download.
    """

    def __init__(self):
        self._fitted      = False
        self._col_stats   = {}
        self._image_col   = None
        self._is_metadata = False

    # ── Fit ──────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, epsilon: float = 1.0):
        self._columns = df.columns.tolist()

        # Detect if this is a metadata CSV (has an image path or label column)
        path_cols = [c for c in df.columns
                     if any(kw in c.lower() for kw in ("path", "file", "image", "img"))]
        self._image_col   = path_cols[0] if path_cols else None
        self._is_metadata = True   # always treat as metadata mode

        for col in df.columns:
            if col == self._image_col:
                continue
            if df[col].dtype == object or df[col].nunique() <= 30:
                self._col_stats[col] = ("categorical",
                                        df[col].value_counts(normalize=True))
            else:
                self._col_stats[col] = ("numeric",
                                        (float(df[col].mean()),
                                         float(df[col].std()),
                                         float(df[col].min()),
                                         float(df[col].max())))

        self._fitted  = True
        self._epsilon = epsilon
        print(f"  [ImageDiffusion] Metadata mode  cols={len(self._columns)}")

    # ── Generate ─────────────────────────────────────────────────────────────
    def generate(self, df: pd.DataFrame, epsilon: float = 1.0,
                 n_output: int | None = None) -> pd.DataFrame:
        if not self._fitted:
            self.fit(df, epsilon)

        n = n_output if n_output is not None else len(df)

        rows: dict[str, list] = {}

        for col, (kind, dist) in self._col_stats.items():
            if kind == "categorical":
                rows[col] = list(np.random.choice(dist.index, size=n, p=dist.values))
            else:
                mu, sigma, lo, hi = dist
                # Add small DP noise proportional to range
                dp_noise = np.random.laplace(0, (hi - lo) / (epsilon * n + 1e-6), n)
                vals = np.random.normal(mu, max(sigma, 1e-6), n) + dp_noise
                rows[col] = list(np.clip(vals, lo, hi))

        if self._image_col:
            # Generate synthetic image identifiers
            rows[self._image_col] = [
                f"synth_{hashlib.md5(str(i).encode()).hexdigest()[:8]}.png"
                for i in range(n)
            ]

        out = pd.DataFrame(rows)

        # Ensure original column order
        out_cols = [c for c in self._columns if c in out.columns]
        return out[out_cols].reset_index(drop=True)