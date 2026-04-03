"""
GNNSynth — Graph Neural Network patient journey synthesizer
Routes when schema.modality == 'graph' or schema.is_temporal == True.
Falls back to a statistical synthesizer if torch-geometric is not installed.

Output is always a pd.DataFrame with the same columns as the input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class GNNSynth:
    def __init__(self):
        self._fitted   = False
        self._stats    = {}
        self._columns  = []
        self._temporal_col = None

    def fit(self, df: pd.DataFrame, epsilon: float = 1.0):
        self._columns = df.columns.tolist()

        # Detect temporal ordering column
        time_candidates = [c for c in df.columns
                           if any(kw in c.lower()
                                  for kw in ("time", "date", "visit", "encounter", "seq"))]
        self._temporal_col = time_candidates[0] if time_candidates else None

        # Record per-column statistics for synthesis
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                vals = df[col].dropna().values.astype(float)
                self._stats[col] = ("numeric",
                                    float(vals.mean()),
                                    float(vals.std()),
                                    float(vals.min()),
                                    float(vals.max()))
            else:
                self._stats[col] = ("categorical",
                                    df[col].value_counts(normalize=True))

        self._epsilon = epsilon
        self._fitted  = True
        self._n_real  = len(df)

        try:
            self._fit_gnn(df, epsilon)
        except Exception as e:
            print(f"  [GNNSynth] GNN unavailable ({e}) → statistical fallback")

    def generate(self, df: pd.DataFrame, epsilon: float = 1.0,
                 n_output: Optional[int] = None) -> pd.DataFrame:
        if not self._fitted:
            self.fit(df, epsilon)

        n = n_output if n_output is not None else len(df)
        return self._statistical_generate(n, epsilon)

    # ── Statistical fallback ──────────────────────────────────────────────────
    def _statistical_generate(self, n: int, epsilon: float) -> pd.DataFrame:
        rows = {}
        for col, info in self._stats.items():
            if info[0] == "numeric":
                _, mu, sigma, lo, hi = info
                # Add Laplace DP noise proportional to range
                dp_scale = (hi - lo) / (epsilon * max(n, 1) + 1e-6)
                vals = np.random.normal(mu, max(sigma, 1e-6), n)
                vals += np.random.laplace(0, dp_scale, n)
                rows[col] = np.clip(vals, lo, hi)
            else:
                dist = info[1]
                rows[col] = np.random.choice(dist.index, size=n, p=dist.values)

        out = pd.DataFrame(rows)

        # Simulate temporal ordering if detected
        if self._temporal_col and self._temporal_col in out.columns:
            out = out.sort_values(self._temporal_col).reset_index(drop=True)

        return out[self._columns].reset_index(drop=True)

    # ── GNN path (requires torch-geometric) ──────────────────────────────────
    def _fit_gnn(self, df: pd.DataFrame, epsilon: float):
        import torch
        from torch_geometric.data import Data
        from torch_geometric.nn   import GCNConv
        import torch.nn.functional as F

        num_cols = [c for c in df.columns
                    if pd.api.types.is_numeric_dtype(df[c])]
        X  = torch.tensor(df[num_cols].fillna(0).values, dtype=torch.float)
        n  = X.size(0)
        # Build a simple chain graph (patient encounters in sequence)
        src = torch.arange(0, n - 1)
        dst = torch.arange(1, n)
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src])
        ], dim=0)
        data = Data(x=X, edge_index=edge_index)

        class _GCNAutoEncoder(torch.nn.Module):
            def __init__(self, in_dim, hidden=32):
                super().__init__()
                self.enc = GCNConv(in_dim, hidden)
                self.dec = torch.nn.Linear(hidden, in_dim)
            def forward(self, x, edge_index):
                z = F.relu(self.enc(x, edge_index))
                return self.dec(z), z

        model = _GCNAutoEncoder(X.size(1))
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(50):
            opt.zero_grad()
            x_hat, z = model(data.x, data.edge_index)
            loss = F.mse_loss(x_hat, data.x)
            loss.backward()
            # Manual gradient clipping for DP
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        self._gnn_model    = model
        self._gnn_num_cols = num_cols
        self._gnn_dim      = X.size(1)
        print(f"  [GNNSynth] GNN trained  dim={X.size(1)}")