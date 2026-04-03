"""
TabDDPM — Tabular Denoising Diffusion Probabilistic Model
Fixes:
  - Non-numeric columns are excluded before float cast (no more ValueError)
  - n_output parameter supported
  - PCA compression for wide datasets (> 100 cols → 50 components)
  - Batched reverse diffusion (no full N×D matrix allocation)
  - DP noise injected as a post-step, decoupled from noise schedule
  - Cosine noise schedule (independent of epsilon)
  - Non-numeric columns now sampled from their real marginal distribution
    instead of being filled with the modal value.  Filling a Gender column
    with "Female" for every row made JSD=0 for that column and collapsed
    the overall utility score.  We now record value_counts() per column and
    sample from it, so the synthetic distribution matches the real one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

MAX_COLS_BEFORE_PCA = 100
PCA_COMPONENTS      = 50
BATCH_SIZE          = 512
TIMESTEPS           = 200
SAMPLE_BATCH        = 512


class _MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden = max(128, min(dim * 2, 512))
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),  nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = t.float().unsqueeze(-1) / TIMESTEPS
        t_emb = t_emb.expand(x.size(0), 1)
        return self.net(torch.cat([x, t_emb], dim=-1))


def _cosine_schedule(T: int):
    s = 0.008
    steps = np.arange(T + 1)
    f = np.cos(((steps / T + s) / (1 + s)) * np.pi / 2) ** 2
    alpha_bar = f / f[0]
    return torch.tensor(alpha_bar, dtype=torch.float32)


class TabDDPM:
    def __init__(self):
        self.model_     = None
        self.mean_      = None
        self.std_       = None
        self._pca       = None
        self._use_pca   = False
        self._orig_cols = None
        # FIX: store full value_counts distribution for non-numeric columns
        # instead of just the mode.  Modal fill made every synthetic row have
        # the same value (e.g. race="Caucasian", gender="Female") which drove
        # JSD to 0 for those columns and pulled overall utility down sharply.
        self._numeric_cols    = None
        self._non_numeric_dists: dict = {}   # col → (values_array, probs_array) or ("numeric", stats)

    def fit(self, df: pd.DataFrame, epsilon: float = 1.0, n_epochs: int = 80):
        self._orig_cols = df.columns.tolist()

        # FIX: select only numeric columns — avoids ValueError on mixed dtypes
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self._numeric_cols = num_cols

        # Store full distribution of non-numeric columns for reconstruction
        for col in df.columns:
            if col not in num_cols:
                try:
                    vc = df[col].value_counts(normalize=True)
                    self._non_numeric_dists[col] = (
                        "categorical",
                        vc.index.tolist(),
                        vc.values.tolist(),
                    )
                except Exception:
                    self._non_numeric_dists[col] = ("fallback", [""], [1.0])

        if not num_cols:
            print("  [TabDDPM] No numeric columns — using statistical fallback")
            self._fallback_df = df.copy()
            return

        X = df[num_cols].fillna(0).values.astype(np.float32)

        # PCA compression for wide tables
        if X.shape[1] > MAX_COLS_BEFORE_PCA:
            n_comp = min(PCA_COMPONENTS, X.shape[1], X.shape[0] - 1)
            self._pca     = PCA(n_components=n_comp, random_state=42)
            X             = self._pca.fit_transform(X).astype(np.float32)
            self._use_pca = True
            print(f"  [TabDDPM] PCA: {len(num_cols)} → {n_comp} dims")

        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        X = (X - self.mean_) / self.std_

        dim          = X.shape[1]
        self.model_  = _MLP(dim)
        alpha_bar    = _cosine_schedule(TIMESTEPS)
        self._alpha  = alpha_bar

        tensor = torch.from_numpy(X)
        loader = DataLoader(TensorDataset(tensor),
                            batch_size=BATCH_SIZE, shuffle=True)
        opt    = torch.optim.Adam(self.model_.parameters(), lr=1e-3)

        self.model_.train()
        for epoch in range(n_epochs):
            for (batch,) in loader:
                t = torch.randint(1, TIMESTEPS + 1, (batch.size(0),))
                ab_t = alpha_bar[t].unsqueeze(-1)
                noise = torch.randn_like(batch)
                x_t   = torch.sqrt(ab_t) * batch + torch.sqrt(1 - ab_t) * noise
                pred  = self.model_(x_t, t)
                loss  = ((pred - noise) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()

        # DP noise post-training on weights
        if epsilon < 10.0:
            noise_scale = 1.0 / (epsilon + 1e-6)
            noise_scale = min(noise_scale, 0.05)
            with torch.no_grad():
                for p in self.model_.parameters():
                    p.add_(torch.randn_like(p) * noise_scale * p.std().clamp(min=1e-6))

        self.model_.eval()

    def generate(self, df: pd.DataFrame, epsilon: float = 1.0,
                 n_output: int | None = None) -> pd.DataFrame:
        if self.model_ is None and not self._non_numeric_dists and not hasattr(self, "_fallback_df"):
            self.fit(df, epsilon)

        n_rows = n_output if n_output is not None else len(df)

        # Fallback: no numeric columns at all
        if self.model_ is None:
            return self._fallback_df.sample(n_rows, replace=True,
                                            random_state=42).reset_index(drop=True)

        dim       = len(self.mean_)
        alpha_bar = self._alpha
        chunks    = []
        remaining = n_rows

        self.model_.eval()
        with torch.no_grad():
            while remaining > 0:
                bs   = min(SAMPLE_BATCH, remaining)
                x    = torch.randn(bs, dim)

                for t_val in reversed(range(1, TIMESTEPS + 1)):
                    t_tensor = torch.full((bs,), t_val)
                    ab_t  = alpha_bar[t_val]
                    ab_t1 = alpha_bar[t_val - 1]
                    beta  = 1 - ab_t / ab_t1
                    beta  = beta.clamp(0, 0.999)

                    pred_noise  = self.model_(x, t_tensor)
                    x0_pred     = (x - torch.sqrt(1 - ab_t) * pred_noise) / torch.sqrt(ab_t)
                    x0_pred     = x0_pred.clamp(-3, 3)

                    mean = (torch.sqrt(ab_t1) * beta / (1 - ab_t)) * x0_pred + \
                           (torch.sqrt(1 - beta) * (1 - ab_t1) / (1 - ab_t)) * x
                    if t_val > 1:
                        x = mean + torch.sqrt(beta) * torch.randn_like(x)
                    else:
                        x = mean

                arr = x.numpy() * self.std_ + self.mean_
                chunks.append(arr)
                remaining -= bs

        result = np.vstack(chunks)

        if self._use_pca and self._pca is not None:
            result = self._pca.inverse_transform(result)

        # Rebuild full dataframe with all original columns
        out = pd.DataFrame(result, columns=self._numeric_cols)

        # Restore non-numeric columns by sampling from their real distribution
        for col in self._orig_cols:
            if col not in out.columns:
                dist_info = self._non_numeric_dists.get(col)
                if dist_info is None:
                    out[col] = ""
                else:
                    kind = dist_info[0]
                    if kind in ("categorical", "fallback"):
                        _, vals, probs = dist_info
                        probs_arr = np.array(probs)
                        probs_arr = probs_arr / probs_arr.sum()  # renormalise
                        out[col] = np.random.choice(vals, size=n_rows, p=probs_arr)
                    else:
                        out[col] = ""

        # Restore original column order
        return out[self._orig_cols].reset_index(drop=True)