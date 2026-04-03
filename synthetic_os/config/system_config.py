"""
System Configuration — clinical AI defaults
Privacy settings are intentionally conservative.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    # ── Differential Privacy ─────────────────────────────────────────────────
    default_epsilon:  float = 0.5    # default ε if HPO not used
    min_epsilon:      float = 0.1    # never go below this
    max_epsilon:      float = 10.0   # hard cap for low-sensitivity data
    privacy_budget:   float = 3.0    # total cumulative ε budget across all runs
    clip_norm:        float = 1.0    # L2 sensitivity for DP mechanisms

    # ── Release Gate ─────────────────────────────────────────────────────────
    privacy_floor:    float = 0.70   # minimum combined privacy score
    utility_target:   float = 0.60   # below this → releases with warning
    mia_threshold:    float = 0.75   # minimum MIA resistance
    singling_floor:   float = 0.80   # minimum singling-out safety
    attr_leak_floor:  float = 0.75   # minimum attribute leakage safety

    # ── Reward Composition ───────────────────────────────────────────────────
    # Clinical AI focus: privacy outweighs utility
    reward_w_privacy:   float = 0.40
    reward_w_utility:   float = 0.30
    reward_w_realism:   float = 0.20
    reward_w_diversity: float = 0.10

    # ── HPO ──────────────────────────────────────────────────────────────────
    hpo_trials:       int   = 5      # Optuna trials per run
    hpo_timeout:      int   = 120    # seconds; None = unlimited

    # ── Training ─────────────────────────────────────────────────────────────
    tabddpm_epochs:   int   = 80
    ctgan_epochs:     int   = 150
    batch_size:       int   = 256

    # ── Meta-learning ────────────────────────────────────────────────────────
    meta_min_samples: int   = 3      # runs before MetaSelector fires