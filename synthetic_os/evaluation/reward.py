"""
Reward Composer
Clinical AI weights:
  Privacy   0.40  (highest — primary focus)
  Utility   0.30
  Realism   0.20
  Diversity 0.10
"""
from __future__ import annotations

import math
import numpy as np


class RewardComposer:
    W_PRIVACY   = 0.40
    W_UTILITY   = 0.30
    W_REALISM   = 0.20
    W_DIVERSITY = 0.10

    # Fallback values used when a component score is nan/inf
    # (e.g. ImageDiffusion returns nan utility in metadata-only mode)
    _FALLBACKS = {
        "privacy":   0.5,
        "utility":   0.0,
        "realism":   0.5,
        "diversity": 0.5,
    }

    def __init__(self, cfg=None):
        pass   # cfg reserved for future weight overrides

    @staticmethod
    def _safe(value: float, name: str, fallbacks: dict) -> float:
        """Return value if finite, else the named fallback. Logs a warning."""
        if math.isnan(value) or math.isinf(value):
            fb = fallbacks.get(name, 0.0)
            print(f"  [RewardComposer] WARNING: {name}={value} is not finite "
                  f"— substituting fallback {fb:.3f}")
            return fb
        return float(value)

    def compute(
        self,
        privacy:   float,
        utility:   float,
        diversity: float,
        realism:   float,
    ) -> float:
        p = self._safe(privacy,   "privacy",   self._FALLBACKS)
        u = self._safe(utility,   "utility",   self._FALLBACKS)
        r = self._safe(realism,   "realism",   self._FALLBACKS)
        d = self._safe(diversity, "diversity", self._FALLBACKS)

        reward = (
            self.W_PRIVACY   * p +
            self.W_UTILITY   * u +
            self.W_REALISM   * r +
            self.W_DIVERSITY * d
        )

        result = float(np.clip(reward, 0.0, 1.0))

        # Final sanity check — should never trigger after the guards above,
        # but defends against edge cases in np.clip itself
        if math.isnan(result) or math.isinf(result):
            print(f"  [RewardComposer] CRITICAL: reward still non-finite after "
                  f"guards — returning 0.0")
            return 0.0

        return result