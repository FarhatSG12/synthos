"""
Provenance Recorder
Generates a JSON-LD receipt alongside every approved synthetic dataset.
Records: model, epsilon, all scores, release decision, SHA-256 hash, budget.
"""
from __future__ import annotations

import math
import json
from datetime import datetime, timezone
from pathlib import Path


class ProvenanceRecorder:
    # Use pathlib so this works on both Windows and Linux (Streamlit Cloud)
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "provenance"

    def __init__(self):
        self.OUTPUT_DIR.mkdir(exist_ok=True)

    @staticmethod
    def _safe_float(v: float, fallback: float = 0.0) -> float:
        """Return v if finite, else fallback — prevents json serialisation errors."""
        return fallback if (math.isnan(v) or math.isinf(v)) else round(float(v), 4)

    def record(
        self,
        dataset_name:     str,
        model_key:        str,
        epsilon:          float,
        privacy_score:    float,
        utility_score:    float,
        diversity_score:  float,
        realism_score:    float,
        reward:           float,
        output_path:      str,
        file_hash:        str,
        budget_remaining: float = 0.0,
    ) -> str:
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        sf = self._safe_float
        receipt = {
            "@context":        "https://schema.org/",
            "@type":           "Dataset",
            "name":            f"synthetic_{dataset_name}",
            "dateCreated":     timestamp,
            "generator":       {
                "@type":       "SoftwareApplication",
                "name":        "SynthOS",
                "version":     "2.0",
            },
            "privacyMechanism": {
                "name":        "Differential Privacy",
                "epsilon":     sf(epsilon),
                "budgetRemaining": sf(budget_remaining),
            },
            "synthesisModel":  model_key.upper(),
            "qualityMetrics":  {
                "privacy":    sf(privacy_score),
                "utility":    sf(utility_score),
                "diversity":  sf(diversity_score),
                "realism":    sf(realism_score),
                "reward":     sf(reward),
            },
            "releaseDecision": "APPROVED",
            "outputFile":      {
                "path":        output_path,
                "sha256":      file_hash,
            },
        }

        safe_name = dataset_name.replace("/", "_").replace("\\", "_")
        ts_short  = timestamp[:19].replace(":", "-").replace("T", "_")
        out_path  = self.OUTPUT_DIR / f"receipt_{safe_name}_{ts_short}.json"
        out_path.write_text(json.dumps(receipt, indent=2))
        print(f"  [Provenance] Receipt written → {out_path}")
        return str(out_path)