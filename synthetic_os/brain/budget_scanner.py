"""
Budget Scanner
Classifies dataset sensitivity level and assigns an epsilon cap.
Tracks total privacy budget consumed across runs.
Blocks generation if budget is exhausted.

Sensitivity → epsilon cap:
  CRITICAL  → 0.5
  HIGH      → 1.0
  MEDIUM    → 3.0
  LOW       → 10.0
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BUDGET_LOG = Path("budget_tracker.json")

SENSITIVITY_KEYWORDS = {
    "critical": [
        "ssn", "social_security", "diagnosis", "icd", "patient_id",
        "encounter_id", "mrn", "dob", "date_of_birth", "insurance",
        "genetic", "hiv", "mental", "psychiatr", "substance",
    ],
    "high": [
        "name", "age", "gender", "race", "ethnicity", "zip", "address",
        "medication", "drug", "lab", "glucose", "blood", "pressure",
        "cholesterol", "weight", "bmi", "height", "readmit",
    ],
    "medium": [
        "hospital", "visit", "procedure", "admission", "discharge",
        "specialty", "payer", "insurance_type",
    ],
}

EPSILON_CAPS = {
    "critical": 0.5,
    "high":     1.0,
    "medium":   3.0,
    "low":      10.0,
}


@dataclass
class BudgetScanResult:
    sensitivity:   str
    epsilon_cap:   float
    budget_remaining: float


class BudgetScanner:
    def __init__(self, cfg=None):
        self._cfg          = cfg
        self._total_budget = getattr(cfg, "privacy_budget", 3.0) if cfg else 3.0
        self._consumed     = self._load_consumed()

    def scan(self, schema, dataset_name: str = "") -> BudgetScanResult:
        columns = getattr(schema, "columns", [])
        col_str = " ".join(c.lower() for c in columns)

        # Determine sensitivity
        sensitivity = "low"
        for level in ("critical", "high", "medium"):
            if any(kw in col_str for kw in SENSITIVITY_KEYWORDS[level]):
                sensitivity = level
                break

        epsilon_cap = EPSILON_CAPS[sensitivity]
        remaining   = max(0.0, self._total_budget - self._consumed)

        print(f"  [BudgetScanner] sensitivity={sensitivity.upper()}"
              f"  ε_cap={epsilon_cap}  budget_remaining={remaining:.2f}")

        if remaining <= 0:
            raise RuntimeError(
                "Privacy budget exhausted. No further generation permitted."
            )

        return BudgetScanResult(
            sensitivity      = sensitivity,
            epsilon_cap      = epsilon_cap,
            budget_remaining = remaining,
        )

    def consume(self, epsilon: float, dataset_name: str = ""):
        self._consumed += epsilon
        log = self._load_log()
        log.append({"dataset": dataset_name, "epsilon": epsilon,
                    "cumulative": self._consumed})
        BUDGET_LOG.write_text(json.dumps(log, indent=2))

    def remaining(self) -> float:
        return max(0.0, self._total_budget - self._consumed)

    def _load_consumed(self) -> float:
        log = self._load_log()
        return sum(e.get("epsilon", 0.0) for e in log)

    def reset(self):
        """Wipe the budget log. Called once per app session on startup."""
        self._consumed = 0.0
        BUDGET_LOG.write_text(json.dumps([]))
        print("  [BudgetScanner] Budget log reset for new session.")

    def _load_log(self) -> list:
        if BUDGET_LOG.exists():
            try:
                return json.loads(BUDGET_LOG.read_text())
            except Exception:
                return []
        return []