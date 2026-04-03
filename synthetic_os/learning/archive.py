"""
Model Archive — Loop B
Stores every run result with full metrics AND failure signatures.
Persists to archive_log.jsonl so history survives across sessions.

Failure signatures enable the router to avoid models that previously
failed on a similar data profile, completing the History Index node
from the workflow diagram.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path

ARCHIVE_LOG = Path("archive_log.jsonl")


@dataclass
class ArchiveEntry:
    model_key:        str
    reward:           float
    meta:             dict
    privacy:          float = 0.0
    utility:          float = 0.0
    realism:          float = 0.0
    diversity:        float = 0.0
    epsilon:          float = 0.0
    dataset:          str   = ""
    released:         bool  = False
    failure_reasons:  list  = field(default_factory=list)
    timestamp:        str   = ""

    def has_failure(self) -> bool:
        return len(self.failure_reasons) > 0

    def failure_summary(self) -> str:
        return "; ".join(self.failure_reasons) if self.failure_reasons else "none"


def _detect_failures(
    privacy:   float,
    utility:   float,
    realism:   float,
    released:  bool,
    gate_msg:  str,
    model_key: str,
    meta:      dict,
) -> list:
    reasons = []
    n_cols   = meta.get("num_cols", meta.get("n_cols", 0))
    sparsity = meta.get("sparsity", 0.0)

    if not released:
        if "privacy" in gate_msg.lower():
            reasons.append(f"privacy_gate_fail:{model_key}")
        if "singling" in gate_msg.lower():
            reasons.append(f"singling_out_fail:{model_key}")
        if "attribute" in gate_msg.lower():
            reasons.append(f"attr_leak_fail:{model_key}")

    if utility < 0.40:
        if model_key == "tabddpm" and n_cols > 500:
            reasons.append("tabddpm_high_dim_utility_collapse")
        elif model_key == "ctgan" and sparsity > 0.6:
            reasons.append("ctgan_sparse_utility_collapse")
        elif model_key == "llm":
            reasons.append("llm_template_low_utility")
        else:
            reasons.append(f"low_utility:{model_key}")

    if realism < 0.40:
        reasons.append(f"low_realism:{model_key}")
    if privacy < 0.70:
        reasons.append(f"low_privacy:{model_key}")

    return reasons


class ModelArchive:
    def __init__(self):
        self._entries: list[ArchiveEntry] = self._load()

    def store(
        self,
        model_key: str,
        reward:    float,
        meta:      dict,
        privacy:   float = 0.0,
        utility:   float = 0.0,
        realism:   float = 0.0,
        diversity: float = 0.0,
        epsilon:   float = 0.0,
        dataset:   str   = "",
        released:  bool  = False,
        gate_msg:  str   = "",
    ) -> ArchiveEntry:
        # Guard: nan/inf reward corrupts meta-learner averages and max() calls
        if not isinstance(reward, (int, float)) or math.isnan(reward) or math.isinf(reward):
            print(f"  [Archive] WARNING: reward={reward} for {model_key} "
                  f"is not finite — clamping to 0.0")
            reward = 0.0
        failures = _detect_failures(
            privacy, utility, realism, released, gate_msg, model_key, meta
        )
        entry = ArchiveEntry(
            model_key       = model_key,
            reward          = reward,
            meta            = meta,
            privacy         = privacy,
            utility         = utility,
            realism         = realism,
            diversity       = diversity,
            epsilon         = epsilon,
            dataset         = dataset,
            released        = released,
            failure_reasons = failures,
            timestamp       = datetime.now(tz=timezone.utc).isoformat(),
        )
        self._entries.append(entry)
        self._persist(entry)

        tag = f"  Failures: {entry.failure_summary()}" if failures else "  Clean run"
        print(f"  [Archive] Stored → {model_key}  Reward: {reward:.4f}{tag}")
        return entry

    def best(self) -> ArchiveEntry:
        if not self._entries:
            return ArchiveEntry("none", 0.0, {})
        b = max(self._entries, key=lambda e: e.reward)
        print(f"  [Archive] Best found → {b.model_key}  Reward: {b.reward:.4f}")
        return b

    def failure_signatures(self) -> dict:
        """Return {model_key: [failure_reasons]} aggregated across all runs."""
        sigs: dict = {}
        for e in self._entries:
            if e.failure_reasons:
                sigs.setdefault(e.model_key, []).extend(e.failure_reasons)
        return sigs

    def should_avoid(self, model_key: str, meta: dict) -> tuple:
        """
        Returns (True, reason) if this model previously failed on a similar profile.
        Called by the router to incorporate History Index failure signatures.
        """
        n_cols   = meta.get("num_cols", meta.get("n_cols", 0))
        sparsity = meta.get("sparsity", 0.0)

        for e in self._entries:
            if e.model_key != model_key or not e.failure_reasons:
                continue
            if "tabddpm_high_dim_utility_collapse" in e.failure_reasons and n_cols > 400:
                return True, "tabddpm previously collapsed on high-dim data"
            if "ctgan_sparse_utility_collapse" in e.failure_reasons and sparsity > 0.5:
                return True, "ctgan previously collapsed on sparse data"
            if f"privacy_gate_fail:{model_key}" in e.failure_reasons:
                return True, f"{model_key} previously failed the privacy gate"
        return False, ""

    def all_entries(self) -> list:
        return list(self._entries)

    def _persist(self, entry: ArchiveEntry):
        record = {k: v for k, v in asdict(entry).items() if k != "meta"}
        record["meta_rows"] = entry.meta.get("num_rows", entry.meta.get("n_rows", 0))
        record["meta_cols"] = entry.meta.get("num_cols", entry.meta.get("n_cols", 0))
        with open(ARCHIVE_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _load(self) -> list:
        if not ARCHIVE_LOG.exists():
            return []
        entries = []
        for line in ARCHIVE_LOG.read_text().splitlines():
            try:
                d = json.loads(line)
                raw_reward = d.get("reward", 0.0)
                # Sanitize nan/inf that may exist in old log entries
                if not isinstance(raw_reward, (int, float)) or math.isnan(raw_reward) or math.isinf(raw_reward):
                    raw_reward = 0.0
                entries.append(ArchiveEntry(
                    model_key       = d.get("model_key", ""),
                    reward          = raw_reward,
                    meta            = {},
                    privacy         = d.get("privacy", 0.0),
                    utility         = d.get("utility", 0.0),
                    realism         = d.get("realism", 0.0),
                    diversity       = d.get("diversity", 0.0),
                    epsilon         = d.get("epsilon", 0.0),
                    dataset         = d.get("dataset", ""),
                    released        = d.get("released", False),
                    failure_reasons = d.get("failure_reasons", []),
                    timestamp       = d.get("timestamp", ""),
                ))
            except Exception:
                pass
        return entries