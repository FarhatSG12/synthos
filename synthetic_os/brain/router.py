"""
Router — selects exactly ONE synthesis model per dataset
Now consults ModelArchive failure signatures (History Index — Loop B)
before committing to a model. If the chosen model has previously failed
on a similar data profile, the router falls back to the next best option.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from synthetic_os.learning.archive import ModelArchive

from synthetic_os.config.schema import DataSchema

ALL_MODELS = {"tabddpm", "ctgan", "llm", "diffusion", "gnn", "ensemble"}


@dataclass
class RoutingDecision:
    model_key:    str
    epsilon:      float
    reason:       str
    skip_models:  list
    fallback_used: bool = False


class Router:
    def __init__(self, cfg=None):
        self._cfg = cfg

    def route(
        self,
        schema:        DataSchema,
        meta_features: dict,
        task:          str,
        epsilon_cap:   float = 10.0,
        archive:       Optional["ModelArchive"] = None,
    ) -> RoutingDecision:

        modality    = getattr(schema, "modality", "tabular")
        rows        = meta_features.get("num_rows",  meta_features.get("n_rows",  500))
        cols        = meta_features.get("num_cols",  meta_features.get("n_cols",  10))
        imbalance   = meta_features.get("imbalance_ratio", 1.0)
        sparsity    = meta_features.get("sparsity", 0.0)
        is_temporal = getattr(schema, "is_temporal", False)

        # ── Primary routing decision ─────────────────────────────────────────
        chosen, epsilon, reason = self._primary_route(
            modality, rows, cols, imbalance, sparsity,
            is_temporal, epsilon_cap,
        )

        # ── Failure signature check (History Index) ──────────────────────────
        fallback_used = False
        if archive is not None:
            avoid, avoid_reason = archive.should_avoid(chosen, meta_features)
            if avoid:
                print(f"  [Router] History index: {avoid_reason}")
                print(f"  [Router] Switching away from {chosen}")
                chosen, epsilon, reason = self._fallback_route(
                    chosen, modality, rows, cols, imbalance, sparsity, epsilon_cap,
                )
                reason += f" [history fallback: {avoid_reason}]"
                fallback_used = True

        skip = list(ALL_MODELS - {chosen})
        print(f"  [Router] {reason}   ε_allocated={epsilon:.3f}")
        return RoutingDecision(
            model_key    = chosen,
            epsilon      = epsilon,
            reason       = reason,
            skip_models  = skip,
            fallback_used= fallback_used,
        )

    # ── Primary routing logic ────────────────────────────────────────────────
    def _primary_route(self, modality, rows, cols, imbalance, sparsity,
                       is_temporal, epsilon_cap):
        if modality == "image":
            return "diffusion", min(1.0, epsilon_cap), "Image modality → ImageDiffusion"
        if modality == "text":
            return "llm", min(3.0, epsilon_cap), "Text modality → LLM"
        if is_temporal or modality == "graph":
            return "gnn", min(1.0, epsilon_cap), "Temporal/graph → GNNSynth"
        if rows > 5000 and cols >= 50:
            return "tabddpm", min(0.7, epsilon_cap), \
                   f"Large/complex tabular ({rows}r × {cols}c) → TabDDPM"
        if imbalance > 3.0 or sparsity > 0.5:
            return "ctgan", min(1.0, epsilon_cap), \
                   f"Imbalanced/sparse (imb={imbalance:.1f}) → CTGAN"
        # FIX: was rows > 1000 — too low, sent small/medium tables to TabDDPM
        # which consistently produces low attribute-leakage scores on them.
        # Now requires BOTH a meaningful row count AND enough columns to
        # justify the extra complexity of TabDDPM over CTGAN.
        if rows > 10_000 and cols >= 10:
            return "tabddpm", min(0.7, epsilon_cap), \
                   f"Large tabular ({rows}r × {cols}c) → TabDDPM"
        return "ctgan", min(1.0, epsilon_cap), f"Small/medium tabular ({rows}r) → CTGAN"

    # ── Fallback routing when history says avoid primary choice ──────────────
    def _fallback_route(self, avoided, modality, rows, cols,
                        imbalance, sparsity, epsilon_cap):
        # Simple fallback ladder
        ladder = {
            "tabddpm":  "ctgan",
            "ctgan":    "tabddpm",
            "llm":      "ctgan",       # shouldn't happen but defensive
            "diffusion":"ctgan",
            "gnn":      "tabddpm",
            "ensemble": "tabddpm",
        }
        fallback = ladder.get(avoided, "ctgan")
        epsilon  = min(1.0, epsilon_cap)
        reason   = f"Fallback from {avoided} → {fallback}"
        return fallback, epsilon, reason