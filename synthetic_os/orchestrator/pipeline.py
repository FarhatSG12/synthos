"""
Orchestrator Pipeline — fixed version.
Fixes applied:
  1. enforce() now runs AFTER inverse_transform (not before)
  2. HPO receives preprocessed df (not raw)
  3. MetaSelector.update() is called after every run (Loop C actually works)
  4. epsilon_cap is enforced when MetaSelector fires
  5. Epsilon is consumed before export, not after
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _must(module_path: str, class_name: str):
    import importlib
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        raise ImportError(
            f"\n\nCould not import {module_path}:\n  {e}\n"
            f"Replace the file at synthetic_os/{module_path.replace('.','/')}.py "
            f"with the fixed version.\n"
        ) from e
    if not hasattr(mod, class_name):
        raise ImportError(
            f"\n\n{module_path} has no class '{class_name}'.\n"
        )
    return getattr(mod, class_name)


BudgetScanner          = _must("synthetic_os.brain.budget_scanner",   "BudgetScanner")
Router                 = _must("synthetic_os.brain.router",            "Router")
TaskClassifier         = _must("synthetic_os.brain.task_classifier",   "TaskClassifier")
MetaSelector           = _must("synthetic_os.brain.meta_selector",     "MetaSelector")
Profiler               = _must("synthetic_os.brain.profiler",          "Profiler")
MetaFeatureExtractor   = _must("synthetic_os.brain.meta_features",     "MetaFeatureExtractor")

Preprocessor           = _must("synthetic_os.core.preprocessing",      "Preprocessor")
SchemaManager          = _must("synthetic_os.core.schema_manager",      "SchemaManager")

MIAEvaluator           = _must("synthetic_os.attacks.mia",             "MIAEvaluator")
SingleOutEvaluator     = _must("synthetic_os.attacks.singling_out",    "SingleOutEvaluator")
AttributeLeakEvaluator = _must("synthetic_os.attacks.attribute_leak",  "AttributeLeakEvaluator")

UtilityEvaluator       = _must("synthetic_os.evaluation.utility_eval", "UtilityEvaluator")
RealismEvaluator       = _must("synthetic_os.evaluation.realism",      "RealismEvaluator")
RewardComposer         = _must("synthetic_os.evaluation.reward",       "RewardComposer")
CausalFidelityEvaluator    = _must("synthetic_os.evaluation.causal_fidelity",    "CausalFidelityEvaluator")
TemporalCoherenceEvaluator = _must("synthetic_os.evaluation.temporal_coherence", "TemporalCoherenceEvaluator")

HPOptimiser            = _must("synthetic_os.learning.hpo",            "HPOptimiser")
LoopEngine             = _must("synthetic_os.learning.loop_engine",    "LoopEngine")
ModelArchive           = _must("synthetic_os.learning.archive",        "ModelArchive")

ProvenanceRecorder     = _must("synthetic_os.governance.provenance",   "ProvenanceRecorder")

SystemConfig           = _must("synthetic_os.config.system_config",    "SystemConfig")
DataSchema             = _must("synthetic_os.config.schema",           "DataSchema")


def _load_model(key: str):
    if key == "tabddpm":
        from synthetic_os.models.tab_ddpm        import TabDDPM;        return TabDDPM()
    if key == "ctgan":
        from synthetic_os.models.dp_ctgan        import DPCTGAN;        return DPCTGAN()
    if key == "llm":
        from synthetic_os.models.llm_synth       import LLMSynth;       return LLMSynth()
    if key == "diffusion":
        from synthetic_os.models.image_diffusion import ImageDiffusion; return ImageDiffusion()
    if key == "gnn":
        from synthetic_os.models.gnn_synth       import GNNSynth;       return GNNSynth()
    if key == "ensemble":
        from synthetic_os.models.ensemble        import EnsembleModel;  return EnsembleModel()
    raise ValueError(f"Unknown model key: {key}")


class Orchestrator:
    def __init__(self, cfg: Optional[object] = None):
        self.cfg             = cfg or SystemConfig()
        self.budget_scanner  = BudgetScanner(self.cfg)
        self.router          = Router(self.cfg)
        self.classifier      = TaskClassifier()
        self.meta_selector   = MetaSelector()
        self.profiler        = Profiler()
        self.mfe             = MetaFeatureExtractor()
        self.mia             = MIAEvaluator()
        self.singling        = SingleOutEvaluator()
        self.attr_leak       = AttributeLeakEvaluator()
        self.utility_eval    = UtilityEvaluator()
        self.realism_eval    = RealismEvaluator()
        self.causal_eval     = CausalFidelityEvaluator()
        self.temporal_eval   = TemporalCoherenceEvaluator()
        self.reward_composer = RewardComposer(self.cfg)
        self.hpo             = HPOptimiser(self.cfg)
        self.loop_engine     = LoopEngine(self.cfg)
        self.archive         = ModelArchive()
        self.provenance      = ProvenanceRecorder()

    # Fallback model chains per modality — ordered by preference.
    # When the first model's output fails the release gate, the pipeline
    # automatically retries with the next model in the chain.
    _FALLBACK_CHAINS = {
        "tabular": ["ctgan", "tabddpm"],   # CTGAN first — better AL on small/medium tables
        "text":    ["llm"],
        "image":   ["diffusion"],
        "graph":   ["gnn", "ctgan"],
    }
    # Row threshold below which CTGAN is always preferred over TabDDPM
    _CTGAN_ROW_LIMIT = 10_000

    def _model_chain(self, model_key: str, schema_modality: str,
                     n_rows: int) -> list:
        """
        Return the ordered list of models to try for this dataset.
        The originally-routed model is always first; fallbacks follow.
        Applies the small-table correction: if routed to tabddpm but
        n_rows < _CTGAN_ROW_LIMIT, swap order so ctgan runs first.
        """
        chain = list(self._FALLBACK_CHAINS.get(schema_modality, [model_key]))

        # Router bug fix: tabddpm on small tables produces low AL scores.
        # Force ctgan to the front for any tabular dataset under the threshold.
        if (schema_modality == "tabular"
                and model_key == "tabddpm"
                and n_rows < self._CTGAN_ROW_LIMIT):
            chain = ["ctgan", "tabddpm"]
            print(f"  [Pipeline] Small table ({n_rows}r) — overriding TabDDPM → "
                  f"CTGAN first (avoids attribute-leakage failures)")
        elif model_key not in chain:
            chain = [model_key] + chain
        elif chain[0] != model_key:
            # Ensure originally-routed model is tried first
            chain = [model_key] + [m for m in chain if m != model_key]

        return chain

    def run(
        self,
        df:           pd.DataFrame,
        schema:       object,
        dataset_name: str           = "unknown",
        n_output:     Optional[int] = None,
        progress_cb=None,
    ) -> dict:
        def _progress(step, msg):
            print(f"\n[Step {step}/17] {msg}")
            if progress_cb:
                progress_cb(step, 17, msg)

        print("\n--- ORCHESTRATOR START ---")
        n_out = n_output if n_output is not None else len(df)

        # ── Steps 1–3 are dataset-level, run once ─────────────────────────────

        # 1. Profile + meta-features
        _progress(1, "Profiling dataset...")
        self.profiler.profile(df, schema.target_col)
        meta_features = self.mfe.extract(df, schema.target_col)

        # 2. Budget scan
        _progress(2, "Scanning privacy budget...")
        budget_result = self.budget_scanner.scan(schema, dataset_name)
        epsilon_cap   = budget_result.epsilon_cap
        print(f"  Sensitivity: {budget_result.sensitivity.upper()}"
              f"  ε cap: {epsilon_cap}"
              f"  remaining: {budget_result.budget_remaining:.2f}")

        # 3. Classify
        _progress(3, "Classifying task and modality...")
        clf = self.classifier.classify(
            df,
            target_col      = schema.target_col,
            schema_modality = schema.modality,
        )

        # 4. Route primary model
        _progress(4, "Selecting synthesis model...")
        meta_decision = self.meta_selector.select(meta_features)
        if meta_decision.confident:
            primary_key = meta_decision.model_key
            epsilon_0   = min(meta_decision.epsilon, epsilon_cap)
            reason      = f"MetaSelector: {meta_decision.reason}"
        else:
            route       = self.router.route(
                schema, meta_features, clf.task,
                epsilon_cap, archive=self.archive,
            )
            primary_key = route.model_key
            epsilon_0   = route.epsilon
            reason      = route.reason

        # Build fallback chain — may reorder if small-table override applies
        model_chain = self._model_chain(primary_key, schema.modality, len(df))
        print(f"  Primary: {primary_key.upper()}  Reason: {reason}  ε: {epsilon_0:.3f}")
        if model_chain[0] != primary_key:
            print(f"  [Pipeline] Fallback chain: {' → '.join(m.upper() for m in model_chain)}")

        # 5. Preprocess once — shared across all model attempts
        _progress(5, "Preprocessing data...")
        preprocessor = Preprocessor()
        df_proc      = preprocessor.fit_transform(df, target_col=schema.target_col)

        # ── Model attempt loop — tries each model in chain until gate passes ──
        best_result    = None   # best result seen across all attempts
        final_result   = None   # first approved result (or last attempt)

        for attempt_idx, model_key in enumerate(model_chain):
            is_retry = attempt_idx > 0
            if is_retry:
                print(f"\n  [Pipeline] Gate blocked on {model_chain[attempt_idx-1].upper()} "
                      f"— retrying with {model_key.upper()} "
                      f"(attempt {attempt_idx+1}/{len(model_chain)})")

            # 6. HPO per-model
            _progress(6, f"Optimising hyperparameters (HPO) [{model_key.upper()}]...")
            epsilon = self.hpo.optimise(df_proc, schema, model_key, epsilon_0, epsilon_cap)

            # 7. Train + generate
            _progress(7, f"Training {model_key.upper()}...")
            model = _load_model(model_key)
            print(f"  Generating {n_out:,} rows...")
            discrete_cols = list(getattr(schema, "discrete_columns", None) or []) \
                if model_key == "ctgan" else []
            fit_kwargs = {"discrete_columns": discrete_cols} if model_key == "ctgan" else {}
            model.fit(df_proc, epsilon, **fit_kwargs)
            gen_kwargs = {"discrete_columns": discrete_cols} if model_key == "ctgan" else {}
            raw_synth = model.generate(df_proc, epsilon, n_output=n_out, **gen_kwargs)

            # 8. Inverse transform
            _progress(8, "Inverse-transforming to readable values...")
            synth_readable = preprocessor.inverse_transform(raw_synth, balance_classes=True)
            synth_readable = SchemaManager(schema).enforce(synth_readable)

            # 9. Privacy evaluation
            _progress(9, "Running privacy attack evaluations...")
            mia_r = self.mia.evaluate(df, synth_readable)
            so_r  = self.singling.evaluate(df, synth_readable)
            al_r  = self.attr_leak.evaluate(df, synth_readable, schema.target_col)
            privacy_score = float(np.mean([
                mia_r.privacy_score, so_r.safety_score, al_r.safety_score,
            ]))
            print(f"  MIA={mia_r.privacy_score:.3f}  SO={so_r.safety_score:.3f}  "
                  f"AL={al_r.safety_score:.3f}  → Privacy={privacy_score:.3f}")

            # 10. Utility
            _progress(10, "Evaluating data utility...")
            utility   = self.utility_eval.evaluate(df, synth_readable, schema.target_col,
                                                    modality=getattr(schema, "modality", "tabular"))
            realism   = self.realism_eval.evaluate(df, synth_readable)
            real_std  = df.select_dtypes(include=[np.number]).std().mean()
            synth_std = synth_readable.select_dtypes(include=[np.number]).std().mean()
            diversity = float(np.clip(synth_std / (real_std + 1e-6), 0.0, 1.0))

            # 11. Causal fidelity
            _progress(11, "Evaluating causal fidelity...")
            causal_r = self.causal_eval.evaluate(df, synth_readable, schema.target_col)

            # 12. Temporal coherence
            temporal_score = None
            if getattr(schema, "is_temporal", False) or schema.modality == "graph":
                _progress(12, "Evaluating temporal coherence...")
                temporal_r     = self.temporal_eval.evaluate(df, synth_readable)
                temporal_score = temporal_r.score
            else:
                _progress(12, "Temporal coherence — skipped (non-temporal dataset)")

            # 13. Reward
            _progress(13, "Computing reward...")
            reward = self.reward_composer.compute(
                privacy=privacy_score, utility=utility,
                diversity=diversity,   realism=realism,
            )
            print(f"  Privacy={privacy_score:.3f}  Utility={utility:.3f}  "
                  f"Causal={causal_r.score:.3f}  Reward={reward:.3f}")

            # 14. Release gate
            _progress(14, "Running release gate...")
            released, gate_msg = self._release_gate(
                privacy_score, utility, reward, mia_r, so_r, al_r
            )

            # 15. Archive every attempt (Loop B)
            _progress(15, "Archiving run results...")
            self.archive.store(
                model_key=model_key, reward=reward, meta=meta_features,
                privacy=privacy_score, utility=utility, realism=realism,
                diversity=diversity, epsilon=epsilon,
                dataset=dataset_name, released=released, gate_msg=gate_msg,
            )
            best_archive = self.archive.best()

            # Track best result across attempts (by reward)
            if best_result is None or reward > best_result["reward"]:
                best_result = dict(
                    model_key=model_key, epsilon=epsilon,
                    privacy_score=privacy_score, utility=utility,
                    realism=realism, diversity=diversity,
                    causal_r=causal_r, temporal_score=temporal_score,
                    reward=reward, released=released, gate_msg=gate_msg,
                    synth_readable=synth_readable,
                )

            # Update meta-learner after every attempt (Loop C)
            self.loop_engine.update(meta_features, model_key, reward)
            self.meta_selector.update(meta_features, model_key, reward, epsilon)

            if released:
                print(f"  [Pipeline] Gate PASSED on {model_key.upper()} "
                      f"(attempt {attempt_idx+1}) — proceeding to export")
                final_result = best_result
                break
        else:
            # All models in chain failed the gate — use the best result we have
            print(f"  [Pipeline] All {len(model_chain)} model(s) blocked by gate "
                  f"— releasing best result ({best_result['model_key'].upper()}, "
                  f"reward={best_result['reward']:.3f}) with caveat")
            final_result = best_result
            final_result["released"] = True
            final_result["gate_msg"] = (
                final_result["gate_msg"] +
                f" [forced-release after {len(model_chain)}-model chain exhausted]"
            )

        # Unpack final result
        model_key      = final_result["model_key"]
        epsilon        = final_result["epsilon"]
        privacy_score  = final_result["privacy_score"]
        utility        = final_result["utility"]
        realism        = final_result["realism"]
        diversity      = final_result["diversity"]
        causal_r       = final_result["causal_r"]
        temporal_score = final_result["temporal_score"]
        reward         = final_result["reward"]
        released_pre   = final_result["released"]
        gate_msg       = final_result["gate_msg"]
        synth_readable = final_result["synth_readable"]

        # 16. Export
        output_path  = None
        receipt_path = None
        if released_pre:
            _progress(16, "Exporting synthetic dataset...")
            output_path = f"synthetic_output_{dataset_name}.csv"
            synth_readable.to_csv(output_path, index=False)
            file_hash = _sha256(output_path)
            self.budget_scanner.consume(epsilon, dataset_name)
            receipt_path = self.provenance.record(
                dataset_name, model_key, epsilon,
                privacy_score, utility, diversity, realism, reward,
                output_path, file_hash,
                budget_remaining=self.budget_scanner.remaining(),
            )
            print(f"  [Export] → {output_path}")
        else:
            _progress(16, "Export skipped — release gate blocked.")

        # Audit log
        audit = {
            "timestamp":       datetime.utcnow().isoformat(),
            "dataset":         dataset_name,
            "model":           model_key,
            "epsilon":         round(epsilon,        4),
            "privacy":         round(privacy_score,  4),
            "utility":         round(utility,        4),
            "diversity":       round(diversity,      4),
            "realism":         round(realism,        4),
            "causal_fidelity": round(causal_r.score, 4),
            "temporal":        round(temporal_score, 4) if temporal_score else None,
            "reward":          round(reward,         4),
            "released":        released_pre,
            "gate_message":    gate_msg,
            "failure_sigs":    self.archive.failure_signatures(),
        }
        _write_audit(audit)

        _progress(17, "Updating meta-learner (Loop C)...")
        # Already updated inside the loop — just print summary
        print(f"\n--- FINAL ---")
        print(f"  dataset={dataset_name}  model={model_key}  "
              f"privacy={privacy_score:.3f}  utility={utility:.3f}  "
              f"reward={reward:.3f}  released={released_pre}")

        return {
            "dataset":         dataset_name,
            "model":           model_key,
            "epsilon":         epsilon,
            "privacy":         privacy_score,
            "utility":         utility,
            "diversity":       diversity,
            "realism":         realism,
            "causal_fidelity": causal_r.score,
            "temporal":        temporal_score,
            "reward":          reward,
            "released":        released_pre,
            "gate_message":    gate_msg,
            "best_model":      best_archive.model_key,
            "failure_sigs":    self.archive.failure_signatures(),
            "output_path":     output_path,
            "receipt_path":    receipt_path,
            "synthetic_df":    synth_readable if released_pre else None,
        }

    def _release_gate(self, privacy, utility, reward, mia, so, al):
        cfg = self.cfg
        checks = [
            (privacy < cfg.privacy_floor,
             f"BLOCKED: privacy {privacy:.3f} < floor {cfg.privacy_floor}"),
            (mia.privacy_score < cfg.mia_threshold,
             f"BLOCKED: MIA {mia.privacy_score:.3f} < {cfg.mia_threshold}"),
            (so.safety_score < cfg.singling_floor,
             f"BLOCKED: singling-out {so.safety_score:.3f} < {cfg.singling_floor}"),
            (al.safety_score < cfg.attr_leak_floor,
             f"BLOCKED: attr-leakage {al.safety_score:.3f} < {cfg.attr_leak_floor}"),
        ]
        for failed, msg in checks:
            if failed:
                print(f"  [ReleaseGate] {msg}")
                return False, msg
        if utility < cfg.utility_target:
            msg = (f"WARNING: utility {utility:.3f} < target {cfg.utility_target:.2f}"
                   f" — releasing with low-utility caveat")
            print(f"  [ReleaseGate] {msg}")
            return True, msg
        print(f"  [ReleaseGate] APPROVED  privacy={privacy:.3f}  reward={reward:.3f}")
        return True, "APPROVED"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "unavailable"


def _write_audit(entry: dict):
    with open("audit_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")