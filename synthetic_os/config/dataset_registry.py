"""
Dataset Registry
Fixes:
  - patient_journey now points to patient_journey.csv (has encounter_sequence
    and time_in_hospital columns that GNNSynth can detect)
  - BASE path uses .parent.parent (correct for synthetic_os/config/ location)
  - xray: auto-generates minimal metadata.csv if missing
  - All paths validated at get_dataset() time
"""
from __future__ import annotations

import os
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BASE = Path(__file__).resolve().parent.parent / "data"


@dataclass
class DatasetEntry:
    name:        str
    path:        str
    target_col:  Optional[str]
    modality:    str
    is_temporal: bool = False
    description: str  = ""
    sensitivity: str  = "medium"
    row_count:   str  = ""


_REGISTRY: dict[str, DatasetEntry] = {

    "heart": DatasetEntry(
        name        = "heart",
        path        = str(BASE / "tabular" / "heart.csv"),
        target_col  = "target",
        modality    = "tabular",
        description = "Heart disease classification — Cleveland Clinic dataset",
        sensitivity = "high",
        row_count   = "303 patients",
    ),

    "diabetes": DatasetEntry(
        name        = "diabetes",
        path        = str(BASE / "tabular" / "pima-indians-diabetes.csv"),
        target_col  = "Outcome",
        modality    = "tabular",
        description = "Pima Indians diabetes — diagnostic measurements",
        sensitivity = "high",
        row_count   = "768 patients",
    ),

    "readmission": DatasetEntry(
        name        = "readmission",
        path        = str(BASE / "tabular" / "diabetic_data.csv"),
        target_col  = "readmitted",
        modality    = "tabular",
        description = "130-US hospitals diabetes readmission records",
        sensitivity = "critical",
        row_count   = "~100,000 encounters",
    ),

    "mtsamples": DatasetEntry(
        name        = "mtsamples",
        path        = str(BASE / "text" / "mtsamples" / "mtsamples.csv"),
        target_col  = "medical_specialty",
        modality    = "text",
        description = "Clinical transcription notes — free text EHR",
        sensitivity = "high",
        row_count   = "4,999 notes",
    ),

    "xray": DatasetEntry(
        name        = "xray",
        path        = str(BASE / "images" / "chest_xray" / "metadata.csv"),
        target_col  = "label",
        modality    = "image",
        description = "COVID-19 chest X-ray — image metadata",
        sensitivity = "critical",
        row_count   = "varies",
    ),

    # FIX: patient_journey now points to its own CSV which has
    # encounter_sequence and time_in_hospital columns — properly temporal
    "patient_journey": DatasetEntry(
        name        = "patient_journey",
        path        = str(BASE / "graph" / "patient_journey.csv"),
        target_col  = "readmitted",
        modality    = "graph",
        is_temporal = True,
        description = "Patient encounter graph — temporal EHR sequences",
        sensitivity = "critical",
        row_count   = "20,000 encounters",
    ),
}


def get_dataset(name: str) -> DatasetEntry:
    if name not in _REGISTRY:
        raise KeyError(
            f"Dataset '{name}' not found. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    entry = _REGISTRY[name]

    if name == "xray" and not os.path.exists(entry.path):
        _generate_xray_metadata(entry.path)

    if not os.path.exists(entry.path):
        raise FileNotFoundError(
            f"\nDataset file not found:\n  {entry.path}\n\n"
            f"Expected folder structure:\n"
            f"  synthetic_os/data/tabular/heart.csv\n"
            f"  synthetic_os/data/tabular/pima-indians-diabetes.csv\n"
            f"  synthetic_os/data/tabular/diabetic_data.csv\n"
            f"  synthetic_os/data/text/mtsamples/mtsamples.csv\n"
            f"  synthetic_os/data/images/chest_xray/metadata.csv\n"
            f"  synthetic_os/data/graph/patient_journey.csv\n"
        )
    return entry


def get_dataset_safe(name: str) -> Optional[DatasetEntry]:
    try:
        return get_dataset(name)
    except (KeyError, FileNotFoundError):
        return None


def list_datasets() -> list[str]:
    return list(_REGISTRY.keys())


def available_datasets() -> list[DatasetEntry]:
    result = []
    for entry in _REGISTRY.values():
        path = entry.path
        if entry.name == "xray" and not os.path.exists(path):
            _generate_xray_metadata(path)
        if os.path.exists(path):
            result.append(entry)
    return result


def _generate_xray_metadata(path: str):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    labels   = ["Normal", "COVID", "Viral Pneumonia", "Lung Opacity"]
    genders  = ["Male", "Female"]
    findings = ["No Finding", "Ground-glass opacity", "Consolidation",
                "Pleural effusion", "Infiltration"]
    random.seed(42)
    rows = []
    for i in range(500):
        label = random.choice(labels)
        rows.append({
            "patient_id":   f"P{i:04d}",
            "label":        label,
            "finding":      random.choice(findings),
            "gender":       random.choice(genders),
            "age":          random.randint(18, 85),
            "image_path":   f"images/{label.lower().replace(' ','_')}_{i:04d}.png",
        })
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [Registry] Auto-generated demo xray metadata → {out}")
