"""
DataSchema — shared data contract across the pipeline
Fixes:
  - has_target() method added (was missing → caused ALL Optuna trials to score 0.0)
  - modality field added (tabular | text | image | graph)
  - is_temporal flag for patient journey / GNN routing
  - discrete_columns field added + resolve_discrete_columns() helper:
      CTGAN/TabDDPM need an explicit list of categorical columns.
      Without it every string column was treated as continuous, causing
      TypeError: Could not convert string '...' to numeric at training time,
      and inflating noise on integer-coded categoricals which collapsed utility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class DataSchema:
    name:             str
    columns:          list[str]
    target_col:       Optional[str]       = None
    modality:         str                 = "tabular"   # tabular | text | image | graph
    is_temporal:      bool                = False
    # Explicit categorical column list for CTGAN / TabDDPM.
    # None → auto-detect via resolve_discrete_columns(df).
    discrete_columns: Optional[list[str]] = field(default=None)

    def has_target(self) -> bool:
        """Return True if a target column is defined and present in columns."""
        return (
            self.target_col is not None
            and self.target_col in self.columns
        )

    def feature_cols(self) -> list[str]:
        """All columns except the target."""
        return [c for c in self.columns if c != self.target_col]

    def resolve_discrete_columns(self, df: "pd.DataFrame") -> list[str]:
        """
        Return the definitive discrete-column list to pass to CTGAN / TabDDPM.

        Priority:
          1. Explicit list stored in self.discrete_columns (set by app.py from
             dtype inspection at upload time) — only columns that exist in df
             are returned so stale schema names never cause KeyErrors.
          2. Auto-detection fallback: object/category dtype columns, plus integer
             columns with ≤ 20 unique values (binary flags, ordinal codes, etc.).

        Every model adapter should call schema.resolve_discrete_columns(df)
        instead of doing its own ad-hoc dtype check.  This is now the single
        source of truth and eliminates the crash that caused the bug.
        """
        import pandas as _pd  # local import — keeps schema.py lightweight

        if self.discrete_columns is not None:
            return [c for c in self.discrete_columns if c in df.columns]

        discrete: list[str] = []
        for col in self.columns:
            if col not in df.columns:
                continue
            dtype = df[col].dtype
            if dtype == object or str(dtype).startswith("category"):
                discrete.append(col)
            elif dtype.kind in ("i", "u") and df[col].nunique() <= 20:
                discrete.append(col)
        return discrete