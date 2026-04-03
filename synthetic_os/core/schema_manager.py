"""
SchemaManager — enforces output schema contract on synthetic DataFrames
Fixes:
  - Columns added in ONE pd.concat call (was inserting one-by-one → PerformanceWarning)
  - Handles OHE column presence check correctly
  - Returns defragmented DataFrame via .copy()
  - Missing-column filler now respects dtype:
      previously ALL missing columns were zero-filled (float).  Categorical
      columns filled with 0.0 were then treated as numeric by downstream
      evaluators, which corrupted distribution metrics and collapsed utility
      scores (Wasserstein, JSD).  Now categoricals get an empty-string filler
      and numerics keep the zero fill.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from synthetic_os.config.schema import DataSchema


class SchemaManager:
    def __init__(self, schema: DataSchema):
        self.schema = schema

    def enforce(self, synthetic_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure synthetic_df has exactly the columns the schema expects.
        Missing columns are filled with sensible defaults by dtype;
        extra columns are dropped.
        """
        expected = self.schema.columns  # list[str]

        # Identify truly missing columns
        missing = [c for c in expected if c not in synthetic_df.columns]

        if missing:
            # Determine which missing columns are categorical so we can fill
            # them with "" rather than 0.0 — zero-filling a string column makes
            # it look numeric to every downstream metric and tanks utility scores.
            discrete_set = set(self.schema.resolve_discrete_columns(synthetic_df))

            filler_cols: dict[str, np.ndarray] = {}
            for c in missing:
                if c in discrete_set:
                    filler_cols[c] = pd.array([""] * len(synthetic_df), dtype=object)
                else:
                    filler_cols[c] = np.zeros(len(synthetic_df), dtype=float)

            filler = pd.DataFrame(filler_cols, index=synthetic_df.index)
            synthetic_df = pd.concat([synthetic_df, filler], axis=1)

        # Drop any extra columns, preserve order
        synthetic_df = synthetic_df[expected]

        # Defragment to avoid downstream PerformanceWarnings
        return synthetic_df.copy()