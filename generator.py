from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from ctgan import CTGAN


def _compute_fidelity(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> dict:
    numeric_cols = real_df.select_dtypes(include="number").columns.tolist()
    fidelity = {}

    for col in numeric_cols:
        real_mean = float(real_df[col].mean())
        synthetic_mean = float(synthetic_df[col].mean())
        gap = abs(real_mean - synthetic_mean)
        fidelity[col] = {
            "real_mean": real_mean,
            "synthetic_mean": synthetic_mean,
            "absolute_gap": gap,
        }

    return fidelity


def train_and_generate(
    input_csv: str,
    output_csv: str,
    epochs: int = 50,
) -> str:
    real_df = pd.read_csv(input_csv)
    if real_df.empty:
        raise ValueError("Uploaded CSV is empty.")

    model = CTGAN(epochs=epochs, verbose=False, cuda=False)
    model.fit(real_df)

    synthetic_df = model.sample(len(real_df))
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(output_csv, index=False)

    fidelity_report = _compute_fidelity(real_df, synthetic_df)
    return json.dumps(fidelity_report)
