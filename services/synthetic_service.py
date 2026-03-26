from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from ctgan import CTGAN

from services.analytics import build_analysis_report

logger = logging.getLogger(__name__)


def train_and_generate(input_csv: str, output_csv: str, epochs: int = 80) -> dict:
    real_df = pd.read_csv(input_csv)
    if real_df.empty:
        raise ValueError("Uploaded CSV is empty.")

    logger.info("Starting CTGAN training for file: %s", input_csv)
    model = CTGAN(epochs=epochs, verbose=False, cuda=False)
    model.fit(real_df)

    synthetic_df = model.sample(len(real_df))
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(output_csv, index=False)

    report = build_analysis_report(real_df, synthetic_df)
    logger.info("Training completed. Overall quality score: %s", report.get("overall_score"))
    return report
