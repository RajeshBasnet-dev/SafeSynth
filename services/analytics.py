from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, skew


def _safe_float(value: float | int | np.floating) -> float:
    if pd.isna(value):
        return 0.0
    return float(value)


def _column_type(series: pd.Series) -> str:
    return "numerical" if pd.api.types.is_numeric_dtype(series) else "categorical"


def _outlier_ratio(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return 0.0
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = clean[(clean < lower) | (clean > upper)]
    return float(len(outliers) / len(clean))


def _quality_label(score: float) -> str:
    if score >= 90:
        return "عالي quality"
    if score >= 70:
        return "usable"
    return "poor"


def build_analysis_report(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    common_cols = [c for c in real_df.columns if c in synth_df.columns]
    numeric_cols = [c for c in common_cols if pd.api.types.is_numeric_dtype(real_df[c])]

    metrics: list[dict] = []
    insights: list[str] = []

    for col in common_cols:
        col_kind = _column_type(real_df[col])
        missing_real = float(real_df[col].isna().mean())
        missing_synth = float(synth_df[col].isna().mean())

        if col_kind == "categorical":
            if missing_real > 0.2:
                insights.append(f"Column '{col}' has high missing values in real data ({missing_real:.1%}).")
            continue

        real_col = real_df[col].dropna()
        synth_col = synth_df[col].dropna()
        if real_col.empty or synth_col.empty:
            continue

        mean_diff = abs(float(real_col.mean()) - float(synth_col.mean()))
        std_diff = abs(float(real_col.std(ddof=0)) - float(synth_col.std(ddof=0)))

        ks_stat = float(ks_2samp(real_col, synth_col).statistic)

        std_ref = max(abs(float(real_col.std(ddof=0))), 1e-6)
        mean_ref = max(abs(float(real_col.mean())), 1.0)
        mean_score = max(0.0, 100 - (mean_diff / mean_ref) * 100)
        std_score = max(0.0, 100 - (std_diff / std_ref) * 100)
        ks_score = max(0.0, 100 - ks_stat * 100)
        fidelity_score = round(0.35 * mean_score + 0.25 * std_score + 0.40 * ks_score, 2)

        metrics.append(
            {
                "column": col,
                "column_type": col_kind,
                "mean_diff": round(mean_diff, 4),
                "std_diff": round(std_diff, 4),
                "ks_score": round(ks_stat, 4),
                "missing_real": round(missing_real, 4),
                "missing_synthetic": round(missing_synth, 4),
                "fidelity_score": fidelity_score,
            }
        )

        skewness = _safe_float(skew(real_col, nan_policy="omit"))
        if skewness > 1:
            insights.append(f"Column '{col}' is right-skewed (skewness={skewness:.2f}).")
        elif skewness < -1:
            insights.append(f"Column '{col}' is left-skewed (skewness={skewness:.2f}).")

        outlier_ratio = _outlier_ratio(real_col)
        if outlier_ratio > 0.05:
            insights.append(f"Column '{col}' contains notable outliers ({outlier_ratio:.1%}).")

        if std_diff / std_ref > 0.3:
            insights.append(f"High variance difference detected in '{col}'.")

    corr_similarity = 100.0
    if len(numeric_cols) >= 2:
        real_corr = real_df[numeric_cols].corr().fillna(0)
        synth_corr = synth_df[numeric_cols].corr().fillna(0)
        corr_delta = float(np.mean(np.abs(real_corr.values - synth_corr.values)))
        corr_similarity = max(0.0, 100 - corr_delta * 100)
    else:
        corr_delta = 0.0

    if corr_similarity < 70:
        insights.append("Correlation drift is high across numerical columns.")

    column_scores = [m["fidelity_score"] for m in metrics]
    base_score = float(np.mean(column_scores)) if column_scores else 0.0
    overall_score = round(0.75 * base_score + 0.25 * corr_similarity, 2)

    quality_label = _quality_label(overall_score)
    summary = f"Overall Score: {overall_score}/100 – {quality_label} synthetic data quality"

    privacy_risk = "LOW"
    leakage_matches = 0.0
    if common_cols:
        real_rows = set(real_df[common_cols].astype(str).agg("|".join, axis=1).tolist())
        synth_rows = synth_df[common_cols].astype(str).agg("|".join, axis=1).tolist()
        if synth_rows:
            leakage_matches = sum(1 for row in synth_rows if row in real_rows) / len(synth_rows)

    avg_ks = float(np.mean([m["ks_score"] for m in metrics])) if metrics else 1.0
    if leakage_matches > 0.05 or avg_ks < 0.05:
        privacy_risk = "HIGH"
    elif leakage_matches > 0.01 or avg_ks < 0.12:
        privacy_risk = "MEDIUM"

    insights.append(f"Privacy leakage heuristic match rate: {leakage_matches:.2%}.")

    return {
        "overall_score": overall_score,
        "quality_label": quality_label,
        "privacy_risk": privacy_risk,
        "summary": summary,
        "correlation_similarity": round(corr_similarity, 2),
        "correlation_delta": round(corr_delta, 4),
        "metrics": metrics,
        "insights": sorted(set(insights)),
    }


def to_json(report: dict) -> str:
    return json.dumps(report)
