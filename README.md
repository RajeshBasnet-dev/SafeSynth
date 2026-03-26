# SafeSynth — Privacy-Preserving Synthetic Data Platform (MVP)

SafeSynth helps fintech teams and ML engineers generate **high-fidelity synthetic financial datasets** from sensitive CSVs, so they can train and share models without exposing raw customer data.

## Business Use Cases

- **Train ML models without exposing sensitive data** in dev/staging.
- **Share realistic datasets safely across teams** (data science, vendors, QA).
- **Accelerate experimentation** when production data access is restricted.

---

## MVP Capabilities

### 1) Synthetic Data Generation
- CTGAN-powered tabular synthesis from uploaded CSV data.
- Background training workflow using FastAPI background tasks.

### 2) Advanced Fidelity & Quality Analysis
For each numerical column, SafeSynth computes:
- Mean difference
- Standard deviation difference
- KS test statistic (`ks_score`)
- Column-level fidelity score (0–100)

Global analysis includes:
- Correlation matrix similarity
- Overall quality score (0–100)
- Quality interpretation:
  - `90+` → **عالي quality**
  - `70–89` → **usable**
  - `<70` → **poor**

### 3) Insights Engine
- Column type detection (numerical/categorical)
- Skewness detection
- Missing value monitoring
- Outlier signal detection
- Drift insights (e.g., high variance differences)

### 4) Privacy Risk Indicator
Simple leakage heuristic flags if synthetic samples are too close to real data:
- `LOW`
- `MEDIUM`
- `HIGH`

### 5) Product UI (Streamlit Dashboard)
- Upload dataset
- Start training
- Refresh status
- View scores + insights + fidelity table
- Compare distributions (Real vs Synthetic)
- Download synthetic CSV

---

## Updated Backend Architecture

```text
SafeSynth/
├── api/
│   ├── app.py              # FastAPI app + startup
│   └── routes.py           # API endpoints
├── core/
│   └── database.py         # SQLAlchemy engine/session
├── models/
│   ├── orm.py              # DB models (DataFile, TrainingTask)
│   └── schemas.py          # API response schemas
├── services/
│   ├── analytics.py        # fidelity, scoring, insights, privacy risk
│   └── synthetic_service.py# CTGAN train/generate orchestration
├── data/
│   └── examples/
│       └── financial_transactions_sample.csv
├── app.py                  # Streamlit MVP dashboard
├── main.py                 # ASGI entrypoint
├── requirements.txt
└── README.md
```

---

## API Endpoints

- `POST /upload` → upload real CSV
- `POST /train/{file_id}` → launch background CTGAN training
- `GET /task/{task_id}` → training status + compact report
- `GET /report/{task_id}` → full analysis report (scores + insights)
- `GET /download/{task_id}` → download synthetic CSV

### Example `GET /report/{task_id}` Response

```json
{
  "task_id": 12,
  "status": "Completed",
  "overall_score": 87.4,
  "quality_label": "usable",
  "privacy_risk": "LOW",
  "summary": "Overall Score: 87.4/100 – usable synthetic data quality",
  "metrics": [
    {
      "column": "Transaction Amount",
      "mean_diff": 2.1,
      "std_diff": 3.5,
      "ks_score": 0.08,
      "fidelity_score": 91.0
    }
  ],
  "insights": [
    "Column 'Transaction Amount' is right-skewed (skewness=1.24)."
  ]
}
```

---

## Quick Start

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run Backend

```bash
uvicorn main:app --reload --port 8000
```

### 3) Run Streamlit

```bash
streamlit run app.py
```

(Optional)

```bash
BACKEND_URL=http://localhost:8000 streamlit run app.py
```

---

## Example Dataset

Use `data/examples/financial_transactions_sample.csv` for a quick MVP demo.

---

## Screenshots

> Add UI screenshots here after running the app:

- `docs/screenshots/dashboard-upload.png`
- `docs/screenshots/results-dashboard.png`

---

## Why This Is Portfolio-Worthy

- Demonstrates **applied privacy-preserving ML engineering**.
- Shows **full-stack AI product delivery** (API + background jobs + UI + reporting).
- Includes **quality scoring and risk interpretation**, not just model training.
- Reflects a real startup-friendly workflow for fintech data teams.
