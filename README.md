# SafeSynth

SafeSynth is a privacy-first tool that converts **real financial CSV data** into **synthetic datasets** using a CTGAN model.

It includes:
- **FastAPI backend** for upload, training, status tracking, and download
- **SQLite + SQLAlchemy** persistence for task lifecycle management
- **Streamlit frontend** for interactive usage and visualization
- **Fidelity check** comparing real vs synthetic numeric column means

---

## Project Structure

- `main.py` → FastAPI backend
- `app.py` → Streamlit frontend
- `models.py` → SQLite/SQLAlchemy models
- `generator.py` → CTGAN training + synthetic generation + fidelity logic
- `requirements.txt` → dependencies

---

## Features

- Upload CSV files through API or Streamlit UI
- Start CTGAN training as a background task
- Track task states: `Pending`, `Training`, `Completed`, `Failed`
- Download generated synthetic CSV after completion
- View fidelity report (column-wise mean comparisons)
- Compare real vs synthetic distributions with Plotly histogram

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

---

## Run the Backend (FastAPI)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend URL: `http://localhost:8000`

---

## Run the Frontend (Streamlit)

In a separate terminal:

```bash
streamlit run app.py
```

Optional custom backend URL:

```bash
BACKEND_URL=http://localhost:8000 streamlit run app.py
```

---

## API Endpoints

### `POST /upload`
Upload a CSV file.

**Response:**

```json
{
  "file_id": 1,
  "filename": "transactions.csv"
}
```

### `POST /train/{file_id}`
Start CTGAN training in a FastAPI background task.

**Response:**

```json
{
  "task_id": 10,
  "status": "Pending"
}
```

### `GET /task/{task_id}`
Get task status and fidelity report (when available).

### `GET /download/{task_id}`
Download the generated synthetic CSV when task is completed.

---

## Workflow

1. Upload real CSV (`/upload` or Streamlit uploader)
2. Start training (`/train/{file_id}`)
3. Refresh status (`/task/{task_id}`)
4. Download synthetic data (`/download/{task_id}`)
5. Inspect fidelity means + histogram in Streamlit

---

## Notes

- CTGAN training time depends on dataset size and hardware.
- This implementation uses CPU by default (`cuda=False`) for compatibility.
- Fidelity check is intentionally simple (mean comparison) and can be extended with richer statistical metrics.

---

## Resume Highlights

- **Background ML orchestration** with FastAPI `BackgroundTasks`
- **Interactive data product** with Streamlit + Plotly
- **Persistent state management** using SQLite + SQLAlchemy
- **Privacy-preserving synthetic data generation** with CTGAN
