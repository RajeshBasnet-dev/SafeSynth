from api.app import app

__all__ = ["app"]
from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from generator import train_and_generate
from models import DataFile, SessionLocal, TaskStatus, TrainingTask, init_db

UPLOAD_DIR = Path("data/uploads")
SYNTHETIC_DIR = Path("data/synthetic")

app = FastAPI(title="SafeSynth API")


@app.on_event("startup")
def startup_event() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _run_training_task(task_id: int) -> None:
    db = SessionLocal()
    try:
        task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        if task is None:
            return

        task.status = TaskStatus.TRAINING
        db.commit()

        data_file = db.query(DataFile).filter(DataFile.id == task.file_id).first()
        if data_file is None:
            raise ValueError("Referenced file not found.")

        output_name = f"synthetic_{task.id}.csv"
        output_path = SYNTHETIC_DIR / output_name
        fidelity_report = train_and_generate(data_file.stored_path, str(output_path))

        task.synthetic_path = str(output_path)
        task.fidelity_report = fidelity_report
        task.status = TaskStatus.COMPLETED
        db.commit()
    except Exception as exc:  # noqa: BLE001
        if task := db.query(TrainingTask).filter(TrainingTask.id == task_id).first():
            task.status = TaskStatus.FAILED
            task.error_message = str(exc)
            db.commit()
    finally:
        db.close()


@app.post("/upload")
def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are allowed.")

    file_id = str(uuid4())
    stored_filename = f"{file_id}.csv"
    stored_path = UPLOAD_DIR / stored_filename

    with stored_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    record = DataFile(original_name=file.filename, stored_path=str(stored_path))
    db.add(record)
    db.commit()
    db.refresh(record)

    return {"file_id": record.id, "filename": record.original_name}


@app.post("/train/{file_id}")
def train_file(file_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    data_file = db.query(DataFile).filter(DataFile.id == file_id).first()
    if data_file is None:
        raise HTTPException(status_code=404, detail="File not found.")

    task = TrainingTask(file_id=file_id, status=TaskStatus.PENDING)
    db.add(task)
    db.commit()
    db.refresh(task)

    background_tasks.add_task(_run_training_task, task.id)
    return {"task_id": task.id, "status": task.status.value}


@app.get("/task/{task_id}")
def get_task_status(task_id: int, db: Session = Depends(get_db)):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    return {
        "task_id": task.id,
        "file_id": task.file_id,
        "status": task.status.value,
        "fidelity_report": task.fidelity_report,
        "error_message": task.error_message,
    }


@app.get("/download/{task_id}")
def download_synthetic(task_id: int, db: Session = Depends(get_db)):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Task is {task.status.value}. Not ready for download.")
    if not task.synthetic_path or not os.path.exists(task.synthetic_path):
        raise HTTPException(status_code=404, detail="Synthetic file not found.")

    return FileResponse(path=task.synthetic_path, filename=f"synthetic_{task.id}.csv", media_type="text/csv")
