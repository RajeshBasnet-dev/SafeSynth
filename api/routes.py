from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from core.database import SessionLocal
from models.orm import DataFile, TaskStatus, TrainingTask
from models.schemas import ReportResponse, TaskResponse, TrainResponse, UploadResponse
from services.analytics import to_json
from services.synthetic_service import train_and_generate

logger = logging.getLogger(__name__)
router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
SYNTHETIC_DIR = Path("data/synthetic")


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

        output_path = SYNTHETIC_DIR / f"synthetic_{task.id}.csv"
        report = train_and_generate(data_file.stored_path, str(output_path))

        task.synthetic_path = str(output_path)
        task.report_json = to_json(report)
        task.status = TaskStatus.COMPLETED
        task.error_message = None
        db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Task %s failed", task_id)
        task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
        if task:
            task.status = TaskStatus.FAILED
            task.error_message = str(exc)
            db.commit()
    finally:
        db.close()


@router.post("/upload", response_model=UploadResponse)
def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV uploads are allowed.")

    stored_path = UPLOAD_DIR / f"{uuid4()}.csv"
    with stored_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    record = DataFile(original_name=file.filename, stored_path=str(stored_path))
    db.add(record)
    db.commit()
    db.refresh(record)

    return UploadResponse(file_id=record.id, filename=record.original_name)


@router.post("/train/{file_id}", response_model=TrainResponse)
def train_file(file_id: int, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    data_file = db.query(DataFile).filter(DataFile.id == file_id).first()
    if data_file is None:
        raise HTTPException(status_code=404, detail="File not found.")

    task = TrainingTask(file_id=file_id, status=TaskStatus.PENDING)
    db.add(task)
    db.commit()
    db.refresh(task)

    background_tasks.add_task(_run_training_task, task.id)
    return TrainResponse(task_id=task.id, status=task.status.value)


@router.get("/task/{task_id}", response_model=TaskResponse)
def get_task_status(task_id: int, db: Session = Depends(get_db)):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    report = json.loads(task.report_json) if task.report_json else None
    return TaskResponse(
        task_id=task.id,
        file_id=task.file_id,
        status=task.status.value,
        report=report,
        error_message=task.error_message,
    )


@router.get("/report/{task_id}", response_model=ReportResponse)
def get_report(task_id: int, db: Session = Depends(get_db)):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Report available only when task is completed.")
    if not task.report_json:
        raise HTTPException(status_code=404, detail="Report not found.")

    report = json.loads(task.report_json)
    return ReportResponse(
        task_id=task.id,
        status=task.status.value,
        overall_score=report.get("overall_score", 0),
        quality_label=report.get("quality_label", "unknown"),
        privacy_risk=report.get("privacy_risk", "unknown"),
        summary=report.get("summary", ""),
        metrics=report.get("metrics", []),
        insights=report.get("insights", []),
    )


@router.get("/download/{task_id}")
def download_synthetic(task_id: int, db: Session = Depends(get_db)):
    task = db.query(TrainingTask).filter(TrainingTask.id == task_id).first()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Task is {task.status.value}. Not ready for download.")
    if not task.synthetic_path or not os.path.exists(task.synthetic_path):
        raise HTTPException(status_code=404, detail="Synthetic file not found.")

    return FileResponse(path=task.synthetic_path, filename=f"synthetic_{task.id}.csv", media_type="text/csv")
