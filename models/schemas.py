from __future__ import annotations

from pydantic import BaseModel


class UploadResponse(BaseModel):
    file_id: int
    filename: str


class TrainResponse(BaseModel):
    task_id: int
    status: str


class TaskResponse(BaseModel):
    task_id: int
    file_id: int
    status: str
    report: dict | None = None
    error_message: str | None = None


class ReportResponse(BaseModel):
    task_id: int
    status: str
    overall_score: float
    quality_label: str
    privacy_risk: str
    summary: str
    metrics: list[dict]
    insights: list[str]
