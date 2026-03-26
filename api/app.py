from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI

from api.routes import SYNTHETIC_DIR, UPLOAD_DIR, router
from core.database import engine
from models.orm import Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title="SafeSynth Privacy-Preserving Synthetic Data Platform", version="1.0.0")
app.include_router(router)


@app.on_event("startup")
def startup_event() -> None:
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    Path(SYNTHETIC_DIR).mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
