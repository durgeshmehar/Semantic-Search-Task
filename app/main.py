"""Application entry point.

Startup order matters: the schema must exist before jobs can be recovered, and
recovery must run before workers start, or a worker could claim a row that
recovery is about to reset.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import config, db, search, upload
from .pipeline import job_queue, worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_dirs()
    db.init_db()

    # Anything left mid-flight by an unclean shutdown goes back on the queue.
    recovered = job_queue.recover_stuck_jobs()
    if recovered:
        logger.info("recovered %d chunks left in-flight by a previous run", recovered)

    worker.start_workers()
    logger.info("started %d indexing workers", config.WORKER_COUNT)

    yield

    worker.stop_workers()
    db.close_connection()


app = FastAPI(
    title="Large File Processing & Search",
    version="1.0.0",
    lifespan=lifespan,
    description=(
        "Upload text files up to 10 GB on a 4 GB machine, resume interrupted "
        "uploads, and search their contents in natural language.\n\n"
        "**Upload flow**\n"
        "1. `POST /files` to register the upload and get a `file_id`.\n"
        "2. `PUT /files/{file_id}/chunk?offset=N` repeatedly with raw chunk bodies.\n"
        "3. `GET /files/{file_id}/status` to watch progress -- or, after an "
        "interruption, to learn the offset to resume from.\n"
        "4. `POST /files/{file_id}/search` once `searchable` is true.\n\n"
        "Indexing runs during the upload, so passages become searchable before "
        "the last byte arrives."
    ),
)

app.include_router(upload.router)
app.include_router(search.router)


@app.get("/health", tags=["meta"], summary="Liveness and worker state")
def health() -> dict:
    return {
        "status": "ok",
        "workers_running": worker._pool.running if worker._pool else False,
        "queue_depth": job_queue.pending_count(),
    }
