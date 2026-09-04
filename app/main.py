"""Application entry point.

Startup order matters: the schema must exist before jobs can be recovered, and
recovery must run before workers start, or a worker could claim a row that
recovery is about to reset.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from . import config, db, search, upload, vector_store
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

    # Fail fast on a misconfigured QDRANT_URL rather than only discovering it
    # on the first upload's indexing attempt.
    try:
        vector_store.get_client().get_collections()
        logger.info("connected to Qdrant at %s", config.QDRANT_URL)
    except Exception:
        logger.exception(
            "could not reach Qdrant at %s -- is the qdrant service up?",
            config.QDRANT_URL,
        )
        raise

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
        "Every request takes an `X-User-Id` header identifying the caller "
        "(any string; omit it and requests share one anonymous identity). "
        "Files are scoped to their owner -- listing, status, search, and "
        "delete only see files created under the same `X-User-Id`. Click "
        "**Authorize** below and set it once to have it applied to every "
        "request tried from this page.\n\n"
        "**Upload flow**\n"
        "1. `POST /files` to register the upload and get a `file_id`.\n"
        "2. `PUT /files/{file_id}/chunk?offset=N` repeatedly with raw chunk bodies.\n"
        "3. `POST /files/{file_id}/complete` once every chunk has been sent.\n"
        "4. `GET /files/{file_id}/status` to watch progress -- or, after an "
        "interruption, to learn the offset to resume from.\n"
        "5. `POST /files/{file_id}/search` once `searchable` is true.\n\n"
        "Indexing runs during the upload, so passages become searchable before "
        "`complete` is even called."
    ),
    # Keeps the X-User-Id value entered via "Authorize" (below) filled in
    # across page reloads and every "Try it out" call, instead of it
    # resetting per request or per endpoint.
    swagger_ui_parameters={"persistAuthorization": True},
)

app.include_router(upload.router)
app.include_router(search.router)


def _custom_openapi() -> dict:
    """Register X-User-Id as a real security scheme, and document the one
    endpoint whose body FastAPI can't infer a schema for on its own.

    Without the security scheme, Swagger has no "Authorize" button for a
    plain header parameter -- every endpoint would need X-User-Id typed in
    separately, by hand, per try. Declaring it as an apiKey security scheme
    applied to every route gives Swagger one Authorize dialog that then
    auto-fills the header on every request tried from the page.

    PUT /files/{file_id}/chunk reads its body via the raw Request rather than
    a typed parameter (see app/upload.py's upload_chunk docstring for why:
    FastAPI 0.115's Body(media_type=...) 422s on real clients' differing
    default Content-Type headers). A raw Request is invisible to FastAPI's
    schema generator, so without this, Swagger's "Try it out" for that
    endpoint shows no body field at all. The requestBody below is added by
    hand purely for documentation -- it does not change how the endpoint
    parses the request at runtime.
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["components"]["securitySchemes"] = {
        "UserId": {
            "type": "apiKey",
            "in": "header",
            "name": "X-User-Id",
            "description": "Any string identifying the caller. Files are scoped to it.",
        }
    }
    for path in schema["paths"].values():
        for operation in path.values():
            operation.setdefault("security", []).append({"UserId": []})

    schema["paths"]["/files/{file_id}/chunk"]["put"]["requestBody"] = {
        "required": True,
        "content": {
            "application/octet-stream": {
                "schema": {
                    "type": "string",
                    "format": "binary",
                    "description": "Raw chunk bytes -- not JSON, not base64, "
                    "not wrapped in a field. The exact bytes of the file at "
                    "[offset, offset+len(body)).",
                }
            }
        },
    }

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi


@app.get("/health", tags=["meta"], summary="Liveness, worker, and Qdrant state")
def health() -> dict:
    try:
        vector_store.get_client().get_collections()
        qdrant_ok = True
    except Exception:
        qdrant_ok = False

    return {
        "status": "ok" if qdrant_ok else "degraded",
        "workers_running": worker._pool.running if worker._pool else False,
        "queue_depth": job_queue.pending_count(),
        "qdrant_reachable": qdrant_ok,
    }
