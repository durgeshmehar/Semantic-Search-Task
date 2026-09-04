# Large File Processing & Search

A backend service that ingests text files up to **10 GB** on a machine with **4 GB of RAM**,
resumes interrupted uploads, and searches contents in **natural language** — matching on meaning
rather than keywords.

Searching `database connectivity problems` returns `Connection to database failed after 30
seconds.` even though they share one word.

---

## Setup

Two services: the API, and Qdrant for vector storage.

```bash
docker compose up --build                       # API on :8000, docs at /docs
docker compose --profile test run --rm tests    # test suite, against a real Qdrant
```

Without Docker (Qdrant is still required):

```bash
docker run -p 6333:6333 qdrant/qdrant:v1.12.1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app
```

### Try it

Every request carries `X-User-Id` (any string), which scopes files to their creator.

```bash
printf 'INFO Starting server\nERROR Connection to database failed after 30 seconds.\nINFO Ready\n' > sample.log
H='-H X-User-Id:demo -H Content-Type:application/json'

FILE_ID=$(curl -s -X POST http://localhost:8000/files $H \
  -d "{\"filename\":\"sample.log\",\"total_size\":$(wc -c < sample.log)}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["file_id"])')

curl -s -X PUT "http://localhost:8000/files/$FILE_ID/chunk?offset=0" -H 'X-User-Id: demo' --data-binary @sample.log
curl -s -X POST "http://localhost:8000/files/$FILE_ID/complete" -H 'X-User-Id: demo'
curl -s "http://localhost:8000/files/$FILE_ID/status" -H 'X-User-Id: demo'
curl -s -X POST "http://localhost:8000/files/$FILE_ID/search" $H \
  -d '{"query":"database connectivity problems","top_k":3}'
```

---

## Architecture

Interactive version with side-by-side comparisons: **[Upload Pipeline Architecture](https://claude.ai/code/artifact/cd179b70-8d75-43b0-b6a8-20cace46a689)**

```
UPLOAD PATH  (synchronous — the client waits for this, and only this)

  client
    │  PUT /files/{id}/chunk?offset=N
    ▼
  upload handler ── append bytes to {id}.partial
                 ── split into line-aligned passages
                 ── enqueue passage rows in SQLite
    │
    ▼
  response (bytes_received, upload_status)


INDEXING PATH  (asynchronous — runs continuously, off the request)

  SQLite chunks table  ──poll──▶  worker pool (×2)
  (pending → processing              │
   → indexed | failed)               ├─ read_range()   → text from disk
                                      ├─ embed()        → 384-dim vector
                                      └─ upsert()       ─────────────┐
                                                                      ▼
                                                          Qdrant collection
                                                          (HNSW · on disk ·
                                                           int8 quantized)


SEARCH PATH  (synchronous — one request touches both stores)

  client
    │  POST /files/{id}/search  {"query": "..."}
    ▼
  embed query ──▶ Qdrant nearest-neighbor search ──▶ (byte range, score)
                                                            │
                                                            ▼
                                              seek() the file on disk → text
                                                            │
                                                            ▼
                                                        response
```

The one decision this follows from: **`PUT /chunk` never waits on embedding.** It appends bytes,
splits lines, and enqueues — all cheap — then returns. Embedding happens later, off the request, in
a worker pool reading the same SQLite queue. Search is the only path touching both Qdrant and disk
in one request, because a hit is a byte range that still has to be read back as text.

---

## API

Interactive documentation: **`/docs`**. Every route requires `X-User-Id`; a file owned by someone
else 404s.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/files` | Register an upload → `file_id` |
| `PUT` | `/files/{file_id}/chunk?offset=N` | Upload one chunk (raw body) |
| `POST` | `/files/{file_id}/complete` | Mark the upload finished |
| `GET` | `/files/{file_id}/status` | Upload **and** processing progress; also the resume endpoint |
| `POST` | `/files/{file_id}/search` | Natural-language search |
| `GET` | `/files` | List your uploads |
| `DELETE` | `/files/{file_id}` | Delete a file and its index |
| `GET` | `/health` | Liveness, workers, queue depth, Qdrant connectivity |

`PUT /chunk`: `offset` must equal `bytes_received`. `409` on mismatch (names the correct offset),
`413` over the chunk size limit, `400` only past the hard `MAX_FILE_BYTES` ceiling — the client's
`total_size` is a sizing hint, not an enforced cap.

`POST /complete`: called once every chunk is sent; this, not a byte count, is what finalizes the
upload. Idempotent.

`POST /search`: `{"query": "...", "top_k": 5}` → results with `text`, `start_byte`/`end_byte`, and
`score` (cosine similarity). Works **during** an upload — whatever is indexed so far is queryable.

---

## Design discussion

### 1. A 10 GB file on a 4 GB machine

The file is never held in memory: chunks are appended to disk and released as they arrive, so peak
memory scales with concurrent uploads, not file size. Chunk rows store `(start_byte, end_byte)`
rather than the text itself — the file on disk already has it, so nothing duplicates the corpus.

Vectors live in Qdrant, one collection per file, with raw vectors and the HNSW graph on disk and
only an int8-quantized copy (4× smaller than float32, ~2–3% recall cost) kept resident. Search is
approximate (HNSW), not exhaustive.

Passage size scales with declared file size (`config.passage_size_for`): small passages embed
precisely, but a 10 GB file at the smallest passage size would need ~22M vectors (~8 GB even
quantized), so larger files get proportionally larger passages, keeping the resident footprint
under 1 GB regardless of file size:

| File size | Passage target | Resident quantized |
|---|---|---|
| ≤1 GB | 600 B | ≤0.8 GB |
| 10 GB | 4,800 B | 1.0 GB |

**Peak memory (10 GB file):** ~640 MB Python/torch/model + ~300 MB workers + ~170 MB uploads +
~100 MB SQLite + ~1.0 GB Qdrant's resident vectors ≈ **~2.2 GB**, split across two containers.

### 2. Interrupted uploads

`GET /status` reports `bytes_received`; the client resumes from there — no separate token.
`bytes_received` is committed per chunk, so this survives a process crash, not just a dropped
connection. A mismatched offset gets `409` naming the correct one, making retries idempotent. The
chunker's own state (a trailing partial line) is persisted too, so resume continues mid-line without
losing or duplicating text.

Completion is explicit (`POST /complete`), not inferred from `bytes_received >= total_size` — a
client-declared, unverified number. `total_size` is only a sizing hint; it's corrected to the real
byte count once `/complete` runs, and a client may send more than it originally declared.

### 3. Multiple concurrent uploads

Each upload is independent: its own `.partial` file, line-buffer state, and Qdrant collection. A
fixed worker pool caps concurrent embedding regardless of how many uploads are in flight, and SQLite
in WAL mode lets uploads write while workers read.

Two requests for the *same* file at the *same* offset — what a client's retry logic produces when a
response is lost after the server already applied it — are serialized, not raced: the whole
read-check-append-write sequence for one chunk runs inside a single `BEGIN IMMEDIATE` transaction,
so a second concurrent request blocks until the first commits, then correctly fails its own offset
check instead of double-appending.

Identity is a client-supplied `X-User-Id` header, not authentication — the minimum needed to answer
"whose file is this." Files carry an `owner_id`; every route scopes to it, and a file that exists but
belongs to someone else returns `404` rather than `403`.

### 4. Processing and indexing efficiently

Indexing overlaps the upload rather than following it — each arriving chunk is split into passages
and queued immediately, so most of the file is searchable before the last byte lands. The file is
never re-read to build the index; passages are split from bytes already in the request handler's
memory.

Passages align to line boundaries: a network chunk boundary can fall mid-word, mid-line, or
mid-UTF-8-character, so the line buffer holds back the trailing incomplete line and prepends it to
the next chunk.

The `chunks` table is the job queue (`pending → processing → indexed | failed`), durable rather than
in-memory. Workers claim rows atomically; rows stranded by a crash reset to `pending` on startup.
Qdrant point IDs are derived from `(file_id, sequence)`, so re-embedding a passage after a crash
overwrites the same point instead of duplicating it.

### 5. How semantic search works

Passages are embedded with `all-MiniLM-L6-v2` into 384-dimensional vectors that place similar
meaning nearby regardless of wording. A query goes through the same model; Qdrant returns the
nearest passages with byte ranges attached as payload, so a hit maps straight to a file location
with no separate lookup. `database connectivity problems` and `Connection to database failed after
30 seconds` share only "database" — keyword search would rank it poorly — but both describe the same
failure, so the model places them close together. Passages overlap by 20% so a sentence split across
a boundary isn't embedded as two meaningless fragments.

### 6. Scaling to thousands of concurrent uploads and searches

- **Embedding CPU** (~500–1,500 passages/sec on CPU) → a GPU embedding service with aggressive
  batching.
- **In-process workers** → the queue interface (claim/complete/fail/recover) swaps SQLite for
  SQS/Kafka without touching worker logic, enabling many worker processes.
- **SQLite metadata** → Postgres, once several API nodes write concurrently.
- **Local disk** → S3 multipart upload, making API nodes stateless behind a load balancer.
- **Per-file Qdrant collections** → a clustered deployment with sharding/replication; past a certain
  file count, one shared collection with a `file_id` filter beats one-per-file.
- **Embedding volume itself** — a keyword index (FTS5/Elasticsearch) as a cheap first-pass filter,
  reserving embeddings for reranking a small candidate set, cuts embedding cost substantially.

---

## Project layout

| Module | Responsibility |
|---|---|
| [app/upload.py](app/upload.py) | Chunked upload, completion, offset validation, status |
| [app/identity.py](app/identity.py) | `X-User-Id` → caller id, for ownership scoping |
| [app/search.py](app/search.py) | Query embedding, Qdrant lookup, byte-range reads |
| [app/pipeline/line_buffer.py](app/pipeline/line_buffer.py) | Bytes → line-aligned passage ranges |
| [app/pipeline/job_queue.py](app/pipeline/job_queue.py) | Durable queue: claim/complete/fail/recover |
| [app/pipeline/worker.py](app/pipeline/worker.py) | Background embedding threads |
| [app/vector_store.py](app/vector_store.py) | Per-file Qdrant collection, idempotent upserts |
| [app/storage.py](app/storage.py) | On-disk layout, atomic finalize, range reads |

Configuration is environment-driven — see [app/config.py](app/config.py).

---

## Testing

```bash
docker compose --profile test run --rm tests
```

- **`test_line_buffer.py`** — lossless passage tiling under arbitrary splits; UTF-8 boundaries.
- **`test_upload.py`** — chunked upload, interruption, resume, completion, ownership isolation.
- **`test_concurrency.py`** — racing identical chunk requests: exactly one succeeds.
- **`test_job_queue.py`** — exclusive claiming, retry-then-fail, crash recovery.
- **`test_config.py`** — a 10 GB file's resident footprint stays within budget.
- **`test_search.py`** — the assignment's example, ranking, byte offsets, search mid-upload.

Also verified by hand against the running stack: `docker kill` on the API mid-upload → restart →
resume → md5-identical file; Qdrant's on-disk/quantization config confirmed live via `curl`; a
second `X-User-Id` gets `404` on someone else's file from every route; paraphrased queries
(`"running out of storage space"` → `"No space left on device"`, no shared words) retrieving the
right section.

---

## Limitations

- **One API process** — the worker pool is in-process; scaling out means the changes in §6.
- **One Qdrant collection per file** — simple at this scale, but a shared collection with a filter
  is the better trade past a few thousand files.
- **Text files only**, no PDF/DOCX extraction.
- **Identity, not authentication** — `X-User-Id` is trusted as given; a real deployment would put an
  auth layer in front of it.
- **Startup crash-recovery assumes one process** — safe because it runs before the server accepts
  connections; a multi-replica deployment would need a heartbeat instead of inferring liveness from
  a status string.
- **Indexing lags on very large files** — a 10 GB upload finishes indexing minutes after the last
  byte; `/status` reports this honestly.

---

## AI tools used

Built with **Claude Code**. The design was pressure-tested by working through concrete scenarios end
to end rather than reviewing in the abstract — tracing what a 10 GB file's index actually costs in
memory, what a client that mis-declares its own file size does to completion logic, what two
retried requests racing each other do to a file on disk. That process is what produced the current
design: Qdrant over a hand-rolled index (real on-disk, quantized, approximate search instead of one
that only claimed to be), an explicit completion endpoint instead of trusting client-declared size,
a transaction boundary that makes concurrent chunk uploads safe, and lightweight ownership scoping.
Test coverage and the manual verification above reflect the same approach: asserting the actual
property that matters (a file ends up byte-identical after a crash, exactly one of five racing
requests succeeds) rather than that the code merely runs.
