# Large File Processing & Search

A backend service that ingests text files up to **10 GB** on a machine with **4 GB of RAM**,
resumes interrupted uploads, and searches contents in **natural language** — matching on meaning
rather than keywords.

Searching `database connectivity problems` returns `Connection to database failed after 30
seconds.` even though they share one word.

---

## Setup

Two services: the API, and Qdrant for vector storage. `docker compose up` waits for Qdrant to be
healthy before starting the API.

```bash
docker compose up --build                       # API on :8000, docs at /docs
docker compose --profile test run --rm tests    # tests, against a real Qdrant
```

Without Docker (Qdrant is still required — no in-process fallback):

```bash
docker run -p 6333:6333 qdrant/qdrant:v1.12.1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app
```

### Try it

Every request carries `X-User-Id` (any string), which scopes the file to its creator — a client
that omits it shares one default identity with every other client that also omits it.

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

## API

Interactive documentation: **`/docs`** (OpenAPI, generated from the code). Every route requires
`X-User-Id` and scopes to it — one user's files are invisible (404, not 403 — see below) to another.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/files` | Register an upload → `file_id` |
| `PUT` | `/files/{file_id}/chunk?offset=N` | Upload one chunk (raw body) |
| `POST` | `/files/{file_id}/complete` | Mark the upload finished |
| `GET` | `/files/{file_id}/status` | Upload **and** processing progress |
| `POST` | `/files/{file_id}/search` | Natural-language search |
| `GET` | `/files` | List your uploads |
| `DELETE` | `/files/{file_id}` | Delete a file and its index |
| `GET` | `/health` | Liveness, workers, queue depth, Qdrant connectivity |

**`PUT /files/{id}/chunk?offset=N`** — raw bytes in the body; `offset` must equal the server's
current `bytes_received`. `409` on mismatch (names the offset to resume from), `413` over the chunk
limit, `400` only if a chunk would exceed the hard `MAX_FILE_BYTES` ceiling — the client's own
declared `total_size` is a sizing hint, not enforced as a cap (see §2).

**`POST /files/{id}/complete`** — call once every chunk has been sent. Idempotent; `409` if no
bytes were ever received. This, not a byte count, is what marks the upload done (§2).

**`GET /files/{id}/status`** — upload and processing reported independently, since bytes can be
fully received while indexing still catches up. Also the **resume endpoint**: continue from
`bytes_received`.

```json
{ "upload_status": "completed", "upload_progress": 1.0,
  "processing_status": "processing", "processing_progress": 0.82,
  "chunks_total": 6982, "chunks_indexed": 5725, "searchable": true }
```

**`POST /files/{id}/search`** — `{"query": "database connectivity problems", "top_k": 5}` →
results carrying `text`, `start_byte`/`end_byte`, and `score` (cosine similarity). `409` if nothing
is indexed yet; search works **during** an upload.

---

## Design discussion

### 1. A 10 GB file on a 4 GB machine

The file is never held in memory. **Streaming upload** — chunks are appended to disk and released;
peak memory scales with concurrent uploads, not file size. **Byte ranges, not text copies** — the
file on disk already has the text, so chunk rows store `(start_byte, end_byte)` rather than
duplicating ~10 GB into the database; search `seek()`s to the offsets of its results.

**Vector storage is Qdrant**, one collection per file, configured with vectors and the HNSW graph
on disk and int8 scalar quantization (4× smaller than float32, ~2–3% recall cost) with only the
quantized copy kept resident. Search is approximate (HNSW), not exhaustive. Verified directly
against a live collection — `curl localhost:6333/collections/<name>` confirms `on_disk: true` and
quantization are actually in effect.

**Passage size scales with file size** (`config.passage_size_for`). Small passages embed
precisely — a relevant line stays a large share of the passage's average — but a 10 GB file at the
default size would produce ~22M vectors (~8 GB quantized). So larger files get proportionally
larger passages, keeping the resident footprint under 1 GB:

| File size | Passage target | Resident quantized |
|---|---|---|
| ≤1 GB | 600 B | ≤0.8 GB |
| 10 GB | 4,800 B | 1.0 GB |

**Peak memory (10 GB file):** ~640 MB Python/torch/model + ~300 MB workers + ~170 MB uploads +
~100 MB SQLite + ~1.0 GB Qdrant's resident vectors ≈ **~2.2 GB**, split across two containers.

### 2. Interrupted uploads

No separate resume token — `GET /status` reports `bytes_received`, and the client continues from
there.

- `bytes_received` is committed per chunk, so uploads survive a process crash, not just a dropped
  connection.
- A mismatched offset gets `409` naming the correct one; retries are idempotent.
- Disk is reconciled against the database if the process died mid-write.
- `.partial` → atomic rename on completion, so a reader never sees a half-written file.
- The chunker's own state (a trailing partial line) is persisted too, so resume continues mid-line
  without losing or duplicating text.

**Completion is an explicit client call, not inferred from `bytes_received >= total_size`.**
`total_size` is client-declared and unverified; trusting it as the completion signal means a client
that mis-states its own file's size — over by a byte, or wildly wrong — leaves the upload stuck in
`uploading` forever, with nothing left to send and no way to finish. `POST /files/{id}/complete` is
the explicit "I am done" signal that doesn't depend on that number being right; `total_size` is
corrected to the real `bytes_received` once completion runs, and it is otherwise only a hint used to
size passages (§1), never a hard cap — a client may legitimately exceed its own earlier estimate.

### 3. Multiple concurrent uploads

Each upload is independent — its own `.partial` file, line-buffer state, and Qdrant collection. A
fixed worker pool (2 by default) caps concurrent embedding regardless of uploads in flight, and
SQLite in WAL mode lets uploads write while workers read.

**Two requests for the *same* file at the *same* offset — the case a client's own retry logic
produces when a response is lost after the server already applied it — are serialized rather than
raced.** The read-check-append-write sequence for one chunk runs inside a single SQLite
transaction; `BEGIN IMMEDIATE` takes the database's write lock before any statement runs, so a
second concurrent request blocks until the first commits, then sees the already-advanced
`bytes_received` and correctly fails its own offset check instead of double-appending. This was a
real bug caught during review, not a hypothetical — see the AI-tools section.

**Identity is a client-supplied `X-User-Id` header**, not real authentication (no login, no
verification of who's asking) — the minimum needed to answer "whose file is this." Files carry an
`owner_id`; listing, status, search, and delete all scope to the caller, and a file that exists but
belongs to someone else returns `404` rather than `403`, so a client can't distinguish "wrong owner"
from "doesn't exist" by probing IDs.

### 4. Processing and indexing efficiently

Indexing **overlaps** the upload rather than following it — each arriving chunk is split into
passages and queued immediately, so most of the file is searchable before the last byte lands. The
file is never re-read to build the index; passages are split from bytes already in the request
handler's memory.

Passages align to **line boundaries** — a network chunk boundary falls mid-word, mid-line, even
mid-UTF-8-character, so the line buffer holds back the trailing incomplete line and prepends it to
the next chunk. The subtlest part of the system; has dedicated tests including a one-byte-at-a-time
fuzz case.

The `chunks` table **is the job queue** (`pending → processing → indexed | failed`), durable rather
than in-memory. Workers claim rows atomically; rows stranded by a crash reset to `pending` on
startup, and a file caught mid-`complete()` (crashed after marking `finalizing` but before the
rename finished) resets to `uploading` so the client's next `/complete` call finishes the job — the
rename itself is a no-op if it already happened. Point IDs in Qdrant are derived from
`(file_id, sequence)`, so re-embedding a passage after a crash safely overwrites the same point
instead of duplicating it.

### 5. How semantic search works

Passages are embedded with `all-MiniLM-L6-v2` into 384-dimensional vectors; the model places text
with similar meaning nearby regardless of wording. A query goes through the same model, and
Qdrant's approximate search returns the nearest passages with byte ranges attached as payload — no
separate lookup needed to locate a hit in the file.

`database connectivity problems` and `Connection to database failed after 30 seconds` share only
"database", so keyword search ranks it poorly; both describe a database failure, so the model
places them close together. Passages overlap by 20% so a sentence split across a boundary isn't
embedded as two meaningless fragments.

### 6. Scaling to thousands of concurrent uploads and searches

What breaks first, and the swap:

- **Embedding CPU** (~500–1,500 passages/sec) → a GPU embedding service with aggressive batching.
- **In-process workers** → the queue interface (claim/complete/fail/recover) already swaps SQLite
  for SQS/Kafka without touching worker logic, enabling many worker processes.
- **SQLite metadata** → Postgres, once several API nodes write concurrently.
- **Local disk** → S3 multipart upload, making API nodes stateless behind a load balancer.
- **Per-file Qdrant collections** → a clustered deployment with sharding/replication; past a
  certain file count, one shared collection with a `file_id` payload filter beats one-per-file.
- **HNSW tuning** (`m`, `ef_construct`) and product quantization on top of the current scalar
  quantization, once tens of millions of vectors per collection are in play.
- **Embedding volume itself** — a keyword index (FTS5/Elasticsearch) as a cheap first-pass filter,
  with embeddings reserved for reranking a small candidate set, cuts embedding cost dramatically.

---

## Architecture

```
API container: upload handler (append bytes, split lines, enqueue) — cheap, non-blocking
             → worker pool (claim, embed, upsert to Qdrant) — expensive, off the request path
Qdrant container: HNSW index, vectors + graph on disk, quantized copy resident

Search: query → embed → Qdrant approximate top-k (byte range + score) → seek() file → text
```

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
docker compose --profile test run --rm tests    # 50 passed, against a real Qdrant
```

- **`test_line_buffer.py`** — lossless passage tiling under arbitrary splits; UTF-8 boundaries.
- **`test_upload.py`** — chunked upload, interruption, resume, byte-exact reassembly, completion
  semantics, ownership isolation.
- **`test_concurrency.py`** — two racing identical chunk requests: exactly one succeeds, the file
  ends up as exactly one copy of the payload.
- **`test_job_queue.py`** — exclusive claiming, retry-then-fail, crash recovery (including a crash
  mid-`complete()`).
- **`test_config.py`** — a 10 GB file's resident footprint stays within budget.
- **`test_search.py`** — the assignment's example, ranking, byte offsets, search mid-upload.

Also verified by hand against the running stack: `docker kill` on the API mid-upload → restart →
resume → md5-identical file (Qdrant, a separate container, was unaffected); Qdrant's
on-disk/quantization config confirmed live via `curl`; ownership isolation (a second `X-User-Id`
gets `404` on someone else's file, from every route); 6 concurrent identical chunk requests →
exactly 1 succeeds, 5 rejected, file size correct; paraphrased queries
(`"running out of storage space"` → `"No space left on device"`, score 0.611, no shared words)
retrieving the right section with no keyword overlap.

---

## Limitations

- **One API process** — the worker pool is in-process; scaling out means the changes in §6.
- **One Qdrant collection per file** — simple at this scale, but a shared collection with a filter
  is the better trade past a few thousand files.
- **Text files only**, no PDF/DOCX extraction.
- **Identity, not authentication** — `X-User-Id` is trusted as given, with no verification of who's
  actually sending it. It stops one careless client from seeing another's files by accident, but not
  a client that deliberately supplies someone else's ID. A real deployment would put a proper auth
  layer in front of it, which is a drop-in replacement for how `get_user_id()` derives the value.
- **Startup crash-recovery assumes one process.** `recover_stuck_jobs()` marks every
  `upload_status='uploading'` row `interrupted` unconditionally, which is only safe because it runs
  during FastAPI's lifespan startup, before the server accepts connections — so no live client can
  be mid-request at that exact moment. A multi-replica deployment sharing this database would need a
  heartbeat or lease instead of inferring liveness from a status string.
- **Indexing lags on very large files** — a 10 GB upload finishes indexing minutes after the last
  byte; `/status` reports this honestly.

---

## AI tools used

Built with **Claude Code**. Notable corrections made along the way:

- **A memory-mapping claim was wrong, and that's why the vector store is Qdrant, not FAISS.** An
  earlier FAISS-based index claimed memory-mapped, bounded-resident search, but the index types in
  use don't actually support that — the whole index loaded into RAM on every search regardless.
  Caught when asked directly whether the claimed techniques were really in use, not by the test
  suite. A real on-disk approximate index in FAISS (`IndexIVFPQ`) requires training on a
  representative sample before insertion, which conflicts with this project's streaming-indexing
  design — so the project moved to Qdrant, whose HNSW index has no training phase. Verified live
  (`curl localhost:6333/collections/<name>`) that on-disk storage and quantization are actually
  in effect.
- **An early design used BM25/keyword search as the primary retrieval path.** Rejected on review —
  the spec asks for matching by meaning, and keyword-first retrieval contradicts that regardless of
  benchmark performance.
- **Running the system end to end caught two bugs that 43 passing tests had missed.** Passage
  chunking took the last line break in its window instead of the first past the target (passages
  ballooned to 4 KB, diluting the one relevant line into surrounding noise), and the final flush
  re-emitted its own overlap as shrinking fragments that scored misleadingly high. Both are fixed
  with regression tests asserting passage *size*, not just that passages tile the input — the
  clearest argument in this project for verifying behavior over trusting a green suite.
- **The memory budget was recalculated twice** after early estimates were off by ~100× and then
  invalidated again by a later precision fix, which is what motivated `passage_size_for()` scaling
  passage size with file size rather than using one fixed value.
- **A dedicated review pass, prompted by a question about a specific line
  (`body = await request.body()`) rather than by test failures, found six real issues** the test
  suite had not caught, because every test controlled both sides of assumptions the code was only
  hoping clients would honor:
  - **A genuine data-corruption race.** Two concurrent chunk PUTs at the same offset — exactly what
    a client's own retry logic produces when a response is lost after the server already applied
    it — could both pass the offset check and both append, duplicating bytes on disk. Fixed by
    moving the entire read-check-append-write sequence inside one SQLite transaction, so the
    database's own write lock serializes the race instead of adding a separate lock object. Verified
    with a test that fires 5 concurrent identical requests and asserts exactly one succeeds — not
    just that the suite stays green, but that the file ends up exactly one copy of the payload, not
    two or five.
  - **Upload completion trusted an unverified client input.** `total_size` decided when an upload
    was "done" (`bytes_received >= total_size`); a client that over-declared its own file's size
    left the upload stuck in `uploading` forever with no bytes left to send. Fixed with an explicit
    `POST /files/{id}/complete`, the standard pattern (S3 multipart upload, tus.io) for exactly this
    reason — `total_size` is now a sizing hint only.
  - **A `.strip()` check silently dropped legitimate content.** Passages of only whitespace (blank
    lines between log entries) were treated the same as a genuinely empty read (file deleted
    mid-flight), so they were marked "indexed" in the database without ever being embedded — a
    quiet, permanent gap in search coverage with no error anywhere. Narrowed to a true-emptiness
    check.
  - **A time-of-check-to-time-of-use gap** between checking whether `.dat` or `.partial` exists and
    opening it, which a search landing at the exact moment an upload's rename runs could turn into
    an unhandled `FileNotFoundError`. Fixed with a one-retry-on-`FileNotFoundError` in the read path.
  - **Adding ownership surfaced the question of who's allowed to write to a file at all**, which is
    what led to finding the concurrency race above rather than assuming "one client, one file" was
    automatically true.

  None of these were found by asking "any bugs?" in the abstract — they came from working through
  what happens when a specific, real-world assumption (a client tells the truth about file size; a
  client never retries; a check-then-use gap never gets hit) turns out false. That's the pattern
  this project's validation leaned on throughout: the earlier chunking bugs, the mmap claim, and
  this pass were all caught by tracing a concrete scenario end to end, not by generic review.
