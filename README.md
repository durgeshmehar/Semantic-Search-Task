# Large File Processing & Search

A backend service that ingests text files up to **10 GB** on a machine with **4 GB of RAM**,
resumes uploads interrupted mid-transfer, and makes the contents searchable in **natural
language** — matching on meaning rather than keyword overlap.

Searching `database connectivity problems` returns the line
`Connection to database failed after 30 seconds.` even though they share only one word.

---

## Quick start

Everything runs in Docker; nothing needs to be installed locally.

```bash
docker compose up --build
```

The API is then at **http://localhost:8000**, with interactive documentation at
**http://localhost:8000/docs**.

The container is capped at 4 GB (`docker-compose.yml`), so the memory constraint is enforced
rather than assumed — an accidental full-file read fails here instead of silently passing on a
larger development machine.

### Running the tests

```bash
docker compose --profile test run --rm tests
```

### Try it end to end

```bash
# 1. Make a test file
printf 'INFO Starting server\nERROR Connection to database failed after 30 seconds.\nINFO Ready\n' > sample.log

# 2. Register the upload
FILE_ID=$(curl -s -X POST http://localhost:8000/files \
  -H 'Content-Type: application/json' \
  -d "{\"filename\":\"sample.log\",\"total_size\":$(wc -c < sample.log)}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["file_id"])')

# 3. Send the bytes
curl -s -X PUT "http://localhost:8000/files/$FILE_ID/chunk?offset=0" \
  --data-binary @sample.log > /dev/null

# 4. Wait until indexed
curl -s "http://localhost:8000/files/$FILE_ID/status"

# 5. Search by meaning
curl -s -X POST "http://localhost:8000/files/$FILE_ID/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"database connectivity problems","top_k":3}'
```

### Running without Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## API

Full interactive documentation: **`/docs`** (OpenAPI, generated from the code).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/files` | Register an upload; returns `file_id` |
| `PUT` | `/files/{file_id}/chunk?offset=N` | Upload one chunk (raw body) |
| `GET` | `/files/{file_id}/status` | Upload **and** processing progress |
| `POST` | `/files/{file_id}/search` | Natural-language search |
| `GET` | `/files` | List uploads |
| `DELETE` | `/files/{file_id}` | Delete a file and its index |
| `GET` | `/health` | Liveness, worker state, queue depth |

### `POST /files`

```json
{ "filename": "server.log", "total_size": 10485760 }
```
```json
{ "file_id": "a3f...", "filename": "server.log", "total_size": 10485760,
  "chunk_size": 16777216, "upload_status": "pending" }
```

### `PUT /files/{file_id}/chunk?offset=N`

Raw bytes in the body. `offset` must equal the server's current `bytes_received`.

```json
{ "file_id": "a3f...", "bytes_received": 8388608, "total_size": 10485760,
  "upload_status": "uploading", "chunks_enqueued": 5461 }
```

`409` on offset mismatch — the message names the offset to resume from.
`413` if the chunk exceeds `chunk_size`. `400` if it would exceed `total_size`.

### `GET /files/{file_id}/status`

Upload and processing are reported **independently**, because bytes can be fully received while
indexing is still catching up.

```json
{ "file_id": "a3f...", "bytes_received": 10485760, "total_size": 10485760,
  "upload_status": "completed", "upload_progress": 1.0,
  "processing_status": "processing", "processing_progress": 0.82,
  "chunks_total": 6982, "chunks_indexed": 5725, "chunks_failed": 0,
  "searchable": true, "error_message": null }
```

This is also the **resume endpoint**: after an interruption, continue from `bytes_received`.

### `POST /files/{file_id}/search`

```json
{ "query": "database connectivity problems", "top_k": 5 }
```
```json
{ "file_id": "a3f...", "query": "database connectivity problems", "total_hits": 3,
  "results": [
    { "text": "ERROR Connection to database failed after 30 seconds.\n",
      "start_byte": 20, "end_byte": 74, "score": 0.62, "sequence": 0 }
  ] }
```

`score` is cosine similarity in `[-1, 1]`. `409` if nothing is indexed yet.
Searching works **during** an upload — whatever is indexed so far is queryable.

---

## Design discussion

### 1. Handling a 10 GB file on a 4 GB machine

The file is never held in memory. Three things make that true:

**Streaming upload.** The client sends the file in chunks (default ceiling 16 MB). Each chunk is
appended to disk and released. Peak memory scales with *concurrent uploads*, not file size — a
100 GB file would use the same memory as a 100 MB one, just take longer.

**Byte ranges instead of text copies.** The uploaded file has to live on disk anyway, so it *is*
the text store. Chunk rows hold `(file_id, start_byte, end_byte)` — 24 bytes rather than ~2 KB of
duplicated text. Storing the text in SQLite too would mean writing the corpus twice (20 GB of disk
for a 10 GB upload) and paying that cost on the upload's critical path. At search time the service
`seek()`s to the offsets of the ~10 results and reads a few KB each.

**A quantized, memory-mapped index.** Vectors are stored as int8 (`IndexScalarQuantizer`) rather
than float32 — 384 B per vector instead of 1,536 B — and read back with `IO_FLAG_MMAP`, so the
resident set is bounded by what searches actually touch rather than by index size. Quantization
costs ~2–3% recall, which does not change which sections a natural-language query returns.

**Passage size scales with file size.** This is the real tension in the design, and it is worth
being explicit about. Small passages embed precisely: one relevant line is a large share of the
passage, so it survives being averaged into a single vector. But small passages mean *more*
vectors — a 10 GB file at the 600 B default would produce ~22M vectors, which is 8 GB even at
int8. That does not fit.

Rather than pick one size that is either imprecise for small files or unaffordable for large ones,
`config.passage_size_for()` scales the target so each file's index stays within a 1 GB budget:

| File size | Passage target | Vectors | Index RAM |
|---|---|---|---|
| 100 MB | 600 B | 218 K | 0.08 GB |
| 1 GB | 600 B | 2.2 M | 0.80 GB |
| 5 GB | 2,400 B | 2.8 M | 1.00 GB |
| 10 GB | 4,800 B | 2.8 M | 1.00 GB |

Typical uploads keep the precise default. Only genuinely huge files are coarsened, and the
honest cost is that recall softens on them — a single line inside a 4.8 KB passage is diluted by
its neighbours. §6 describes the two-stage retrieval that removes this tradeoff at scale.

Measured budget (10 GB file, worst case):

| Component | RAM |
|---|---|
| Python + FastAPI + torch + MiniLM | ~640 MB |
| FAISS resident portion (int8, mmap'd) | ~1.0 GB |
| 2 embedding workers (batch 32) | ~300 MB |
| 10 concurrent uploads @ 16 MB + line buffers | ~170 MB |
| SQLite page cache | ~100 MB |
| **Peak** | **~2.2 GB** |

### 2. Interrupted uploads

There is no separate resume endpoint or resume token. `GET /files/{id}/status` reports
`bytes_received`; the client continues from there.

What makes this robust:

- **`bytes_received` is committed to SQLite** with each chunk, so it survives a process crash, not
  just a dropped connection. Restart the container mid-upload and the client can still resume.
- **Offset validation.** A chunk whose offset ≠ `bytes_received` is rejected with `409` naming the
  correct offset. This makes retries idempotent — re-sending a chunk whose response was lost is
  refused rather than duplicated — and stops out-of-order chunks from leaving holes.
- **Disk is reconciled against the database.** If the process died between writing bytes and
  committing metadata, the file may be ahead of the record. The database is authoritative and the
  file is truncated back before appending.
- **`.partial` → atomic rename.** An upload accumulates in `{file_id}.partial` and is renamed to
  `{file_id}.dat` on completion. The rename is atomic, so a reader never observes a half-written
  file under the final name.
- **The chunker's state is persisted too.** The trailing partial line held back by the line buffer
  is stored on the file row, so an upload resumed after a restart continues mid-line without
  losing or duplicating text.

### 3. Multiple concurrent uploads

Each upload is independent: its own `.partial` file, its own line-buffer state, its own FAISS
index. Nothing is shared on the write path, so uploads never contend.

Two things bound total resource use:

- **A fixed worker pool** (2 by default) caps concurrent embedding regardless of how many uploads
  are in flight. Ten simultaneous uploads produce ten times the queued work, not ten times the
  memory.
- **SQLite in WAL mode** lets the upload path write while workers read, so indexing never blocks
  byte acceptance.

### 4. Processing and indexing efficiently

**Indexing overlaps the upload rather than following it.** Waiting for the upload to finish before
starting to embed would mean two sequential long phases. Instead, each arriving chunk is split
into passages immediately and queued, so by the time the last byte lands most of the file is
already searchable.

**The file is never re-read to build the index.** Passages are split from the bytes already in
memory in the request handler. One sequential write pass, zero full read passes.

**Passages align to line boundaries.** A network chunk boundary falls wherever TCP put it —
mid-word, mid-line, mid-UTF-8-character. The line buffer holds back the trailing incomplete line
(and incomplete multi-byte sequence) and prepends it to the next chunk, so search results are
never sliced mid-sentence. This is the subtlest part of the system and has dedicated tests,
including a fuzz case that feeds a file one byte at a time.

**The upload path stays cheap.** Its only work per chunk is: append bytes, split lines, insert
queue rows. Embedding — the expensive, CPU-bound half — happens in background workers. Accepting
bytes runs at disk speed no matter how far behind indexing is.

**The queue is durable, not in-memory.** The `chunks` table *is* the queue
(`pending → processing → indexed | failed`). Workers claim rows atomically inside an `IMMEDIATE`
transaction, so two workers can never embed the same passage. On startup, rows stranded in
`processing` by a crash are reset to `pending` — the only cost of an unclean shutdown is
re-embedding a handful of passages. A failing passage retries up to a limit, then is marked
`failed` without blocking the file; everything already indexed stays searchable.

### 5. How semantic search works

Passages (600 B by default, with 120 bytes of overlap so a match straddling a boundary is still
findable — see the sizing table in §1) are embedded with `all-MiniLM-L6-v2` into 384-dimensional
vectors. The model maps text to a space where *meaning* determines position, so passages about the
same thing land near each other regardless of wording.

Vectors are L2-normalised at encode time, which makes cosine similarity equal to an inner product
— so FAISS's `IndexFlatIP` computes semantic similarity directly.

A query goes through the same model, FAISS returns the nearest vector positions, those map back to
byte ranges via the `chunks` table, and the service reads the text from the file.

Why the example works: `database connectivity problems` and `Connection to database failed after
30 seconds` share only the word "database". Keyword search would rank this poorly. But both
sentences describe a database connection failure, so the model places them close together and the
passage ranks first.

The overlap between passages matters more than it looks — without it, a sentence split across a
boundary would be embedded as two fragments, neither carrying the full meaning.

### 6. Scaling to thousands of concurrent uploads and searches

The current design targets the stated constraint (4 GB, a handful of users). Here is what breaks
first, in order, and what replaces it:

**Embedding CPU is the first bottleneck.** MiniLM on CPU does ~500–1,500 passages/sec, so a full
10 GB file is 15–35 minutes of compute. At scale this dominates everything else. Move embedding to
a dedicated GPU service with aggressive batching — roughly 100× throughput — and scale it
independently of the API.

**Single-process workers.** The queue interface here is deliberately small (claim / complete /
fail / recover) so the SQLite implementation swaps for SQS or Kafka without touching worker logic.
That allows many worker processes across many machines instead of two threads.

**SQLite for metadata.** Fine for one process; it is not a multi-writer database. Replace with
Postgres once several API nodes write concurrently.

**Local disk for file storage.** This is what ties an upload to one machine. Move to S3 multipart
upload — which also provides resumability natively — and the API nodes become stateless and
horizontally scalable behind a load balancer.

**Per-file FAISS indexes on local disk.** Replace with a managed vector database (Qdrant, Milvus)
that shards and replicates, so search capacity scales independently of upload capacity.

**Embedding volume as the dominant cost.** At very large scale, embedding every passage becomes
the main spend. A two-stage approach cuts it by ~95%: embed coarse ~50 KB sections to find
candidate regions, then embed finely only within the candidates for reranking. Adding a keyword
index (SQLite FTS5 / Elasticsearch) as a cheap first-pass filter serves the same purpose, and also
covers what embeddings are weak at — exact identifiers like error codes and UUIDs.

---

## Architecture

```
Client
  │  PUT /files/{id}/chunk?offset=N
  ▼
┌─────────────────────────────────────────┐
│ Upload handler          (cheap, bounded)│
│  1. append bytes → {id}.partial         │
│  2. split complete lines → passages     │
│  3. INSERT pending rows                 │
│  4. return                              │──► response, no CPU wait
└─────────────────────────────────────────┘
                  │ chunks table = durable queue
                  ▼
┌─────────────────────────────────────────┐
│ Worker pool                (expensive)  │
│  claim → read byte range → embed →      │
│  append to FAISS → mark indexed         │
└─────────────────────────────────────────┘
                  │
                  ▼
       {id}.faiss  +  chunks.vector_position

Search:  query → embed → FAISS top-k → byte ranges → seek() file → text
```

| Module | Responsibility |
|---|---|
| [app/upload.py](app/upload.py) | Chunked upload, offset validation, status |
| [app/search.py](app/search.py) | Query embedding, FAISS lookup, byte-range reads |
| [app/pipeline/line_buffer.py](app/pipeline/line_buffer.py) | Bytes → line-aligned passage ranges |
| [app/pipeline/job_queue.py](app/pipeline/job_queue.py) | Durable queue: claim/complete/fail/recover |
| [app/pipeline/worker.py](app/pipeline/worker.py) | Background embedding threads |
| [app/faiss_index.py](app/faiss_index.py) | Per-file index, quantization, mmap |
| [app/storage.py](app/storage.py) | On-disk layout, atomic finalize, range reads |
| [app/db.py](app/db.py) | SQLite schema, WAL, transactions |

## Configuration

All settings are environment variables (see [app/config.py](app/config.py)):

| Variable | Default | Purpose |
|---|---|---|
| `DATA_DIR` | `data` | Uploads, indexes, database |
| `WORKER_COUNT` | `2` | Embedding threads |
| `EMBED_BATCH_SIZE` | `32` | Passages per forward pass |
| `MAX_CHUNK_BYTES` | `16777216` | Largest accepted chunk |
| `PASSAGE_TARGET_BYTES` | `600` | Target passage size |
| `USE_QUANTIZATION` | `true` | int8 vectors |

## Testing

```bash
docker compose --profile test run --rm tests
```

- **`test_line_buffer.py`** — passages must tile the input losslessly however the stream is split;
  includes a one-byte-at-a-time fuzz case and multi-byte UTF-8 boundaries.
- **`test_upload.py`** — chunked upload, interruption, resume from the reported offset, byte-exact
  reassembly, offset-mismatch rejection.
- **`test_job_queue.py`** — exclusive claiming, retry-then-fail, crash recovery of in-flight rows.
- **`test_config.py`** — adaptive passage sizing keeps a 10 GB file's index inside the memory
  budget while small files keep precise passages.
- **`test_search.py`** — the assignment's example end to end, ranking order, byte offsets that
  locate the returned text, search during an in-progress upload, Unicode content.

Beyond the suite, the following were verified by hand against the running container:

| Check | Result |
|---|---|
| Interrupted upload, resume from reported offset | file md5-identical to the original |
| `docker kill` mid-upload, restart, resume | progress preserved, status reports `interrupted`, md5 matches |
| Wrong offset / replayed chunk | `409` with the correct resume offset, file uncorrupted |
| Indexing during upload | 37 passages searchable while upload was 33% complete |
| `"running out of storage space"` | → `"No space left on device"` (0.611, no shared words) |
| `"wrong password entered"` | → `"Failed login attempt … bad password"` (0.482) |
| `"someone signed in to their account"` | → `"User alice logged in"` (0.453) |
| `"database connectivity problems"` | → `"Connection to database failed after 30 seconds."` |

## Known limitations

Deliberate scope choices, given the assignment targets a 4 GB machine and is explicitly not
required to be production-ready:

- **One API process.** The worker pool and its queue live in-process, so multiple uvicorn workers
  would each start their own pool. Scaling out means the changes in §6.
- **Text files only.** No PDF/DOCX extraction.
- **No authentication.** Any client can read any `file_id`.
- **Indexing lags on very large files.** A full 10 GB upload finishes indexing minutes after the
  last byte. `/status` reports this honestly rather than claiming completion early.
- **Quantization is applied on completion**, so a file searched mid-upload uses the larger flat
  index until then.

## AI tools used

Built with **Claude Code** (Claude Opus 5) as a pair-programming assistant.

**How it was used:** design discussion and trade-off analysis (the queue-durability and
storage-duplication decisions came out of that back-and-forth), then implementation of the modules
above, tests, and this README.

**How the output was validated and changed:**

- **The memory budget was recalculated, and the first estimate was wrong.** An early draft claimed
  the FAISS index for a 10 GB file would be "a few hundred MB". Working through the arithmetic —
  ~1.1M passages × 384 dims × 4 bytes — gives ~1.7 GB, roughly 100× the original figure. That
  error is what drove int8 quantization and memory-mapping into the design.
- **An early design used BM25/FTS5 as the primary retrieval path** with embeddings as a reranker.
  That was rejected on review: the specification asks for matching "based on meaning rather than
  exact keyword matches", and keyword-first retrieval contradicts it regardless of benchmark
  performance. Per-passage embeddings became the primary path.
- **Tests caught a crash.** `_safe_split_point` in the line buffer indexed past the end of the
  buffer when a split landed exactly at its end — found by
  `test_long_line_exceeding_max_is_split_safely`, not by inspection.
- **Running it end to end caught two search-quality bugs that every test had passed.** On the
  first real upload, the assignment's own example query failed to retrieve its target line. Two
  causes, both invisible to tests that only checked coverage:
  1. `_find_cut` took the *last* line break in its window rather than the first past the target,
     so every passage stretched to `max_bytes` (~4 KB). The target line was one line inside 4 KB
     of unrelated startup logs, and its embedding was an average dominated by the noise.
  2. `flush()` re-emitted its own overlap tail as ever-shorter fragments (57 B, then 28 B, then
     14 B). Tiny passages score misleadingly high — a fragment containing two of the query's
     words has high cosine similarity to it — so they crowded out real results.

  Both are now fixed with regression tests that assert passage *size* distribution, not just that
  passages tile the input. This is the clearest argument for verifying behaviour rather than
  trusting a green suite: 43 tests passed while semantic search was quietly broken.
- **The memory budget was recalculated a second time** after those fixes. Shrinking passages to
  600 B for precision pushed a 10 GB file to ~22M vectors (8 GB at int8) — the fix for search
  quality broke the memory constraint. That tension is what motivated
  `config.passage_size_for()`, which scales passage size with file size, and
  `test_config.py`, which asserts a 10 GB file stays inside the budget.
- **Every test here asserts real behaviour**, not that the code merely runs: byte-exact
  reassembly after a simulated interruption, disjoint claims across concurrent workers, recovery
  of rows stranded by a crash, and search results whose byte offsets are verified against the
  original file.
