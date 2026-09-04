# Large File Processing & Search

A backend service that ingests text files up to **10 GB** on a machine with **4 GB of RAM**,
resumes interrupted uploads, and searches contents in **natural language** — matching on meaning
rather than keywords.

Searching `database connectivity problems` returns `Connection to database failed after 30
seconds.` even though they share one word.

---

## Setup

Everything runs in Docker; nothing needs installing locally. `docker compose up` starts two
services — the API and a Qdrant instance for vector storage — and waits for Qdrant to report
healthy before starting the API.

```bash
docker compose up --build      # API on :8000, docs at /docs; Qdrant on :6333
docker compose --profile test run --rm tests    # 50 tests, against a real Qdrant
```

Without Docker, a local Qdrant is still required (there is no in-process fallback):

```bash
docker run -p 6333:6333 qdrant/qdrant:v1.12.1

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app
```

### Try it

```bash
printf 'INFO Starting server\nERROR Connection to database failed after 30 seconds.\nINFO Ready\n' > sample.log

FILE_ID=$(curl -s -X POST http://localhost:8000/files \
  -H 'Content-Type: application/json' \
  -d "{\"filename\":\"sample.log\",\"total_size\":$(wc -c < sample.log)}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["file_id"])')

curl -s -X PUT "http://localhost:8000/files/$FILE_ID/chunk?offset=0" --data-binary @sample.log

curl -s "http://localhost:8000/files/$FILE_ID/status"

curl -s -X POST "http://localhost:8000/files/$FILE_ID/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"database connectivity problems","top_k":3}'
```

---

## API

Interactive documentation: **`/docs`** (OpenAPI, generated from the code).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/files` | Register an upload → `file_id` |
| `PUT` | `/files/{file_id}/chunk?offset=N` | Upload one chunk (raw body) |
| `GET` | `/files/{file_id}/status` | Upload **and** processing progress |
| `POST` | `/files/{file_id}/search` | Natural-language search |
| `GET` | `/files` | List uploads |
| `DELETE` | `/files/{file_id}` | Delete a file and its index |
| `GET` | `/health` | Liveness, workers, queue depth, Qdrant connectivity |

**`POST /files`** — `{"filename": "server.log", "total_size": 10485760}`
→ `{"file_id": "a3f...", "chunk_size": 16777216, "upload_status": "pending"}`

**`PUT /files/{id}/chunk?offset=N`** — raw bytes in the body; `offset` must equal the server's
current `bytes_received`.
→ `{"bytes_received": 8388608, "upload_status": "uploading", "chunks_enqueued": 5461}`

`409` on offset mismatch (the message names the offset to resume from), `413` if the chunk exceeds
`chunk_size`, `400` if it would exceed `total_size`.

**`GET /files/{id}/status`** — upload and processing are reported independently, since bytes can be
fully received while indexing is still catching up. This is also the **resume endpoint**: continue
from `bytes_received`.

```json
{ "bytes_received": 10485760, "total_size": 10485760,
  "upload_status": "completed", "upload_progress": 1.0,
  "processing_status": "processing", "processing_progress": 0.82,
  "chunks_total": 6982, "chunks_indexed": 5725, "chunks_failed": 0,
  "searchable": true, "error_message": null }
```

**`POST /files/{id}/search`** — `{"query": "database connectivity problems", "top_k": 5}`

```json
{ "total_hits": 3, "results": [
    { "text": "ERROR Connection to database failed after 30 seconds.\n",
      "start_byte": 20, "end_byte": 74, "score": 0.62, "sequence": 0 } ] }
```

`score` is cosine similarity in `[-1, 1]`. `409` if nothing is indexed yet. Search works **during**
an upload — whatever is indexed so far is queryable.

---

## Design discussion

### 1. A 10 GB file on a 4 GB machine

The file is never held in memory.

**Streaming upload.** The client sends chunks (16 MB ceiling); each is appended to disk and
released. Peak memory scales with *concurrent uploads*, not file size.

**Byte ranges, not text copies.** The uploaded file must live on disk anyway, so it *is* the text
store. Chunk rows hold `(file_id, start_byte, end_byte)` — 24 bytes rather than ~600 of duplicated
text. Copying text into SQLite would write the corpus twice and put that cost on the upload's
critical path. Search `seek()`s to the offsets of its ~10 results.

**Vector storage is Qdrant, disk-backed with quantized vectors kept resident.** Each file gets its
own collection (`file_<file_id>`), configured with `on_disk: true` for raw vectors and the HNSW
graph, and int8 scalar quantization (`always_ram: true` for the quantized copy only — see
[app/vector_store.py](app/vector_store.py)). A vector costs 384 B quantized versus 1,536 B at
float32 — 4× smaller, ~2–3% recall cost — and only that quantized copy needs to be resident; raw
vectors and the graph structure are read from disk as a search touches them. This is a real,
verified property, not an assumption: `curl localhost:6333/collections/<name>` on a live collection
confirms `on_disk: true` and the quantization config are actually in effect, not merely requested.
Search itself is approximate (HNSW — a graph search that visits a small neighborhood rather than
every vector), not the exhaustive brute-force comparison a naive per-file index would do.

**Passage size scales with file size.** This is the real tension, independent of the storage
engine. Small passages embed precisely — one relevant line is a large share of the passage, so it
survives being averaged into a single vector. But small passages mean *more* vectors: a 10 GB file
at the 600 B default would produce ~22M vectors, ~8.4 GB even quantized. So
`config.passage_size_for()` scales the target to keep each file's resident (quantized) footprint
within 1 GB:

| File size | Passage target | Vectors | Resident quantized |
|---|---|---|---|
| 100 MB | 600 B | 218 K | 0.08 GB |
| 1 GB | 600 B | 2.2 M | 0.80 GB |
| 10 GB | 4,800 B | 2.8 M | 1.00 GB |

Typical uploads keep the precise default; only huge files coarsen, and recall softens on them as a
result.

**Peak memory (10 GB file):** ~640 MB Python/torch/model (API container) + ~300 MB workers +
~170 MB uploads + ~100 MB SQLite + ~1.0 GB resident quantized vectors (Qdrant, its own container
and memory limit) ≈ **~2.2 GB total**, split across two containers rather than one process.

### 2. Interrupted uploads

No separate resume endpoint or token: `GET /status` reports `bytes_received` and the client
continues from there.

- **`bytes_received` is committed per chunk**, so uploads survive a process crash, not just a
  dropped connection.
- **Offset validation.** A mismatched offset gets `409` naming the correct one. Retries are
  idempotent — re-sending a chunk whose response was lost is refused rather than duplicated.
- **Disk is reconciled against the database.** If the process died between writing bytes and
  committing metadata, the file is truncated back to the recorded offset.
- **`.partial` → atomic rename** on completion, so a reader never sees a half-written file under
  the final name.
- **The chunker's state is persisted too** — the trailing partial line is stored on the file row,
  so a resumed upload continues mid-line without losing or duplicating text.

### 3. Multiple concurrent uploads

Each upload is independent: its own `.partial` file, line-buffer state, and Qdrant collection.
Nothing is shared on the write path.

A **fixed worker pool** (2 by default) caps concurrent embedding regardless of uploads in flight,
and **SQLite in WAL mode** lets the upload path write while workers read, so indexing never blocks
byte acceptance.

### 4. Processing and indexing efficiently

**Indexing overlaps the upload rather than following it.** Each arriving chunk is split into
passages and queued immediately, so most of the file is searchable before the last byte lands.

**The file is never re-read to build the index** — passages are split from the bytes already in the
request handler's memory. One sequential write pass, zero full read passes.

**Passages align to line boundaries.** A network chunk boundary falls mid-word, mid-line, even
mid-UTF-8-character. The line buffer holds back the trailing incomplete line and prepends it to the
next chunk, so results are never sliced mid-sentence. This is the subtlest part and has dedicated
tests, including a fuzz case feeding a file one byte at a time.

**The upload path stays cheap:** append bytes, split lines, insert queue rows. Embedding — the
expensive half — happens in background workers.

**The queue is durable, not in-memory.** The `chunks` table *is* the queue (`pending → processing →
indexed | failed`). Workers claim rows atomically inside an `IMMEDIATE` transaction, so two workers
never embed the same passage. On startup, rows stranded in `processing` by a crash reset to
`pending`. A failing passage retries, then is marked `failed` without blocking the file.

**Re-indexing a passage is safe by construction, not just by queue discipline.** Each passage's
Qdrant point ID is derived deterministically from `(file_id, sequence)` (a uuid5), so if a worker
crashes mid-batch and the same passage is claimed and embedded again after restart, the second
upsert overwrites the same point rather than creating a duplicate. This means point uniqueness
doesn't depend solely on the job queue never double-claiming a row — it holds even if that
invariant were ever violated.

### 5. How semantic search works

Passages are embedded with `all-MiniLM-L6-v2` into 384-dimensional vectors. The model maps text to
a space where *meaning* determines position, so passages about the same thing land near each other
regardless of wording. The Qdrant collection is configured for cosine distance directly, so no
manual normalization step is needed on our side.

A query goes through the same model; Qdrant's `query_points` returns the nearest passages by
approximate (HNSW) similarity search, each carrying its byte range and sequence number as payload —
so a hit maps straight to a location in the file with no separate lookup table. The matching text
itself is then read from the uploaded file at that byte range (`app/storage.py`), since the corpus
isn't duplicated into Qdrant's payload or SQLite.

Why the example works: `database connectivity problems` and `Connection to database failed after 30
seconds` share only "database", so keyword search ranks it poorly. Both describe a database
connection failure, so the model places them close together.

Passages overlap by 20%, so a sentence split across a boundary isn't embedded as two fragments
neither of which carries the full meaning.

### 6. Scaling to thousands of concurrent uploads and searches

What breaks first, in order:

**Embedding CPU.** MiniLM on CPU does ~500–1,500 passages/sec, so a 10 GB file is 15–35 minutes of
compute. Move embedding to a GPU service with aggressive batching (~100× throughput), scaled
independently of the API.

**Single-process workers.** The queue interface is deliberately small (claim/complete/fail/recover)
so the SQLite implementation swaps for SQS or Kafka without touching worker logic — many worker
processes instead of two threads.

**SQLite for metadata** → Postgres, once several API nodes write concurrently.

**Local disk storage** → S3 multipart upload (which also provides resumability natively), making
API nodes stateless and horizontally scalable behind a load balancer.

**Per-file Qdrant collections** → a clustered Qdrant deployment (or a managed vector DB) with
sharding and replication, so search scales independently of upload and survives a node failure. One
collection per file is simple and gives clean isolation at this scale, but thousands of files means
thousands of small collections, which has its own overhead — a single collection with a `file_id`
payload filter would likely be the better trade past a certain file count.

**Approximate search was chosen up front here, not deferred to a later migration.** An earlier
version of this project used FAISS with an exhaustive flat index, because FAISS's approximate
option (`IndexIVFPQ`) requires training on a representative vector sample before anything can be
inserted — which conflicts with indexing continuously as an upload streams in. Qdrant's default
index, HNSW, is a graph built incrementally with no training phase, so it fits that streaming
pattern natively; this is the actual reason for using Qdrant rather than FAISS directly, not just
"a vector database is more scalable." At the scale this section is about — many files, each with
tens of millions of vectors, searched concurrently — the next lever is `hnsw_config.m` and
`ef_construct` tuning (graph connectivity vs. build cost) and, past that, product quantization on
top of the current scalar quantization for additional compression.

**Embedding volume as the dominant cost.** Two-stage retrieval cuts it ~95%: embed coarse ~50 KB
sections to find candidate regions, then embed finely only within candidates for reranking. A
keyword index (FTS5, Elasticsearch) as a first-pass filter serves the same purpose and also covers
what embeddings are weak at — exact identifiers like error codes and UUIDs.

---

## Architecture

```
                        API container                              Qdrant container
Client ──PUT chunk?offset=N──►  Upload handler          (cheap, bounded)
                                 1. append → {id}.partial
                                 2. split lines → passages
                                 3. INSERT pending rows
                                 4. return  ──────────────► response, no CPU wait
                                          │
                                          ▼  chunks table = durable queue
                                Worker pool               (expensive)
                                 claim → read range → embed
                                 → upsert (uuid5 point id) ───────►  HNSW index, on-disk vectors,
                                 → mark indexed                      int8 quantized (resident)

Search:  query → embed → ───────────────────────────────────►  approximate top-k
                                                                 (byte range + score in payload)
                          ◄─── byte ranges ── seek() file → text
```

| Module | Responsibility |
|---|---|
| [app/upload.py](app/upload.py) | Chunked upload, offset validation, status |
| [app/search.py](app/search.py) | Query embedding, Qdrant lookup, byte-range reads |
| [app/pipeline/line_buffer.py](app/pipeline/line_buffer.py) | Bytes → line-aligned passage ranges |
| [app/pipeline/job_queue.py](app/pipeline/job_queue.py) | Durable queue: claim/complete/fail/recover |
| [app/pipeline/worker.py](app/pipeline/worker.py) | Background embedding threads |
| [app/vector_store.py](app/vector_store.py) | Per-file Qdrant collection, idempotent upserts, search |
| [app/storage.py](app/storage.py) | On-disk layout, atomic finalize, range reads |

Configuration is environment-driven — see [app/config.py](app/config.py) (`WORKER_COUNT`,
`PASSAGE_TARGET_BYTES`, `MAX_CHUNK_BYTES`, `QDRANT_URL`, `QUANTIZATION_ALWAYS_RAM`, …).

---

## Testing

```bash
docker compose --profile test run --rm tests      # 50 passed, against a real Qdrant
```

- **`test_line_buffer.py`** — passages tile the input losslessly however the stream is split;
  one-byte-at-a-time fuzzing, multi-byte UTF-8 boundaries, passage-size distribution.
- **`test_upload.py`** — chunked upload, interruption, resume, byte-exact reassembly, offset
  rejection.
- **`test_job_queue.py`** — exclusive claiming, retry-then-fail, crash recovery.
- **`test_config.py`** — a 10 GB file's resident quantized-vector footprint stays within budget.
- **`test_search.py`** — the assignment's example end to end, ranking, byte offsets, search during
  upload, Unicode. Runs against the real embedding model and a real Qdrant collection, not mocks.

Also verified by hand against the running two-container stack:

| Check | Result |
|---|---|
| Interrupted upload → resume | file md5-identical to the original |
| `docker kill` on the API container mid-upload → restart → resume | progress preserved (Qdrant, a separate container, was unaffected); md5 matches |
| Wrong / replayed offset | `409` with resume offset, file uncorrupted |
| Indexing during upload | 37 passages searchable at 33% uploaded |
| Qdrant collection config, inspected directly via `curl localhost:6333/collections/<name>` | `on_disk: true` for vectors and HNSW, int8 scalar quantization all confirmed actually applied, not just requested |
| `"running out of storage space"` | → `"No space left on device"` (0.611, no shared words) |
| `"wrong password entered"` | → `"Failed login attempt … bad password"` (0.482) |

---

## Limitations

Deliberate scope choices, given the 4 GB target and the "not production-ready" note:

- **One API process** — the worker pool is in-process (separate from Qdrant, which now runs in its
  own container), so multiple uvicorn workers would each start their own pool. Scaling out means
  the changes in §6.
- **One Qdrant collection per file** — clean isolation and simple deletion at this scale, but
  thousands of files means thousands of small collections; a single collection with a `file_id`
  payload filter would be the better trade at that point.
- **Text files only**; no PDF/DOCX extraction.
- **No authentication** — any client can read any `file_id`.
- **Indexing lags on very large files** — a 10 GB upload finishes indexing minutes after the last
  byte. `/status` reports this honestly.

---

## AI tools used

Built with **Claude Code**.

How the output was validated and corrected:

- **The memory budget was wrong twice, and arithmetic caught it.** An early draft claimed a 10 GB
  file's index would be "a few hundred MB"; working it through gave ~1.7 GB — ~100× off. That drove
  int8 scalar quantization into the design. Later, shrinking passages for search quality pushed a
  10 GB file to ~22M vectors (8 GB), which is what motivated `config.passage_size_for()`.
- **A memory-mapping claim was asserted without checking FAISS's actual support matrix, and was
  wrong — this is what led to migrating off FAISS entirely.** The code called
  `read_index(path, faiss.IO_FLAG_MMAP)` and the README claimed the resident set was "bounded by
  what searches touch." In fact `IO_FLAG_MMAP` only takes effect for `IndexIVF` variants backed by
  `OnDiskInvertedLists`; for the flat/scalar-quantized indexes used at the time, FAISS loaded the
  whole file into memory regardless of the flag, and `search()` re-read the index from disk on
  every call regardless. Caught when asked directly whether LSH, Product Quantization, and mmap
  were actually in use, rather than by the test suite.

  Getting a real version of that property in FAISS means `IndexIVFPQ`, and IVF/PQ both require
  training on a representative vector sample before anything can be inserted — which conflicts with
  this project's streaming design (passages are indexed continuously as bytes arrive, before the
  full distribution is known). Rather than build a buffer-then-train-then-convert workaround, the
  project migrated the vector store to **Qdrant**, whose default index (HNSW) is built
  incrementally with no training phase, so it fits the existing streaming pipeline without changing
  it. On-disk storage and int8 scalar quantization then became collection configuration
  (`app/vector_store.py`) rather than index-management code to write and validate. This was
  verified, not assumed: `curl localhost:6333/collections/<name>` on a live collection during
  manual testing confirmed `on_disk: true` and the quantization config were actually in effect.

  The migration also picked up two smaller improvements the earlier design didn't have: point IDs
  are now derived deterministically from `(file_id, sequence)`, so re-embedding a passage after a
  worker crash safely overwrites the same point instead of relying solely on the job queue to
  prevent duplication; and searches no longer join back to SQLite to map a hit to a byte range,
  since Qdrant carries `start_byte`/`end_byte` as point payload directly.
- **An early design used BM25/FTS5 as the primary retrieval path** with embeddings as a reranker.
  Rejected on review: the spec asks for matching "based on meaning rather than exact keyword
  matches", and keyword-first retrieval contradicts that regardless of benchmark performance.
- **Running it end to end caught two bugs that every test had passed.** On the first real upload the
  assignment's own example query failed to retrieve its target line:
  1. `_find_cut` took the *last* line break in its window rather than the first past the target, so
     passages stretched to ~4 KB and one relevant line was averaged into surrounding noise.
  2. `flush()` re-emitted its own overlap tail as ever-shorter fragments (57 B → 28 B → 14 B). Tiny
     passages score misleadingly high, crowding out real results.

  Both now have regression tests asserting passage *size*, not just that passages tile the input.
  43 tests passed while semantic search was quietly broken — the clearest argument for verifying
  behaviour over trusting a green suite.
