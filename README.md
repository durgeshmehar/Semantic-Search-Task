# Large File Processing & Search

A backend service that ingests text files up to **10 GB** on a machine with **4 GB of RAM**,
resumes interrupted uploads, and searches contents in **natural language** — matching on meaning
rather than keywords.

Searching `database connectivity problems` returns `Connection to database failed after 30
seconds.` even though they share one word.

---

## Setup

Everything runs in Docker; nothing needs installing locally.

```bash
docker compose up --build      # API on :8000, docs at /docs
docker compose --profile test run --rm tests    # 50 tests
```

Without Docker:

```bash
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
| `GET` | `/health` | Liveness, workers, queue depth |

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

**Quantized index.** Vectors are stored int8 (384 B) rather than float32 (1,536 B) via scalar
quantization (`IndexScalarQuantizer`) — each of the 384 numbers rounded independently to one byte.
4× smaller, ~2–3% recall cost. This is the only compression applied; search is still exhaustive
(every query compares against every vector in the file), and the index is fully loaded into memory
on each search rather than genuinely memory-mapped — FAISS's `IO_FLAG_MMAP` only takes effect for
`IndexIVF` variants with an on-disk layout, which this project doesn't use. See §6 for what a real
approximate-search index would add.

**Passage size scales with file size.** This is the real tension. Small passages embed precisely —
one relevant line is a large share of the passage, so it survives being averaged into a single
vector. But small passages mean *more* vectors: a 10 GB file at the 600 B default would produce
~22M vectors, 8 GB even at int8. So `config.passage_size_for()` scales the target to keep each
index within 1 GB:

| File size | Passage target | Vectors | Index RAM |
|---|---|---|---|
| 100 MB | 600 B | 218 K | 0.08 GB |
| 1 GB | 600 B | 2.2 M | 0.80 GB |
| 10 GB | 4,800 B | 2.8 M | 1.00 GB |

Typical uploads keep the precise default; only huge files coarsen, and recall softens on them as a
result. §6 describes what removes that tradeoff at scale.

**Peak memory (10 GB file):** ~640 MB Python/torch/model + ~1.0 GB index + ~300 MB workers +
~170 MB uploads + ~100 MB SQLite ≈ **2.2 GB**.

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

Each upload is independent: its own `.partial` file, line-buffer state, and FAISS index. Nothing is
shared on the write path.

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

### 5. How semantic search works

Passages are embedded with `all-MiniLM-L6-v2` into 384-dimensional vectors. The model maps text to
a space where *meaning* determines position, so passages about the same thing land near each other
regardless of wording. Vectors are L2-normalised, making cosine similarity an inner product — so
FAISS's `IndexFlatIP` computes semantic similarity directly.

A query goes through the same model; FAISS returns nearest vector positions, which map back to byte
ranges via the `chunks` table, and the text is read from the file.

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

**Per-file FAISS indexes** → a managed vector DB (Qdrant, Milvus) that shards and replicates, so
search scales independently of upload.

**Exhaustive search itself.** This project's index (`IndexFlatIP` / `IndexScalarQuantizer`)
compares a query against every stored vector — no bucketing, no approximation. That's fine at a few
million vectors per file but doesn't scale to tens of millions across many files searched
concurrently. The standard fix is **`IndexIVFPQ`**: IVF (inverted file) clusters vectors into
~√n buckets so a query only scans the nearest few (the bucketing idea behind LSH-style approaches),
and PQ (Product Quantization) compresses each vector into a handful of codebook indices — 16–64×
smaller than scalar quantization's flat 4×, versus scanning ~30K vectors instead of millions. It
wasn't used here because IVF/PQ both require training on a representative sample before vectors can
be added, which conflicts with this project's streaming design (passages are indexed continuously
as bytes arrive, before the full distribution is known). At scale, the fix is to buffer flat until
enough vectors exist, train IVF+PQ on that sample, then convert — combined with a real memory-mapped
on-disk layout (`OnDiskInvertedLists`), which is what FAISS's mmap support actually requires.

**Embedding volume as the dominant cost.** Two-stage retrieval cuts it ~95%: embed coarse ~50 KB
sections to find candidate regions, then embed finely only within candidates for reranking. A
keyword index (FTS5, Elasticsearch) as a first-pass filter serves the same purpose and also covers
what embeddings are weak at — exact identifiers like error codes and UUIDs.

---

## Architecture

```
Client ──PUT chunk?offset=N──►  Upload handler          (cheap, bounded)
                                 1. append → {id}.partial
                                 2. split lines → passages
                                 3. INSERT pending rows
                                 4. return  ──────────────► response, no CPU wait
                                          │
                                          ▼  chunks table = durable queue
                                Worker pool               (expensive)
                                 claim → read range → embed →
                                 append to FAISS → mark indexed

Search:  query → embed → FAISS top-k → byte ranges → seek() file → text
```

| Module | Responsibility |
|---|---|
| [app/upload.py](app/upload.py) | Chunked upload, offset validation, status |
| [app/search.py](app/search.py) | Query embedding, FAISS lookup, byte-range reads |
| [app/pipeline/line_buffer.py](app/pipeline/line_buffer.py) | Bytes → line-aligned passage ranges |
| [app/pipeline/job_queue.py](app/pipeline/job_queue.py) | Durable queue: claim/complete/fail/recover |
| [app/pipeline/worker.py](app/pipeline/worker.py) | Background embedding threads |
| [app/faiss_index.py](app/faiss_index.py) | Per-file index, scalar quantization |
| [app/storage.py](app/storage.py) | On-disk layout, atomic finalize, range reads |

Configuration is environment-driven — see [app/config.py](app/config.py) (`WORKER_COUNT`,
`PASSAGE_TARGET_BYTES`, `MAX_CHUNK_BYTES`, `USE_QUANTIZATION`, …).

---

## Testing

```bash
docker compose --profile test run --rm tests      # 50 passed
```

- **`test_line_buffer.py`** — passages tile the input losslessly however the stream is split;
  one-byte-at-a-time fuzzing, multi-byte UTF-8 boundaries, passage-size distribution.
- **`test_upload.py`** — chunked upload, interruption, resume, byte-exact reassembly, offset
  rejection.
- **`test_job_queue.py`** — exclusive claiming, retry-then-fail, crash recovery.
- **`test_config.py`** — a 10 GB file's index stays within the memory budget.
- **`test_search.py`** — the assignment's example end to end, ranking, byte offsets, search during
  upload, Unicode.

Also verified by hand against the running container:

| Check | Result |
|---|---|
| Interrupted upload → resume | file md5-identical to the original |
| `docker kill` mid-upload → restart → resume | progress preserved, md5 matches |
| Wrong / replayed offset | `409` with resume offset, file uncorrupted |
| Indexing during upload | 37 passages searchable at 33% uploaded |
| `"running out of storage space"` | → `"No space left on device"` (0.611, no shared words) |
| `"wrong password entered"` | → `"Failed login attempt … bad password"` (0.482) |

---

## Limitations

Deliberate scope choices, given the 4 GB target and the "not production-ready" note:

- **One API process** — the worker pool is in-process, so multiple uvicorn workers would each start
  their own. Scaling out means the changes in §6.
- **Text files only**; no PDF/DOCX extraction.
- **No authentication** — any client can read any `file_id`.
- **Indexing lags on very large files** — a 10 GB upload finishes indexing minutes after the last
  byte. `/status` reports this honestly.

---

## AI tools used

Built with **Claude Code** (Claude Opus 5) as a pair-programming assistant: design discussion,
implementation, tests, and this README.

How the output was validated and corrected:

- **The memory budget was wrong twice, and arithmetic caught it.** An early draft claimed a 10 GB
  file's index would be "a few hundred MB"; working it through gave ~1.7 GB — ~100× off. That drove
  int8 scalar quantization into the design. Later, shrinking passages for search quality pushed a
  10 GB file to ~22M vectors (8 GB), which is what motivated `config.passage_size_for()`.
- **A memory-mapping claim was asserted without checking FAISS's actual support matrix, and was
  wrong.** The code called `read_index(path, faiss.IO_FLAG_MMAP)` and the README claimed the
  resident set was "bounded by what searches touch." In fact `IO_FLAG_MMAP` only takes effect for
  `IndexIVF` variants backed by `OnDiskInvertedLists`; for the flat/scalar-quantized indexes used
  here, FAISS loads the whole file into memory regardless of the flag — and `search()` re-read the
  index from disk on every call regardless. Caught when asked directly whether LSH, Product
  Quantization, and mmap were actually in use, rather than by the test suite. The flag has been
  removed from the code and the README now states plainly what is and isn't implemented: scalar
  quantization only (not PQ), exhaustive flat search (not IVF/LSH-style bucketing), no real
  memory-mapping — see §6 for what `IndexIVFPQ` would add and why it wasn't built given the
  streaming design.
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
