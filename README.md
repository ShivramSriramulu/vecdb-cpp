# VecDb — Vector Database (C++)

VecDB is a minimal standalone vector database in C++17. It uses an **HNSW index** for fast approximate nearest-neighbor search, with persistence, metadata filtering, concurrency control, and crash-safe transactions (WAL). This document explains how to build and run it, how the API and architecture work, and how HNSW compares to a brute-force baseline.

---

## 1. Build and Run

### Prerequisites
- C++17 compiler (e.g. GCC, Clang, MSVC)
- CMake 3.16+

### Build
```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

### Run
- **Benchmark + persistence + metadata demo:**  
  `./vecdb_example`

- **Run tests:**  
  `ctest` (from `build/`) or `cmake --build . && ctest`

---

## 2. API Reference

You talk to VecDB through a **Collection**: you give it an index (usually HNSW), then call insert, search, search_filtered, save, and load. Types and main calls are below.

### Types (`vecdb/types.hpp`)
- `Vector` = `std::vector<float>`
- `VectorID` = `uint64_t`
- `Metadata` = `std::unordered_map<std::string, std::string>`

### Collection (database layer)

```cpp
#include "vecdb/collection.hpp"
#include "vecdb/hnsw_index.hpp"

// Create collection with HNSW index (M, efConstruction, efSearch)
auto index = std::make_unique<vecdb::HNSWIndex>(16, 50, 100);
vecdb::Collection collection(std::move(index));

// Insert vector (optionally with metadata)
collection.insert(id, vector);
collection.insert(id, vector, {{"category", "animal"}, {"color", "black"}});

// k-NN search
std::vector<vecdb::VectorID> ids = collection.search(query_vector, k);

// Filtered search (metadata filter; oversamples then filters)
std::vector<vecdb::VectorID> filtered =
    collection.search_filtered(query_vector, k, "category", "animal");

// OLAP: bulk insert (single lock, fewer round-trips)
std::vector<std::pair<vecdb::VectorID, vecdb::Vector>> items = { {0, v0}, {1, v1}, ... };
std::vector<vecdb::Metadata> metas = { meta0, meta1, ... };
collection.insert_batch(items, metas);

// Remove or update a document
collection.erase(id);
collection.update(id, new_vector, new_metadata);

// Persist full collection (index + metadata) and reload after restart
collection.save("my_collection.bin");
vecdb::Collection loaded = vecdb::Collection::load("my_collection.bin");
```

### Index-only persistence (HNSW, no metadata)

```cpp
vecdb::HNSWIndex index(16, 50, 100);
// ... insert many vectors ...
index.save("index.bin");

// After restart
vecdb::HNSWIndex loaded;
loaded.load("index.bin");
auto results = loaded.search(query, k);
```

### Access patterns (OLTP vs OLAP)
- **OLTP:** Single-record operations: `insert(id, vec, meta)` and `search(query, k)` per request; low latency, concurrent reads via `shared_mutex`.
- **OLAP:** Bulk load: `insert_batch(items, metas)` to ingest many vectors under one lock; then run batch or ad-hoc `search` / `search_filtered` over the built index.

### Data layout
- **Storage:** Data layer (`Storage` / `InMemoryStorage`): vectors + metadata per id; used by Collection for all puts and metadata lookups.
- **Index:** Pluggable via `IIndex` (insert, erase, search, save, load); default is `HNSWIndex` (vectors + graph).
- **Collection:** Public API: owns `InMemoryStorage` + one `IIndex`; exposes insert, insert_batch, erase, update, search, search_filtered, save, load; uses a shared mutex for concurrency.

---

## 3. Design Doc

This section describes how VecDB is built: the main components (Storage, Index, Collection, WAL) and how data and search work.

### Architecture
- **Storage** = data layer: stores vectors and metadata per id. `InMemoryStorage` implements `put(id, vec, meta)`, `get(id)`, `get_metadata(id)`, `erase(id)`. Collection uses it as the single place for document data.
- **IIndex** = search engine interface: `insert`, `erase`, `search`, `save(path)`, `load(path)`. Persistence is pluggable: no `dynamic_cast` in Collection::save; index file starts with a type id for future index types.
- **Collection** = public API: owns `InMemoryStorage` + one `IIndex`; exposes insert, insert_batch, erase, update, search, search_filtered, save, load; handles concurrency (read/write locks). Insert/update write to Storage and Index; search_filtered reads metadata from Storage.
- **HNSWIndex** = HNSW implementation: vectors + graph; implements erase and save/load (with type id in file for backward compatibility).

Separation: **Storage** (data), **IIndex** (indexing), and **Collection** (API) are separate. Collection::save/load call `index_->save(path)` and write metadata to `path.meta`.

### HNSW data structures (for the curious)

- **Node:** each vector is a node with `id`, `vector`, `level`, and `neighbors` per layer.
- **Globals:** entry point, max level, and parameters M, efConstruction, efSearch.
- **Construction:** For each new vector we do greedy routing on upper layers, then at each level run a beam search (efConstruction), connect to the M nearest neighbors, and promote the best candidate to the next level.
- **Search:** Greedy descent on upper layers; on the bottom layer we run a beam search (efSearch) and return the top-k nearest neighbors.

### Persistence strategy

- **Index file:** Binary format: type id (1 = HNSW), then graph parameters and all nodes (id, level, vector, neighbors). Old files without type id still load.
- **Metadata file:** `Collection::save(path)` also writes `path.meta` with key-value metadata per vector. On load, we read both and repopulate Storage so search and filtered search work as before.
- **Usage:** Call `collection.save(path)` before exit; after restart call `Collection::load(path)` to get back the same collection.

---

### 3.1 Architecture Overview

VecDB follows a layered database architecture separating query APIs, indexing, persistence, and durability.

```text
                    ┌──────────────────────┐
                    │      Application     │
                    │  (user / benchmark)  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Collection      │
                    │   (Public API)       │
                    │ • insert/erase/update│
                    │ • search_filtered    │
                    │ • Concurrency locks  │
                    │ • save/load          │
                    └──────────┬───────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  InMemoryStorage │  │    IIndex        │  │       WAL        │
│  (Data layer)    │  │  (e.g. HNSW)     │  │ BEGIN/INSERT/    │
│ • put/get/erase  │  │ • insert/erase   │  │ COMMIT, replay   │
│ • metadata       │  │ • search         │  └──────────────────┘
└──────────────────┘  │ • save/load      │
                      └──────────────────┘
```

#### Component Responsibilities

**Collection (public API)**

Coordinates the database:
- exposes insert, insert_batch, erase, update, search, search_filtered, save, load
- uses **Storage** for vectors and metadata (single data layer)
- uses **IIndex** for ANN search and persistence (no dynamic_cast on save)
- handles concurrency (`std::shared_mutex`)
- integrates transactions and WAL; commit applies the whole batch via one `insert_batch` (atomic)

**Storage (data layer)**  
`InMemoryStorage`: canonical vectors and metadata; Collection reads/writes through it.

**IIndex / HNSWIndex (search engine)**

Responsible only for similarity search:
- hierarchical navigable small-world graph
- greedy routing on upper layers
- beam search (`efSearch`) on base layer
- scalable approximate nearest neighbor queries

The index stores vectors but has no knowledge of metadata or transactions.

**WAL (durability layer)**

Ensures crash safety:
- append-only log
- transactional records: `BEGIN`, `INSERT`, `COMMIT`
- startup replay restores committed operations
- incomplete transactions are ignored

#### Request Flow

**Insert (OLTP)**
- Client → `Collection` → `Storage::put()` + `IIndex::insert()`; with transaction: WAL append then batch apply on commit.

**Search**
- Client → `Collection` → `IIndex::search()` → optional metadata filter (via Storage).

**Recovery**
- Startup → `WAL::replay()` → rebuild `Collection` state

---

## 4. Trade-offs & Future Improvements

We list limitations and possible next steps so the design choices are clear.

### Trade-offs
- **Filtered search:** Implemented as “oversample (k×5) then filter in memory.” No index-level filtering; heavy filtering can hurt latency.
- **Concurrency:** Single writer (insert/insert_batch/erase/update) vs many readers (search) via `shared_mutex`; no multi-versioning.
- **Parameters:** Defaults (e.g. M=16, efConstruction=50, efSearch=100) are tuned for small/medium datasets; large N may need different tuning.
- **Index types:** Only HNSWIndex is implemented; the index file has a type id so additional index types can be added with their own save/load.
- **HNSW delete/update:** Delete and update are implemented (erase from graph + storage; update = erase then insert), but HNSW delete/update is **hard to do correctly** and **expensive**: removing a node leaves "dangling" edges in the graph, and we do not rewire neighbors. The graph can become less accurate over many deletes. A production system would need tombstoning, rewiring, or periodic rebuilds.

### Transactions & WAL
- **Transactions:** Minimal, single-writer transactions via `Transaction`. On commit, all pending inserts are applied in one `insert_batch` so the batch is visible atomically (no partial state).
- **Atomicity and crash semantics:** The "atomicity" claim depends on **crash point and WAL ordering**, not just the single lock. We write `BEGIN`, then `INSERT` records, then `COMMIT`. On replay we only apply transactions that have a matching `COMMIT`; if the process crashes after `INSERT`s but before `COMMIT`, those inserts are not replayed. So atomicity is "all or nothing" only when combined with this replay rule.
- **Durability:** Write-ahead log (`WAL`) with `BEGIN` / `INSERT` / `COMMIT` records; only fully committed transactions are replayed.
- **Crash recovery:** On startup, `WAL::replay` replays committed transactions into a fresh `Collection`, then truncates the WAL to avoid double-application.

### Future improvements
- Optional index-level filtering (e.g. filtered HNSW or secondary index on metadata) for better filtered-search latency.
- Transactions or snapshot isolation for consistent read-after-write.
- More comprehensive tests (property-based, stress, multi-threaded).
- Configurable oversample factor for search_filtered (e.g. k×5 as a parameter).

---

## 5. Benchmark and Brute-Force vs HNSW

### Why we compare brute-force and HNSW

A **naive** vector database would answer k-NN by scanning every vector (brute force). That is simple and **exact** but **does not scale**: search cost grows linearly with the number of vectors. The assignment asks for a **scalable** indexing algorithm. We implement **HNSW** (Hierarchical Navigable Small World) so that search is much faster on large collections, at the cost of **approximate** results. We keep a **brute-force** implementation in the codebase only to compute **ground-truth** results: when we run the benchmark, we compare HNSW’s answers to brute-force’s exact k-NN to report **Recall@k**. So the comparison is there to (1) show that HNSW scales better than brute force, and (2) measure how accurate the approximation is (recall).

### Comparison table (Brute-Force vs HNSW)

| Aspect | Brute-Force | HNSW (this implementation) |
|--------|-------------|-----------------------------|
| **Search time complexity** | O(N · d): must compare query to every vector | O(log N · M · d): graph traversal, sublinear in N |
| **Insert time complexity** | O(1) per vector (no index) | O(log N · M · d) per vector (graph updates) |
| **Recall** | 1.0 (exact k-NN) | Approximate (e.g. ~0.77 Recall@10 in our benchmark) |
| **Scalability** | Poor: search gets slower as N grows | Good: search stays fast as N grows |
| **Use in VecDB** | Used only to compute ground truth for recall; not exposed in the public API | Default index; used for all search in the Collection API |

*N = number of vectors, d = dimension, M = HNSW graph degree.*

In our benchmark run (5k vectors, dim=128, 100 queries, k=10), **HNSW** gave about **4.87 ms per query** and **Recall@10 ≈ 0.77**. A brute-force search over 5k vectors would be exact (Recall 1.0) but would do 5,000 distance computations per query and would get much slower as N grows; HNSW keeps search fast by using the graph instead of scanning all vectors.

### How to run the benchmark

Run `./vecdb_example` to get:

- **Insert time** (ms) for 5k vectors, dim=128.
- **Search time** total and per query (100 queries, k=10).
- **Recall@10** (HNSW results compared to brute-force ground truth).
- **Persistence check:** save index, load, compare search results.
- **Collection save/load with metadata:** save full collection, load, run filtered search.
- **Filtered search demo:** 100 vectors with `category` even/odd, query with `category=even`.
- **WAL transaction demo:** commit a small transactional batch and verify via filtered search.

**Latest run (build + ctest + vecdb_example):**

Build:
```text
[ 63%] Built target vecdb
[ 81%] Built target vecdb_example
[100%] Built target vecdb_tests
```

Tests (`ctest` / `./vecdb_tests`):
```text
PASS: insert and search (n=200, k=5)
PASS: persistence (save/load, result match)
PASS: search_filtered (metadata filter)
PASS: collection save/load with metadata
PASS: insert_batch (OLAP-style bulk insert)
PASS: WAL transactions replay (BEGIN/INSERT/COMMIT)
All tests passed.
```

Benchmark (`./vecdb_example`, AppleClang 17, 5k vectors, dim=128):

```text
====================================
Testing with 5000 vectors
====================================
Insert time: 8933 ms
Total search time: 487 ms
Average per query: 4.87 ms
Recall@10: 0.766

Persistence test (original vs loaded)
Original IDs: 2714 2546 246 1549 3556 109 771 566 1331 1171 
Loaded IDs:   2714 2546 246 1549 3556 109 771 566 1331 1171 

Collection save/load + metadata: saved to demo_collection.bin
Filtered search (category=even, k=5) after load: 26 94 22 38 50 

Transaction + WAL demo:
Committed tx_demo vectors, filtered search returned IDs: 100000 100001 

VecDB Build Completed Successfully
```

---

## 6. Project layout

- `include/vecdb/` — Public API: `types.hpp`, `index.hpp`, `hnsw_index.hpp`, `collection.hpp`, `storage.hpp`
- `src/` — Implementation: `hnsw_index.cpp`, `collection.cpp`, `storage.cpp`, `bruteforce.cpp` (used for recall evaluation only), `main.cpp` (benchmark + demo)
- `tests/` — Unit and integration tests (see below)

---

## 7. Tests

From the `build/` directory you can run all tests with:

```bash
ctest
# or run the test binary directly:
./vecdb_tests
```

Tests verify:
- **Insert & search:** Insert vectors, run k-NN, check result size and consistency.
- **Persistence:** Build index, save, load, compare search results for the same query.
- **Search filtered:** Metadata filter returns only IDs matching key=value.
- **Collection save/load with metadata:** Save collection (index + metadata), load, compare search results and filtered search.
- **insert_batch:** Bulk insert with metadata; search and search_filtered work.
- **WAL transactions replay:** Commit a transaction, restart, replay WAL; committed inserts are restored.

**Result:** 100% tests passed (1/1 CTest target; 6 test cases inside `vecdb_tests`). See `tests/test_insert_search_persistence.cpp`.
