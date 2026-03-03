# VecDb — Vector Database (C++)

VecDB is a minimal standalone vector database implemented in C++17 featuring a hierarchical HNSW index, persistence, metadata filtering, concurrency control, and crash-safe transactions via a write-ahead log (WAL). Built for efficient high-dimensional similarity search.

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
- **Index:** Pluggable via `IIndex`; default implementation is `HNSWIndex` (vectors + graph only).
- **Collection:** Owns one `IIndex` and a `metadata_` map; provides insert, search, and search_filtered with a shared mutex for concurrency.

---

## 3. Design Doc

### Architecture
- **Collection** = database layer: owns index + metadata, exposes insert, insert_batch, search, search_filtered, save, load; handles concurrency (read/write locks).
- **IIndex** = search engine interface: insert(id, vector), search(query, k).
- **HNSWIndex** = HNSW implementation: only vectors and graph; no metadata. Implements save/load for the graph.

Separation: indexing and storage are separate. Collection::save/load persist both the index (via HNSWIndex) and metadata (in a separate `.meta` file).

### HNSW data structures
- **Node:** `VectorID id`, `Vector vector`, `int level`, `unordered_map<int, vector<VectorID>> neighbors` (neighbors per layer).
- **Globals:** `entry_point_`, `max_level_`, `M_`, `efConstruction_`, `efSearch_`.
- **Construction:** Upper layers use greedy routing; at each level we run `search_layer` (efConstruction beam search), sort candidates by distance to the new vector, connect to closest M, prune with distance-cached `nth_element`, then set next-level entry to best candidate (no extra greedy step in the construction loop).
- **Search:** Greedy descent on upper layers; at layer 0, efSearch beam search (candidate + result heaps, visited set), then return top-k.

### Persistence strategy
- **Index:** Binary file: globals (entry_point, max_level, M, efConstruction, efSearch), node count, then per node: id, level, vector dimension + raw floats, then per level: level id, neighbor count, neighbor IDs.
- **Metadata:** `Collection::save(path)` writes the index to `path` and metadata to `path.meta` (binary: per-id key-value pairs). `Collection::load(path)` reads both and returns a new Collection.
- **Usage:** Call `collection.save(path)` then `Collection::load(path)` after restart; search and search_filtered behave as before save.

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
                    │----------------------│
                    │ • Public API         │
                    │ • Metadata store     │
                    │ • Concurrency locks  │
                    │ • Persistence        │
                    │ • Transactions       │
                    └──────────┬───────────┘
                               │
           ┌───────────────────┴───────────────────┐
           ▼                                       ▼
┌──────────────────────┐              ┌──────────────────────┐
│      HNSWIndex       │              │         WAL          │
│----------------------│              │----------------------│
│ • Vector graph       │              │ BEGIN / INSERT       │
│ • ANN search engine  │              │ COMMIT records       │
│ • efSearch beam      │              │ Crash recovery       │
│ • Binary persistence │              │ Replay on startup    │
└──────────────────────┘              └──────────────────────┘
```

#### Component Responsibilities

**Collection (database layer)**

Acts as the database coordinator:
- exposes insert/search APIs
- manages metadata
- handles concurrency (`std::shared_mutex`)
- orchestrates persistence
- integrates transactions and WAL

This layer separates database concerns from indexing logic.

**HNSWIndex (search engine)**

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
- Client → `Collection` → WAL append → `HNSWIndex::insert()`

**Search**
- Client → `Collection` → `HNSWIndex::search()` → metadata filter (optional)

**Recovery**
- Startup → `WAL::replay()` → rebuild `Collection` state

---

## 4. Trade-offs & Future Improvements

### Trade-offs
- **Collection::save/load** requires the index to be HNSWIndex (dynamic_cast); other index types would need their own save/load and a type id in the file for polymorphic load.
- **Filtered search:** Implemented as “oversample (k×5) then filter in memory.” No index-level filtering; heavy filtering can hurt latency.
- **Concurrency:** Single writer (insert/insert_batch) vs many readers (search) via `shared_mutex`; no transactions or multi-versioning.
- **Parameters:** Defaults (e.g. M=16, efConstruction=50, efSearch=100) are tuned for small/medium datasets; large N may need different tuning.
- **Storage vs index separation:** In this implementation the HNSW index owns the vectors and the graph, and `Collection` owns only metadata. There is no dedicated `Storage` layer that owns all data and lets indexes store only IDs, so swapping storage backends or adding more index types would require more refactoring.
- **Pluggable index vs persistence:** The in-memory API is pluggable via `IIndex`, but on-disk persistence is tied to `HNSWIndex` (via `dynamic_cast`). Supporting a different ANN index that can also be saved/loaded would need a new format and some extra plumbing.
- **Future direction (not implemented):** A more textbook design would give `Collection` a `Storage` (vectors + metadata) and one or more `IIndex` implementations that store only IDs and ask `Storage` for vectors. This would make it easier to add delete/update and multiple index types, at the cost of extra complexity.

### Transactions & WAL
- **Transactions:** Minimal, single-writer transactions via `Transaction` (no isolation levels, no MVCC).
- **Durability:** Write-ahead log (`WAL`) with `BEGIN` / `INSERT` / `COMMIT` records; only fully committed transactions are replayed.
- **Crash recovery:** On startup, `WAL::replay` replays committed transactions into a fresh `Collection`, then truncates the WAL to avoid double-application.

### Future improvements
- Optional index-level filtering (e.g. filtered HNSW or secondary index on metadata) for better filtered-search latency.
- Transactions or snapshot isolation for consistent read-after-write.
- More comprehensive tests (property-based, stress, multi-threaded).
- Configurable oversample factor for search_filtered (e.g. k×5 as a parameter).

---

## 5. Benchmark (optional)

Run `./vecdb_example` to get:

- **Insert time** (ms) for 5k vectors, dim=128.
- **Search time** total and per query (100 queries, k=10).
- **Recall@10** vs brute-force ground truth.
- **Persistence check:** save index, load, compare search results.
- **Collection save/load with metadata:** save full collection, load, run filtered search to confirm metadata persisted.
- **Filtered search demo:** 100 vectors with `category` even/odd, query with `category=even`.
- **WAL transaction demo:** commit a small transactional batch and verify it via filtered search.

Single clean run on this machine (AppleClang 17, 5k vectors):

```text
====================================
Testing with 5000 vectors
====================================
Insert time: 8796 ms
Total search time: 518 ms
Average per query: 5.18 ms
Recall@10: 0.766

Persistence test (original vs loaded)
Original IDs: 2714 2546 246 1549 3556 109 771 566 1331 1171 
Loaded IDs:   2714 2546 246 1549 3556 109 771 566 1331 1171 

Collection save/load + metadata: saved to demo_collection.bin
Filtered search (category=even, k=5) after load: 26 94 22 38 50 

Transaction + WAL demo:
Committed tx_demo vectors, filtered search returned IDs: 100000 100001 
```

---

## 6. Project layout

- `include/vecdb/` — Public API: `types.hpp`, `index.hpp`, `hnsw_index.hpp`, `collection.hpp`, `storage.hpp`
- `src/` — Implementation: `hnsw_index.cpp`, `collection.cpp`, `storage.cpp`, `bruteforce.cpp`, `main.cpp` (benchmark + demo)
- `tests/` — Unit and integration tests (see below)

---

## 7. Tests

From `build/`:

```bash
ctest
```

Tests verify:
- **Insert & search:** Insert vectors, run k-NN, check result size and consistency.
- **Persistence:** Build index, save, load, compare search results for the same query.
- **Search filtered:** Metadata filter returns only IDs matching key=value.
- **Collection save/load with metadata:** Save collection (index + metadata), load, compare search results and filtered search.
- **insert_batch:** Bulk insert with metadata; search and search_filtered work.

See `tests/test_insert_search_persistence.cpp`.
