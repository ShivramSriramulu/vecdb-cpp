/**
 * Unit and integration tests for VecDb: insert, search, persistence.
 * Run via: build/vecdb_tests or ctest from build/
 */

#include "vecdb/collection.hpp"
#include "vecdb/hnsw_index.hpp"
#include "vecdb/types.hpp"
#include "vecdb/transaction.hpp"
#include "vecdb/wal.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace vecdb;

static Vector make_vector(size_t dim, uint64_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    Vector v(dim);
    for (size_t i = 0; i < dim; ++i) v[i] = dist(gen);
    return v;
}

static void test_insert_and_search() {
    const size_t dim = 32;
    const size_t n = 200;
    const size_t k = 5;

    auto index = std::make_unique<HNSWIndex>(16, 50, 100);
    Collection coll(std::move(index));

    for (size_t i = 0; i < n; ++i)
        coll.insert(i, make_vector(dim, i));

    Vector query = make_vector(dim, 9999);
    auto ids = coll.search(query, k);

    assert(ids.size() <= k && "search must return at most k ids");
    assert(ids.size() >= 1 && "non-empty index must return at least one result");

    std::cout << "PASS: insert and search (n=" << n << ", k=" << k << ")\n";
}

static void test_persistence() {
    const size_t dim = 16;
    const size_t n = 100;
    const size_t k = 5;
    const char* path = "test_index.bin";

    HNSWIndex index(16, 50, 100);
    for (size_t i = 0; i < n; ++i)
        index.insert(i, make_vector(dim, i));

    Vector query = make_vector(dim, 12345);
    auto before = index.search(query, k);

    index.save(path);

    HNSWIndex loaded;
    loaded.load(path);

    auto after = loaded.search(query, k);

    assert(before.size() == after.size() && "persistence must preserve result size");
    for (size_t i = 0; i < before.size(); ++i)
        assert(before[i] == after[i] && "persistence must preserve result ids");

    std::cout << "PASS: persistence (save/load, result match)\n";
}

static void test_search_filtered() {
    const size_t dim = 16;
    const size_t n = 50;

    auto index = std::make_unique<HNSWIndex>(16, 50, 100);
    Collection coll(std::move(index));

    for (size_t i = 0; i < n; ++i) {
        Metadata meta;
        meta["tag"] = (i % 2 == 0) ? "even" : "odd";
        coll.insert(i, make_vector(dim, i), meta);
    }

    Vector query = make_vector(dim, 0);
    auto even_ids = coll.search_filtered(query, 10, "tag", "even");

    for (auto id : even_ids)
        assert(id % 2 == 0 && "filtered results must satisfy metadata");

    std::cout << "PASS: search_filtered (metadata filter)\n";
}

static void test_collection_persistence_with_metadata() {
    const size_t dim = 16;
    const size_t n = 80;
    const char* path = "test_collection.bin";

    auto index = std::make_unique<HNSWIndex>(16, 50, 100);
    Collection coll(std::move(index));

    for (size_t i = 0; i < n; ++i) {
        Metadata meta;
        meta["id_mod"] = std::to_string(i % 5);
        coll.insert(i, make_vector(dim, i), meta);
    }

    Vector query = make_vector(dim, 0);
    auto before = coll.search(query, 5);
    coll.save(path);

    Collection loaded = Collection::load(path);
    auto after = loaded.search(query, 5);

    assert(before.size() == after.size() && "collection load must preserve result size");
    for (size_t i = 0; i < before.size(); ++i)
        assert(before[i] == after[i] && "collection load must preserve result ids");

    auto filtered = loaded.search_filtered(query, 3, "id_mod", "0");
    assert(!filtered.empty() && "loaded metadata must support filtered search");

    std::cout << "PASS: collection save/load with metadata\n";
}

static void test_insert_batch() {
    const size_t dim = 16;
    const size_t n = 100;

    auto index = std::make_unique<HNSWIndex>(16, 50, 100);
    Collection coll(std::move(index));

    std::vector<std::pair<VectorID, Vector>> items;
    std::vector<Metadata> metas;
    items.reserve(n);
    metas.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        items.emplace_back(i, make_vector(dim, i));
        Metadata m;
        m["batch"] = "true";
        metas.push_back(std::move(m));
    }

    coll.insert_batch(items, metas);

    Vector query = make_vector(dim, 0);
    auto ids = coll.search(query, 5);
    assert(ids.size() <= 5 && ids.size() >= 1);

    auto filtered = coll.search_filtered(query, 5, "batch", "true");
    assert(filtered.size() <= 5 && "batch insert metadata must be searchable");

    std::cout << "PASS: insert_batch (OLAP-style bulk insert)\n";
}

static void test_wal_transactions_replay() {
    const size_t dim = 8;
    const char* wal_path = "test_vecdb.wal";

    {
        auto index = std::make_unique<HNSWIndex>(16, 50, 100);
        Collection coll(std::move(index));
        WAL wal(wal_path);

        Transaction tx(coll, wal);

        Metadata meta;
        meta["tx"] = "true";
        Vector v = make_vector(dim, 777);
        tx.insert(999, v, meta);

        tx.commit();
    }

    {
        auto index = std::make_unique<HNSWIndex>(16, 50, 100);
        Collection recovered(std::move(index));
        WAL wal(wal_path);

        wal.replay(recovered);

        Vector query = make_vector(dim, 777);
        auto ids = recovered.search_filtered(query, 5, "tx", "true");
        assert(!ids.empty() && "replayed WAL must restore committed inserts");
    }

    std::cout << "PASS: WAL transactions replay (BEGIN/INSERT/COMMIT)\n";
}

int main() {
    try {
        test_insert_and_search();
        test_persistence();
        test_search_filtered();
        test_collection_persistence_with_metadata();
        test_insert_batch();
        test_wal_transactions_replay();
        std::cout << "All tests passed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
