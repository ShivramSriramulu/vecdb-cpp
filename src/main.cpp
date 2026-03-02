#include "vecdb/collection.hpp"
#include "vecdb/hnsw_index.hpp"
#include "vecdb/transaction.hpp"
#include "vecdb/wal.hpp"
#include "bruteforce.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <unordered_map>
#include <vector>

using namespace vecdb;

Vector random_vector(size_t dim) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    Vector v(dim);
    for (size_t i = 0; i < dim; ++i) {
        v[i] = dist(gen);
    }
    return v;
}

double compute_recall(const std::vector<VectorID>& ann,
                      const std::vector<VectorID>& gt) {
    if (gt.empty()) return 1.0;
    int match = 0;
    for (auto id : ann) {
        if (std::find(gt.begin(), gt.end(), id) != gt.end())
            match++;
    }
    return static_cast<double>(match) / static_cast<double>(gt.size());
}

void run_benchmark(size_t num_vectors) {
    const size_t dim = 128;
    const size_t num_queries = 100;
    const size_t k = 10;

    auto index = std::make_unique<HNSWIndex>(
        16,   // M
        50,   // efConstruction
        100   // efSearch
    );
    Collection collection(std::move(index));

    std::unordered_map<VectorID, Vector> data;

    std::cout << "\n====================================\n";
    std::cout << "Testing with " << num_vectors << " vectors\n";
    std::cout << "====================================\n";

    auto insert_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_vectors; ++i) {
        Vector v = random_vector(dim);
        collection.insert(i, v);
        data[i] = std::move(v);
    }

    auto insert_end = std::chrono::high_resolution_clock::now();
    auto insert_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        insert_end - insert_start
    ).count();

    std::cout << "Insert time: " << insert_time << " ms\n";

    auto search_start = std::chrono::high_resolution_clock::now();
    double total_recall = 0.0;

    for (size_t i = 0; i < num_queries; ++i) {
        auto query = random_vector(dim);
        auto ann = collection.search(query, k);
        auto gt = brute_force_search(data, query, k);
        total_recall += compute_recall(ann, gt);
    }

    auto search_end = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        search_end - search_start
    ).count();

    std::cout << "Total search time: " << search_time << " ms\n";
    std::cout << "Average per query: "
              << static_cast<double>(search_time) / num_queries
              << " ms\n";
    std::cout << "Recall@" << k << ": "
              << (total_recall / num_queries) << "\n";
}

int main() {
    std::vector<size_t> test_sizes = {5000};  // run 5k first; use {5000, 10000, 20000} for full benchmark

    for (auto size : test_sizes) {
        run_benchmark(size);
    }

    // Persistence test (same params as benchmark)
    const size_t dim = 128;
    const size_t num_vectors = 5000;
    const size_t k = 10;

    HNSWIndex index(16, 50, 100);
    std::vector<Vector> data;
    data.reserve(num_vectors);

    for (size_t i = 0; i < num_vectors; ++i) {
        data.push_back(random_vector(dim));
        index.insert(i, data.back());
    }

    Vector query = random_vector(dim);

    auto original_results = index.search(query, k);

    const std::string filename = "index.bin";
    index.save(filename);

    HNSWIndex loaded;
    loaded.load(filename);

    auto loaded_results = loaded.search(query, k);

    std::cout << "\nPersistence test (original vs loaded)\n";
    std::cout << "Original IDs: ";
    for (auto id : original_results) {
        std::cout << static_cast<unsigned long long>(id) << " ";
    }
    std::cout << "\nLoaded IDs:   ";
    for (auto id : loaded_results) {
        std::cout << static_cast<unsigned long long>(id) << " ";
    }
    std::cout << "\n";

    // -------- Collection save/load with metadata (full persistence) --------
    {
        auto demo_index = std::make_unique<HNSWIndex>(16, 50, 100);
        Collection demo_collection(std::move(demo_index));

        const size_t demo_n = 100;
        for (size_t i = 0; i < demo_n; ++i) {
            Vector v = random_vector(dim);
            Metadata meta;
            meta["category"] = (i % 2 == 0) ? "even" : "odd";
            demo_collection.insert(i, v, meta);
        }

        const std::string collection_path = "demo_collection.bin";
        demo_collection.save(collection_path);

        Collection loaded = Collection::load(collection_path);
        Vector demo_query = random_vector(dim);
        auto filtered = loaded.search_filtered(demo_query, 5, "category", "even");

        std::cout << "\nCollection save/load + metadata: saved to " << collection_path << "\n";
        std::cout << "Filtered search (category=even, k=5) after load: ";
        for (auto id : filtered)
            std::cout << static_cast<unsigned long long>(id) << " ";
        std::cout << "\n";
    }

    // -------- Transaction + WAL demo (durable inserts) --------
    {
        const std::string wal_path = "vecdb.wal";

        auto index = std::make_unique<HNSWIndex>(16, 50, 100);
        Collection tx_collection(std::move(index));
        WAL wal(wal_path);

        // Recover any committed transactions from previous runs.
        wal.replay(tx_collection);

        // Run a new transaction.
        Transaction tx(tx_collection, wal);
        const size_t dim = 16;

        Vector v1 = random_vector(dim);
        Metadata m1;
        m1["source"] = "tx_demo";
        m1["batch"] = "1";
        tx.insert(100000, v1, m1);

        Vector v2 = random_vector(dim);
        Metadata m2;
        m2["source"] = "tx_demo";
        m2["batch"] = "1";
        tx.insert(100001, v2, m2);

        tx.commit();

        Vector q = random_vector(dim);
        auto ids = tx_collection.search_filtered(q, 5, "source", "tx_demo");

        std::cout << "\nTransaction + WAL demo:\n";
        std::cout << "Committed tx_demo vectors, filtered search returned IDs: ";
        for (auto id : ids) {
            std::cout << static_cast<unsigned long long>(id) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nVecDB Build Completed Successfully\n";
    std::cout << "See above for Benchmarks, Persistence, Metadata, WAL demo.\n";

    return 0;
}
