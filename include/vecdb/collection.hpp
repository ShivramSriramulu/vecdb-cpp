#pragma once

#include "index.hpp"
#include "storage.hpp"
#include "types.hpp"

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vecdb {

class Collection {
public:
    explicit Collection(std::unique_ptr<IIndex> index);

    /** Construct from loaded index + metadata (e.g. after load()). */
    Collection(std::unique_ptr<IIndex> index,
               std::unordered_map<VectorID, Metadata> metadata);

    void insert(VectorID id, const Vector& vec,
                const Metadata& meta = {});

    /** OLAP: bulk insert (single lock, batch of vectors + optional metadata). */
    void insert_batch(const std::vector<std::pair<VectorID, Vector>>& items,
                      const std::vector<Metadata>& metas = {});

    /** Remove a document from storage and index. */
    void erase(VectorID id);

    /** Replace document: erase then insert. */
    void update(VectorID id, const Vector& vec, const Metadata& meta = {});

    std::vector<VectorID> search(const Vector& query, size_t k);

    std::vector<VectorID> search_filtered(const Vector& query,
                                         size_t k,
                                         const std::string& key,
                                         const std::string& value);

    /** Persist index + metadata to disk (uses IIndex::save, no dynamic_cast). */
    void save(const std::string& path) const;

    /** Load collection from disk (index + metadata). Returns a new Collection. */
    static Collection load(const std::string& path);

    Collection(const Collection&) = delete;
    Collection& operator=(const Collection&) = delete;
    Collection(Collection&&) noexcept;
    Collection& operator=(Collection&&) noexcept;

private:
    /** After load, fill storage from index (e.g. HNSW get_vector). */
    void repopulate_storage_from_index();

    InMemoryStorage storage_;
    std::unique_ptr<IIndex> index_;
    std::unordered_map<VectorID, Metadata> metadata_;
    mutable std::shared_mutex mutex_;
};

}  // namespace vecdb
