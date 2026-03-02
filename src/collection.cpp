#include "vecdb/collection.hpp"
#include "vecdb/hnsw_index.hpp"

#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>

namespace vecdb {

namespace {

std::string meta_path(const std::string& path) {
    return path + ".meta";
}

void write_metadata(const std::string& path,
                    const std::unordered_map<VectorID, Metadata>& metadata) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Failed to open metadata file for writing: " + path);

    size_t count = metadata.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    for (const auto& [id, meta] : metadata) {
        out.write(reinterpret_cast<const char*>(&id), sizeof(id));
        size_t num_pairs = meta.size();
        out.write(reinterpret_cast<const char*>(&num_pairs), sizeof(num_pairs));

        for (const auto& [k, v] : meta) {
            size_t len = k.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(k.data(), len);
            len = v.size();
            out.write(reinterpret_cast<const char*>(&len), sizeof(len));
            out.write(v.data(), len);
        }
    }
}

std::unordered_map<VectorID, Metadata> read_metadata(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};  // no metadata file is ok (e.g. old save)

    size_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));

    std::unordered_map<VectorID, Metadata> metadata;
    for (size_t i = 0; i < count; ++i) {
        VectorID id = 0;
        in.read(reinterpret_cast<char*>(&id), sizeof(id));
        size_t num_pairs = 0;
        in.read(reinterpret_cast<char*>(&num_pairs), sizeof(num_pairs));

        Metadata meta;
        for (size_t j = 0; j < num_pairs; ++j) {
            size_t len = 0;
            in.read(reinterpret_cast<char*>(&len), sizeof(len));
            std::string key(len, '\0');
            if (len) in.read(&key[0], len);

            in.read(reinterpret_cast<char*>(&len), sizeof(len));
            std::string val(len, '\0');
            if (len) in.read(&val[0], len);

            meta[std::move(key)] = std::move(val);
        }
        metadata[id] = std::move(meta);
    }
    return metadata;
}

}  // namespace

Collection::Collection(std::unique_ptr<IIndex> index)
    : index_(std::move(index)) {}

Collection::Collection(std::unique_ptr<IIndex> index,
                       std::unordered_map<VectorID, Metadata> metadata)
    : index_(std::move(index)), metadata_(std::move(metadata)) {}

void Collection::insert(VectorID id, const Vector& vec,
                        const Metadata& meta) {
    std::unique_lock lock(mutex_);
    index_->insert(id, vec);
    metadata_[id] = meta;
}

void Collection::insert_batch(const std::vector<std::pair<VectorID, Vector>>& items,
                              const std::vector<Metadata>& metas) {
    std::unique_lock lock(mutex_);
    for (size_t i = 0; i < items.size(); ++i) {
        const auto& [id, vec] = items[i];
        index_->insert(id, vec);
        metadata_[id] = (i < metas.size()) ? metas[i] : Metadata{};
    }
}

std::vector<VectorID> Collection::search(const Vector& query, size_t k) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return index_->search(query, k);
}

std::vector<VectorID> Collection::search_filtered(const Vector& query,
                                                  size_t k,
                                                  const std::string& key,
                                                  const std::string& value) {
    std::shared_lock lock(mutex_);

    // Oversample ANN results (production trick: ANN ≠ exact, so get more candidates)
    auto candidates = index_->search(query, k * 5);

    std::vector<VectorID> result;
    for (auto id : candidates) {
        auto it = metadata_.find(id);
        if (it == metadata_.end()) continue;

        auto meta_it = it->second.find(key);
        if (meta_it == it->second.end()) continue;

        if (meta_it->second == value) {
            result.push_back(id);
            if (result.size() == k) break;
        }
    }
    return result;
}

void Collection::save(const std::string& path) const {
    std::shared_lock lock(mutex_);
    auto* hnsw = dynamic_cast<HNSWIndex*>(index_.get());
    if (!hnsw)
        throw std::runtime_error("Collection::save requires HNSWIndex");
    hnsw->save(path);
    write_metadata(meta_path(path), metadata_);
}

Collection Collection::load(const std::string& path) {
    auto index = std::make_unique<HNSWIndex>();
    index->load(path);
    auto metadata = read_metadata(meta_path(path));
    return Collection(std::move(index), std::move(metadata));
}

}  // namespace vecdb
