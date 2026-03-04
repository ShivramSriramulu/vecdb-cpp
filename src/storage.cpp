#include "vecdb/storage.hpp"

namespace vecdb {

void InMemoryStorage::put(VectorID id, const Vector& vector,
                          const Metadata& meta) {
    data_[id] = vector;
    metadata_[id] = meta;
}

std::optional<Vector> InMemoryStorage::get(VectorID id) const {
    auto it = data_.find(id);
    if (it == data_.end()) return std::nullopt;
    return it->second;
}

std::optional<Metadata> InMemoryStorage::get_metadata(VectorID id) const {
    auto it = metadata_.find(id);
    if (it == metadata_.end()) return std::nullopt;
    return it->second;
}

void InMemoryStorage::erase(VectorID id) {
    data_.erase(id);
    metadata_.erase(id);
}

std::size_t InMemoryStorage::size() const {
    return data_.size();
}

}  // namespace vecdb
