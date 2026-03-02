 #include "vecdb/storage.hpp"

 namespace vecdb {

void InMemoryStorage::put(VectorID id, const Vector& vector) {
   data_[id] = vector;
 }

std::optional<Vector> InMemoryStorage::get(VectorID id) const {
   auto it = data_.find(id);
   if (it == data_.end()) {
     return std::nullopt;
   }
   return it->second;
 }

void InMemoryStorage::erase(VectorID id) {
   data_.erase(id);
 }

 std::size_t InMemoryStorage::size() const {
   return data_.size();
 }

 }  // namespace vecdb
