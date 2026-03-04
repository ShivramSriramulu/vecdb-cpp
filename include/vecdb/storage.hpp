#pragma once

#include "types.hpp"

#include <optional>
#include <unordered_map>

namespace vecdb {

/**
 * Data layer: stores vectors and optional metadata per id.
 * Used by Collection so that indexing (IIndex) and storage are separate.
 */
class Storage {
public:
    virtual ~Storage() = default;

    virtual void put(VectorID id, const Vector& vector,
                     const Metadata& meta = {}) = 0;
    virtual std::optional<Vector> get(VectorID id) const = 0;
    virtual std::optional<Metadata> get_metadata(VectorID id) const = 0;
    virtual void erase(VectorID id) = 0;
    virtual std::size_t size() const = 0;
};

class InMemoryStorage : public Storage {
public:
    void put(VectorID id, const Vector& vector,
             const Metadata& meta = {}) override;
    std::optional<Vector> get(VectorID id) const override;
    std::optional<Metadata> get_metadata(VectorID id) const override;
    void erase(VectorID id) override;
    std::size_t size() const override;

private:
    std::unordered_map<VectorID, Vector> data_;
    std::unordered_map<VectorID, Metadata> metadata_;
};

}  // namespace vecdb
