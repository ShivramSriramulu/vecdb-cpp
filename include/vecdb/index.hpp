#pragma once

#include "types.hpp"
#include <string>
#include <vector>

namespace vecdb {

class IIndex {
public:
    virtual void insert(VectorID id, const Vector& vec) = 0;
    virtual void erase(VectorID id) = 0;
    virtual std::vector<VectorID> search(const Vector& query, size_t k) = 0;
    virtual void save(const std::string& path) const = 0;
    virtual void load(const std::string& path) = 0;
    virtual ~IIndex() = default;
};

}  // namespace vecdb
