 #pragma once

#include "types.hpp"
#include <vector>

namespace vecdb {

class IIndex {
public:
    virtual void insert(VectorID id, const Vector& vec) = 0;
    virtual std::vector<VectorID> search(const Vector& query, size_t k) = 0;
    virtual ~IIndex() = default;
};

}
