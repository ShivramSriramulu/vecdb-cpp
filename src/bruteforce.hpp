#pragma once

#include "vecdb/types.hpp"

#include <unordered_map>
#include <vector>

namespace vecdb {

std::vector<VectorID> brute_force_search(
    const std::unordered_map<VectorID, Vector>& data,
    const Vector& query,
    size_t k);

}
