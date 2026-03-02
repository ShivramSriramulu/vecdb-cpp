#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace vecdb {

using Vector = std::vector<float>;
using VectorID = uint64_t;

using Metadata = std::unordered_map<std::string, std::string>;

}
