#include "bruteforce.hpp"

#include <algorithm>

namespace vecdb {

static float l2(const Vector& a, const Vector& b) {
    float s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

std::vector<VectorID> brute_force_search(
    const std::unordered_map<VectorID, Vector>& data,
    const Vector& query,
    size_t k) {
    std::vector<std::pair<float, VectorID>> dists;

    for (auto& [id, vec] : data) {
        dists.emplace_back(l2(query, vec), id);
    }

    std::sort(dists.begin(), dists.end());

    std::vector<VectorID> result;
    for (size_t i = 0; i < std::min(k, dists.size()); ++i) {
        result.push_back(dists[i].second);
    }

    return result;
}

}  // namespace vecdb
