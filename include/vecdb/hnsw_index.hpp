 #pragma once

#include "index.hpp"
#include <unordered_map>
#include <random>
#include <string>
#include <vector>

namespace vecdb {

struct Node {
    VectorID id;
    Vector vector;
    int level;
    std::unordered_map<int, std::vector<VectorID>> neighbors;
};

class HNSWIndex : public IIndex {
public:
    HNSWIndex(int M = 16, int efConstruction = 200, int efSearch = 50);

    void insert(VectorID id, const Vector& vec) override;
    std::vector<VectorID> search(const Vector& query, size_t k) override;

    void save(const std::string& filename) const;
    void load(const std::string& filename);

private:
    std::unordered_map<VectorID, Node> nodes_;

    VectorID entry_point_;
    int max_level_;

    int M_;
    int efConstruction_;
    int efSearch_;

    float l2_distance(const Vector& a, const Vector& b) const;
    int sample_level();
    VectorID greedy_search(VectorID start,
                           const Vector& query,
                           int level);

    std::vector<VectorID> search_layer(const Vector& query,
                                       VectorID entry,
                                       size_t ef,
                                       int level);
};

}
