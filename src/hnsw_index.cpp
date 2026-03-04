#include "vecdb/hnsw_index.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace vecdb {

HNSWIndex::HNSWIndex(int M, int efConstruction, int efSearch)
    : entry_point_(0),
      max_level_(-1),
      M_(M),
      efConstruction_(efConstruction),
      efSearch_(efSearch) {}

float HNSWIndex::l2_distance(const Vector& a, const Vector& b) const {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

int HNSWIndex::sample_level() {
    static std::default_random_engine gen;
    static std::uniform_real_distribution<float> dist(0.0, 1.0);

    int level = 0;
    while (dist(gen) < 0.5f) level++;
    return level;
}

VectorID HNSWIndex::greedy_search(VectorID start,
                                  const Vector& query,
                                  int level) {
    VectorID current = start;
    float current_dist = l2_distance(query, nodes_[current].vector);

    bool changed = true;

    while (changed) {
        changed = false;
        for (VectorID neighbor : nodes_[current].neighbors[level]) {
            float dist = l2_distance(query, nodes_[neighbor].vector);
            if (dist < current_dist) {
                current = neighbor;
                current_dist = dist;
                changed = true;
            }
        }
    }

    return current;
}

std::vector<VectorID> HNSWIndex::search_layer(const Vector& query,
                                               VectorID entry,
                                               size_t ef,
                                               int level) {
    using DistPair = std::pair<float, VectorID>;

    auto cmp_min = [](const DistPair& a, const DistPair& b) {
        return a.first > b.first;
    };

    auto cmp_max = [](const DistPair& a, const DistPair& b) {
        return a.first < b.first;
    };

    std::priority_queue<DistPair, std::vector<DistPair>, decltype(cmp_min)>
        candidate_queue(cmp_min);

    std::priority_queue<DistPair, std::vector<DistPair>, decltype(cmp_max)>
        result_queue(cmp_max);

    std::unordered_set<VectorID> visited;

    float dist = l2_distance(query, nodes_[entry].vector);

    candidate_queue.emplace(dist, entry);
    result_queue.emplace(dist, entry);
    visited.insert(entry);

    while (!candidate_queue.empty()) {

        auto [curr_dist, curr_id] = candidate_queue.top();
        candidate_queue.pop();

        if (!result_queue.empty() && curr_dist > result_queue.top().first)
            break;

        for (VectorID neighbor : nodes_[curr_id].neighbors[level]) {

            if (visited.count(neighbor)) continue;
            visited.insert(neighbor);

            float d = l2_distance(query, nodes_[neighbor].vector);

            if (result_queue.size() < ef ||
                d < result_queue.top().first) {

                candidate_queue.emplace(d, neighbor);
                result_queue.emplace(d, neighbor);

                if (result_queue.size() > ef)
                    result_queue.pop();
            }
        }
    }

    std::vector<DistPair> results;
    while (!result_queue.empty()) {
        results.push_back(result_queue.top());
        result_queue.pop();
    }

    std::sort(results.begin(), results.end());

    std::vector<VectorID> output;
    for (const auto& r : results)
        output.push_back(r.second);

    return output;
}

// Insert a new vector into the HNSW graph.
// 1) Sample a random maximum level for this node.
// 2) Route greedily from the global entry point down through upper layers.
// 3) For each layer from the node's top level down to 0:
//      - run a beam search (efConstruction) to find candidate neighbors,
//      - connect bidirectionally to up to M closest candidates,
//      - prune neighbor lists back to M using cached distances.
// This follows the standard HNSW construction described in the paper.
void HNSWIndex::insert(VectorID id, const Vector& vec) {

    Node node;
    node.id = id;
    node.vector = vec;
    node.level = sample_level();

    nodes_[id] = node;

    if (nodes_.size() == 1) {
        entry_point_ = id;
        max_level_ = node.level;
        return;
    }

    VectorID current = entry_point_;

    // -------- STEP 1: ROUTING ON UPPER LAYERS --------
    for (int level = max_level_; level > node.level; --level) {
        current = greedy_search(current, vec, level);
    }

    // -------- STEP 2: CONSTRUCTION ON EACH LEVEL --------
    for (int level = std::min(node.level, max_level_);
         level >= 0;
         --level) {

        // Find candidate neighbors
        auto candidates = search_layer(
            vec,
            current,
            static_cast<size_t>(efConstruction_),
            level);

        std::sort(candidates.begin(), candidates.end(),
            [&](VectorID a, VectorID b) {
                return l2_distance(vec, nodes_[a].vector)
                     < l2_distance(vec, nodes_[b].vector);
            });

        size_t m = std::min(static_cast<size_t>(M_), candidates.size());

        for (size_t i = 0; i < m; ++i) {

            VectorID neighbor = candidates[i];

            // Bidirectional connect
            nodes_[id].neighbors[level].push_back(neighbor);
            nodes_[neighbor].neighbors[level].push_back(id);
        }

        // -------- Proper Pruning (distance-cached, cheap comparator) --------
        auto& id_neighbors = nodes_[id].neighbors[level];

        if (id_neighbors.size() > static_cast<size_t>(M_)) {
            std::vector<std::pair<float, VectorID>> scored;
            scored.reserve(id_neighbors.size());

            for (auto nid : id_neighbors) {
                float d = l2_distance(nodes_[id].vector,
                                       nodes_[nid].vector);
                scored.emplace_back(d, nid);
            }

            size_t M = static_cast<size_t>(M_);
            std::nth_element(
                scored.begin(),
                scored.begin() + static_cast<std::ptrdiff_t>(M),
                scored.end(),
                [](const auto& a, const auto& b) {
                    return a.first < b.first;
                });

            id_neighbors.clear();
            for (size_t i = 0; i < M; ++i)
                id_neighbors.push_back(scored[i].second);
        }

        for (size_t i = 0; i < m; ++i) {
            VectorID neighbor = candidates[i];

            auto& neigh_list = nodes_[neighbor].neighbors[level];

            if (neigh_list.size() > static_cast<size_t>(M_)) {
                std::vector<std::pair<float, VectorID>> scored;
                scored.reserve(neigh_list.size());

                for (auto nid : neigh_list) {
                    float d = l2_distance(nodes_[neighbor].vector,
                                          nodes_[nid].vector);
                    scored.emplace_back(d, nid);
                }

                size_t M = static_cast<size_t>(M_);
                std::nth_element(
                    scored.begin(),
                    scored.begin() + static_cast<std::ptrdiff_t>(M),
                    scored.end(),
                    [](const auto& a, const auto& b) {
                        return a.first < b.first;
                    });

                neigh_list.clear();
                for (size_t i = 0; i < M; ++i)
                    neigh_list.push_back(scored[i].second);
            }
        }

        // -------- Entry for next level = best candidate --------
        if (!candidates.empty())
            current = candidates[0];
    }

    if (node.level > max_level_) {
        entry_point_ = id;
        max_level_ = node.level;
    }
}

// Search for approximate k-NN of query:
// 1) Greedy descent from the entry point on all upper layers (no backtracking).
// 2) On layer 0, run a beam search with width = efSearch:
//      - candidate_queue explores promising nodes,
//      - result_queue maintains the current efSearch best neighbors.
// 3) Return the closest k IDs from result_queue.
std::vector<VectorID> HNSWIndex::search(const Vector& query, size_t k) {
    if (nodes_.empty()) {
        return {};
    }

    VectorID current = entry_point_;

    for (int level = max_level_; level > 0; --level) {
        current = greedy_search(current, query, level);
    }

    using DistPair = std::pair<float, VectorID>;

    auto cmp_min = [](const DistPair& a, const DistPair& b) {
        return a.first > b.first;
    };
    std::priority_queue<DistPair, std::vector<DistPair>, decltype(cmp_min)>
        candidate_queue(cmp_min);

    auto cmp_max = [](const DistPair& a, const DistPair& b) {
        return a.first < b.first;
    };
    std::priority_queue<DistPair, std::vector<DistPair>, decltype(cmp_max)>
        result_queue(cmp_max);

    std::unordered_set<VectorID> visited;

    float dist = l2_distance(query, nodes_[current].vector);

    candidate_queue.emplace(dist, current);
    result_queue.emplace(dist, current);
    visited.insert(current);

    const size_t ef = static_cast<size_t>(efSearch_);

    while (!candidate_queue.empty()) {

        auto [curr_dist, curr_id] = candidate_queue.top();
        candidate_queue.pop();

        if (!result_queue.empty() && curr_dist > result_queue.top().first)
            break;

        for (VectorID neighbor : nodes_[curr_id].neighbors[0]) {

            if (visited.count(neighbor)) continue;
            visited.insert(neighbor);

            float d = l2_distance(query, nodes_[neighbor].vector);

            if (result_queue.size() < ef ||
                d < result_queue.top().first) {

                candidate_queue.emplace(d, neighbor);
                result_queue.emplace(d, neighbor);

                if (result_queue.size() > ef)
                    result_queue.pop();
            }
        }
    }

    std::vector<DistPair> results;

    while (!result_queue.empty()) {
        results.push_back(result_queue.top());
        result_queue.pop();
    }

    std::sort(results.begin(), results.end());

    std::vector<VectorID> final_result;
    for (size_t i = 0; i < std::min(k, results.size()); ++i)
        final_result.push_back(results[i].second);

    return final_result;
}

namespace {
constexpr uint32_t kHNSWIndexTypeId = 1;
}  // namespace

std::optional<Vector> HNSWIndex::get_vector(VectorID id) const {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return std::nullopt;
    return it->second.vector;
}

void HNSWIndex::erase(VectorID id) {
    auto it = nodes_.find(id);
    if (it == nodes_.end()) return;

    for (auto& [nid, node] : nodes_) {
        if (nid == id) continue;
        for (auto& [level, neighs] : node.neighbors) {
            neighs.erase(std::remove(neighs.begin(), neighs.end(), id),
                         neighs.end());
        }
    }
    nodes_.erase(it);

    if (nodes_.empty()) {
        entry_point_ = 0;
        max_level_ = -1;
        return;
    }
    if (entry_point_ == id) {
        entry_point_ = nodes_.begin()->first;
    }
    max_level_ = -1;
    for (const auto& [_, node] : nodes_) {
        if (node.level > max_level_) max_level_ = node.level;
    }
}

void HNSWIndex::save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);

    if (!out) {
        throw std::runtime_error("Failed to open file for writing");
    }

    uint32_t type_id = kHNSWIndexTypeId;
    out.write(reinterpret_cast<const char*>(&type_id), sizeof(type_id));

    out.write(reinterpret_cast<const char*>(&entry_point_), sizeof(entry_point_));
    out.write(reinterpret_cast<const char*>(&max_level_), sizeof(max_level_));
    out.write(reinterpret_cast<const char*>(&M_), sizeof(M_));
    out.write(reinterpret_cast<const char*>(&efConstruction_), sizeof(efConstruction_));
    out.write(reinterpret_cast<const char*>(&efSearch_), sizeof(efSearch_));

    size_t node_count = nodes_.size();
    out.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));

    for (const auto& [id, node] : nodes_) {
        (void)id;

        out.write(reinterpret_cast<const char*>(&node.id), sizeof(node.id));
        out.write(reinterpret_cast<const char*>(&node.level), sizeof(node.level));

        size_t dim = node.vector.size();
        out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        out.write(reinterpret_cast<const char*>(node.vector.data()),
                  dim * sizeof(float));

        size_t level_count = node.neighbors.size();
        out.write(reinterpret_cast<const char*>(&level_count), sizeof(level_count));

        for (const auto& [level, neighbors] : node.neighbors) {
            out.write(reinterpret_cast<const char*>(&level), sizeof(level));

            size_t neighbor_count = neighbors.size();
            out.write(reinterpret_cast<const char*>(&neighbor_count),
                      sizeof(neighbor_count));

            if (neighbor_count > 0) {
                out.write(reinterpret_cast<const char*>(neighbors.data()),
                          neighbor_count * sizeof(VectorID));
            }
        }
    }
}

void HNSWIndex::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);

    if (!in) {
        throw std::runtime_error("Failed to open file for reading");
    }

    nodes_.clear();

    uint32_t type_id = 0;
    in.read(reinterpret_cast<char*>(&type_id), sizeof(type_id));
    if (type_id != kHNSWIndexTypeId) {
        // Backward compatibility: old format has no type_id; first 4 bytes are
        // the low 32 bits of entry_point_ (little-endian).
        uint32_t entry_high = 0;
        in.read(reinterpret_cast<char*>(&entry_high), sizeof(entry_high));
        entry_point_ = (uint64_t(type_id)) | (uint64_t(entry_high) << 32);
    } else {
        in.read(reinterpret_cast<char*>(&entry_point_), sizeof(entry_point_));
    }
    in.read(reinterpret_cast<char*>(&max_level_), sizeof(max_level_));
    in.read(reinterpret_cast<char*>(&M_), sizeof(M_));
    in.read(reinterpret_cast<char*>(&efConstruction_), sizeof(efConstruction_));
    in.read(reinterpret_cast<char*>(&efSearch_), sizeof(efSearch_));

    size_t node_count = 0;
    in.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));

    for (size_t i = 0; i < node_count; ++i) {
        Node node;

        in.read(reinterpret_cast<char*>(&node.id), sizeof(node.id));
        in.read(reinterpret_cast<char*>(&node.level), sizeof(node.level));

        size_t dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(dim));

        node.vector.resize(dim);
        if (dim > 0) {
            in.read(reinterpret_cast<char*>(node.vector.data()),
                    dim * sizeof(float));
        }

        size_t level_count = 0;
        in.read(reinterpret_cast<char*>(&level_count), sizeof(level_count));

        for (size_t j = 0; j < level_count; ++j) {
            int level = 0;
            in.read(reinterpret_cast<char*>(&level), sizeof(level));

            size_t neighbor_count = 0;
            in.read(reinterpret_cast<char*>(&neighbor_count),
                    sizeof(neighbor_count));

            std::vector<VectorID> neighbors(neighbor_count);
            if (neighbor_count > 0) {
                in.read(reinterpret_cast<char*>(neighbors.data()),
                        neighbor_count * sizeof(VectorID));
            }

            node.neighbors[level] = std::move(neighbors);
        }

        nodes_[node.id] = std::move(node);
    }
}

}  // namespace vecdb
