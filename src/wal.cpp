#include "vecdb/wal.hpp"

#include "vecdb/collection.hpp"
#include "vecdb/types.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace vecdb {

WAL::WAL(const std::string& filename)
    : filename_(filename) {}

void WAL::append(const std::string& record) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream out(filename_, std::ios::app);
    if (!out) {
        return;
    }
    out << record << '\n';
}

void WAL::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream out(filename_, std::ios::trunc);
    (void)out;
}

void WAL::replay(Collection& collection) {
    std::ifstream in(filename_);
    if (!in) return;

    std::string line;
    bool in_tx = false;
    std::vector<std::string> tx_lines;

    while (std::getline(in, line)) {
        if (line == "BEGIN") {
            in_tx = true;
            tx_lines.clear();
        } else if (line == "COMMIT") {
            if (in_tx) {
                for (auto& l : tx_lines) {
                    std::istringstream ss(l);
                    std::string cmd;
                    ss >> cmd;

                    if (cmd == "INSERT") {
                        VectorID id{};
                        size_t dim{};
                        ss >> id >> dim;

                        Vector vec(dim);
                        for (size_t i = 0; i < dim; ++i) {
                            ss >> vec[i];
                        }

                        Metadata meta;
                        std::string token;
                        while (ss >> token) {
                            auto pos = token.find('=');
                            if (pos == std::string::npos) continue;
                            std::string key = token.substr(0, pos);
                            std::string value = token.substr(pos + 1);
                            meta[std::move(key)] = std::move(value);
                        }

                        collection.insert(id, vec, meta);
                    }
                }
            }
            in_tx = false;
        } else {
            if (in_tx) {
                tx_lines.push_back(line);
            }
        }
    }

    // After successful replay, truncate WAL to avoid double-application.
    clear();
}

}  // namespace vecdb

