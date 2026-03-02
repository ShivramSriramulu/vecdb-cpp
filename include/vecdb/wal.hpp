#pragma once

#include <mutex>
#include <string>

namespace vecdb {

class Collection;  // forward declaration

// Minimal write-ahead log for transactional inserts.
// Format (one record per line):
//   BEGIN
//   INSERT <id> <dim> v0 v1 ... v{dim-1} [key=value ...]
//   COMMIT
class WAL {
public:
    explicit WAL(const std::string& filename);

    // Append a single log record line (without trailing newline).
    void append(const std::string& record);

    // Clear the WAL file (e.g. after successful recovery/checkpoint).
    void clear();

    // Replay committed transactions into the given collection.
    // Uncommitted (missing COMMIT) transactions are ignored.
    void replay(Collection& collection);

private:
    std::string filename_;
    std::mutex mutex_;
};

}  // namespace vecdb

