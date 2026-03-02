#pragma once

#include "vecdb/types.hpp"

#include <vector>

namespace vecdb {

class Collection;
class WAL;

// Minimal single-writer transaction with WAL-backed durability.
// No isolation levels or MVCC; operations are buffered in-memory
// and written to WAL, then applied atomically on commit.
class Transaction {
public:
    Transaction(Collection& collection, WAL& wal);
    ~Transaction();

    void insert(VectorID id, const Vector& vec, const Metadata& meta = {});

    void commit();
    void rollback();

private:
    struct PendingOp {
        VectorID id;
        Vector vec;
        Metadata meta;
    };

    Collection& collection_;
    WAL& wal_;
    bool active_;
    std::vector<PendingOp> pending_;
};

}  // namespace vecdb

