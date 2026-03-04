#include "vecdb/transaction.hpp"

#include "vecdb/collection.hpp"
#include "vecdb/wal.hpp"

#include <sstream>

namespace vecdb {

Transaction::Transaction(Collection& collection, WAL& wal)
    : collection_(collection),
      wal_(wal),
      active_(true) {
    wal_.append("BEGIN");
}

Transaction::~Transaction() {
    if (active_) {
        rollback();
    }
}

void Transaction::insert(VectorID id,
                         const Vector& vec,
                         const Metadata& meta) {
    if (!active_) return;

    pending_.push_back(PendingOp{id, vec, meta});

    std::ostringstream ss;
    ss << "INSERT " << id << " " << vec.size();
    for (float v : vec) {
        ss << " " << v;
    }
    for (const auto& [k, v] : meta) {
        ss << " " << k << "=" << v;
    }

    wal_.append(ss.str());
}

void Transaction::commit() {
    if (!active_) return;

    std::vector<std::pair<VectorID, Vector>> items;
    std::vector<Metadata> metas;
    items.reserve(pending_.size());
    metas.reserve(pending_.size());
    for (auto& op : pending_) {
        items.emplace_back(op.id, op.vec);
        metas.push_back(std::move(op.meta));
    }
    collection_.insert_batch(items, metas);

    wal_.append("COMMIT");
    pending_.clear();
    active_ = false;
}

void Transaction::rollback() {
    if (!active_) return;
    pending_.clear();
    active_ = false;
}

}  // namespace vecdb

