#include "strata/core/dense_placement.hpp"
#include "strata/core/weights.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
}

int main() {
    using namespace strata::core;
    const auto dir = std::filesystem::temp_directory_path() /
        ("strata-dense-placement-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        // Disjoint two/three-card ranges must retain every layer exactly once.
        for (int l = 0; l < 48; ++l) {
            const std::string name = "blk." + std::to_string(l) + ".attn_q.weight";
            require(dense_weight_needed(name, 0, 2) == (l < 2), "two-layer stage ownership");
            const int owners = (int) dense_weight_needed(name, 0, 2) +
                               (int) dense_weight_needed(name, 2, 24) +
                               (int) dense_weight_needed(name, 24, 48);
            require(owners == 1, "missing or duplicated per-layer weight");
        }
        // Input preparation depends on these even on a GPU that does not own layer 1.
        for (const char* name : {"blk.1.ple_key.weight", "blk.1.ple_value.weight", "blk.1.ple_norm_key.weight",
                                 "blk.1.ple_norm_query.weight", "blk.1.ple_norm_conv.weight", "blk.1.ple_conv1d.weight",
                                 "token_embd.weight", "output_hc_norm.weight", "output.weight"}) {
            require(dense_weight_needed(name, 2, 48), "shared/input weight was omitted");
        }
        require(!dense_weight_needed("blk.10.attn_q.weight", 0, 2), "layer number prefix confusion");
        require(dense_weight_layer("blk.9999999999999999999999.weight") == -1, "overflowing layer name");

        std::filesystem::create_directory(dir);
        std::ofstream(dir / "index.txt") <<
            "# align 256 pool 1536 tensors 4\n"
            "blk.0.attn_q.weight 0 0 0 257 0 257\n"
            "blk.1.attn_q.weight 0 0 257 256 512 256\n"
            "blk.1.ple_key.weight 0 0 513 256 768 256\n"
            "output.weight 0 0 769 257 1024 257\n";
        std::string err;
        std::map<std::string, uint64_t> allocations;
        require(WeightTable::allocation_bytes(dir.string(), allocations, err), err.c_str());
        DenseWeightSizes sizes(48);
        for (const auto& [name, bytes] : allocations) sizes.add(name, bytes);
        require(sizes.bytes(0, 1) == 1280 && sizes.bytes(1, 48) == 1024, "stage allocation accounting");
        std::set<std::string> skip;
        for (const auto& [name, bytes] : allocations) {
            (void) bytes;
            if (!dense_weight_needed(name, 0, 1)) skip.insert(name);
        }
        uint64_t pool = 0;
        require(WeightTable::pool_bytes(dir.string(), pool, err, &skip), err.c_str());
        require(pool == sizes.bytes(0, 1), "planner differs from actual compacted arena");
        // An override occupies native memory only, never its skipped canonical bytes.
        skip.insert("blk.0.attn_q.weight");
        require(WeightTable::pool_bytes(dir.string(), pool, err, &skip), err.c_str());
        require(pool == 768, "canonical override was counted twice");
        std::ofstream(dir / "index.txt") << "# align 256 pool 256 tensors 1\ninvalid\n";
        require(!WeightTable::allocation_bytes(dir.string(), allocations, err), "malformed index accepted");
        std::filesystem::remove_all(dir);
        std::cout << "dense placement: ownership, shared dependencies and allocation accounting passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove_all(dir);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
