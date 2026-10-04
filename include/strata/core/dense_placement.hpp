#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace strata::core {

// PLE prepares inputs on the first GPU. Its names belong to blk.1, but its
// lifetime and users are model-wide rather than an ordinary transformer layer.
inline int64_t dense_weight_layer(const std::string& name) {
    if (!name.starts_with("blk.") || name.starts_with("blk.1.ple_")) return -1;
    size_t at = 4;
    int64_t layer = 0;
    if (at == name.size() || name[at] < '0' || name[at] > '9') return -1;
    for (; at < name.size() && name[at] >= '0' && name[at] <= '9'; ++at) {
        const int digit = name[at] - '0';
        if (layer > (std::numeric_limits<int64_t>::max() - digit) / 10) return -1;
        layer = layer * 10 + digit;
    }
    return at < name.size() && name[at] == '.' ? layer : -1;
}

inline bool dense_weight_needed(const std::string& name, int64_t lo, int64_t hi) {
    const int64_t layer = dense_weight_layer(name);
    return layer < 0 || (layer >= lo && layer < hi);
}

// Header-only allocation accounting: the split search uses exactly the same
// ownership rule as the canonical and native loaders, before uploading weights.
struct DenseWeightSizes {
    explicit DenseWeightSizes(int64_t count) : layers((size_t) count, 0) {}
    uint64_t shared = 0;
    std::vector<uint64_t> layers;

    void add(const std::string& name, uint64_t bytes) {
        const int64_t layer = dense_weight_layer(name);
        if (layer >= 0 && layer < (int64_t) layers.size()) layers[(size_t) layer] += bytes;
        else shared += bytes;
    }
    uint64_t bytes(int64_t lo, int64_t hi) const {
        uint64_t result = shared;
        for (int64_t l = lo; l < hi; ++l) result += layers.at((size_t) l);
        return result;
    }
};
} // namespace strata::core
