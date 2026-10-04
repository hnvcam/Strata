// Header-only model inspection plus the existing engine's allocation sizing.
// Does not initialize CUDA, upload weights, or run inference.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/dense_placement.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/core/session.hpp"
#include "strata/core/layer.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    using namespace strata::core;
    std::string err;
    std::set<std::string> skip;
    std::map<std::string, uint64_t> native, canonical;
    auto shards = strata::gguf_split_paths(argv[2]);
    if (!NativeDense::served_names(shards, true, skip, err, &native) ||
        !NativeDense::keep_unquantized_ple_key(argv[1], skip, err) ||
        !WeightTable::allocation_bytes(argv[1], canonical, err)) {
        std::cerr << err << '\n'; return 1;
    }
    skip.insert("output.weight");
    skip.insert("token_embd.weight");
    DenseWeightSizes dense(48);
    for (auto& [name, bytes] : canonical) if (!skip.count(name)) dense.add(name, bytes);
    for (auto& [name, bytes] : native) dense.add(name, (bytes + 255) / 256 * 256);
    if (!native.empty()) dense.shared += 1ull << 20;
    std::vector<uint64_t> experts(48, 0);
    std::ifstream input(std::string(argv[1]) + "/native_experts.txt");
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') continue;
        int layer, gu, down;
        uint64_t offset, blob;
        std::istringstream row(line);
        row >> layer >> gu >> down >> offset >> blob;
        if (layer < 0 || layer >= 48 || !row) return 1;
        experts[layer] = ((blob + 255) / 256 * 256) * 512;
    }
    ModelGeometry g;
    qsa_set_kv_int8(true);
    std::cout << "n,layers,expert_bytes,dense_bytes,session_bytes,window_reserve_bytes,vram_reserve_bytes,total_before_prompt_context_bytes\n";
    for (int n = 40; n <= 46; ++n) {
        uint64_t eb = 0;
        for (int l = n; l < 47; ++l) eb += experts[l];
        uint64_t db = dense.bytes(n, 47), sb = session_bytes(g, 131072, 10, n, 47);
        uint64_t wb = 96ull << 20, rb = 1248ull << 20;
        std::cout << n << ',' << 47-n << ',' << eb << ',' << db << ',' << sb << ','
                  << wb << ',' << rb << ',' << eb+db+sb+wb+rb << '\n';
    }
}
