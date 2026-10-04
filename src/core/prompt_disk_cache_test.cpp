#include "strata/core/prompt_disk_cache.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>

using namespace strata::core;
namespace fs = std::filesystem;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
DiskPromptState fixture(size_t payload = 257) {
    DiskPromptState state;
    for (int i = 0; i < 2; ++i) {
        DiskPromptStage stage;
        stage.layer_lo = i * 4; stage.layer_hi = (i + 1) * 4;
        stage.running.ids = {1,2,3}; stage.running.imgs = {{1,45}};
        stage.running.gdn.assign(113, uint8_t(i+1)); stage.running.ple = {7,8,9};
        stage.running.tails = {10,11}; stage.running.dead = {12,13}; stage.running.block_pos = {14,15};
        ConversationKv kv;
        kv.format = 1; kv.cells = 4; kv.heads = 1; kv.head_dim = 64;
        kv.page_size = 4; kv.pooled_rows = 1; kv.idx_dim = 8;
        kv.k.resize(payload, uint8_t(i+2)); kv.v.resize(payload, uint8_t(i+3));
        kv.k_scale = {1,2}; kv.v_scale = {3,4}; kv.pooled = {5,6};
        stage.kv.push_back(std::move(kv)); state.stages.push_back(std::move(stage));
    }
    state.draft = state.stages.back().kv[0]; state.draft.format = 17;
    state.draft.pooled_rows = 0; state.draft.pooled.resize(0);
    state.cvec = false;
    return state;
}
fs::path entry(const fs::path& dir, const std::string& key) { return dir / "prompt-v1" / (key + ".spc"); }
}
int main() {
    check(prompt_disk_sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "SHA-256 empty vector");
    check(prompt_disk_sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256 abc vector");
    check(prompt_disk_sha256(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "SHA-256 multi-block vector");
    const auto root = fs::temp_directory_path() / ("strata-disk-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    const auto identity = prompt_disk_sha256("model A");
    PromptDiskCache cache(root, identity, 100000, 3, 131072);
    std::vector<int64_t> ids{1,2,3,4,5};
    const auto keys = cache.keys(ids, {{1,45}}, false, {2,3,4});
    check(keys.size() == 3 && keys[1].tokens == 3, "all cumulative boundaries hashed");
    const auto other = cache.keys({1,2,3,4,9}, {{1,45}}, false, {2,3,4});
    check(other[2].hash == keys[2].hash, "later tokens do not invalidate earlier prefix");
    const auto changed = cache.keys({9,2,3,4,5}, {{1,45}}, false, {2,3,4});
    check(changed[0].hash != keys[0].hash && changed[2].hash != keys[2].hash, "preceding message invalidates all descendants");
    check(cache.keys(ids,{{3,91}},false,{2})[0].hash == cache.keys(ids,{},false,{2})[0].hash, "later image does not invalidate prefix");
    check(cache.keys(ids,{{1,46}},false,{3})[0].hash != keys[1].hash, "image content changes prefix identity");
    check(cache.keys(ids,{{1,45}},true,{3})[0].hash != keys[1].hash, "steering mode isolates identity");
    std::string error;
    auto state = fixture();
    check(cache.store(keys[1].hash,state,error), "two-device state published");
    const auto bytes = cache.file_bytes(keys[1].hash);
    {
        PromptDiskCache restarted(root,identity,100000,3,131072);
        auto loaded = restarted.load(keys[1].hash,error);
        check(loaded && loaded->stages.size()==2 && !loaded->cvec, "survives cache reconstruction / restart");
        for(size_t i=0;i<2;++i) {
            const auto& a=loaded->stages[i];const auto& b=state.stages[i];
            check(a.layer_lo==b.layer_lo && a.layer_hi==b.layer_hi && a.running.ids==b.running.ids &&
                  a.running.imgs==b.running.imgs && a.running.gdn==b.running.gdn && a.running.ple==b.running.ple &&
                  a.running.tails==b.running.tails && a.running.dead==b.running.dead && a.running.block_pos==b.running.block_pos,
                  "all device running state and metadata preserved");
            check(a.kv[0].k==b.kv[0].k && a.kv[0].v==b.kv[0].v && a.kv[0].k_scale==b.kv[0].k_scale &&
                  a.kv[0].v_scale==b.kv[0].v_scale && a.kv[0].pooled==b.kv[0].pooled, "all device K/V, scales and indexer preserved");
        }
        check(loaded->draft.k==state.draft.k && loaded->draft.v==state.draft.v && loaded->draft.format==17,
              "rotated MTP K/V preserved");
    }
    {
        auto streamed = cache.load_stream(keys[1].hash,error);
        check(bool(streamed) && streamed->stages.size()==2, "full snapshot readable by streaming codec");
        std::vector<uint8_t> bytes(state.stages[1].running.gdn.size());
        check(streamed->read(streamed->stages[1].running[0],bytes.data(),0,bytes.size()) &&
              bytes==state.stages[1].running.gdn, "streaming recurrent-state span exact");
        bytes.resize(state.draft.k.size());
        std::vector<uint8_t> expected(bytes.size());state.draft.k.read(expected.data(),0,expected.size());
        check(streamed->read(streamed->draft.data[0],bytes.data(),0,bytes.size()) && bytes==expected,
              "streaming MTP span exact");
        check(!streamed->read(streamed->draft.data[0],bytes.data(),bytes.size(),1), "stream read respects payload bounds");
    }
    {
        std::fstream f(entry(root,keys[1].hash),std::ios::binary|std::ios::in|std::ios::out);
        f.seekp(300);f.put(char(255));
    }
    check(!cache.load(keys[1].hash,error) && !cache.contains(keys[1].hash) && !error.empty(), "corrupt bytes rejected and removed");
    check(cache.store(keys[1].hash,state,error), "repopulate corrupt entry");
    fs::resize_file(entry(root,keys[1].hash),bytes-1);
    check(!cache.load(keys[1].hash,error), "truncated file rejected");
    check(cache.store(keys[1].hash,state,error), "repopulate truncated entry");
    {
        std::fstream f(entry(root,keys[1].hash),std::ios::binary|std::ios::in|std::ios::out);
        f.seekp(160); // first stage begins after header/identity/key/mode/count; then layer range and token count
        for(int i=0;i<8;++i)f.put(char(255));
    }
    check(!cache.load(keys[1].hash,error), "invalid allocation length safely rejected");
    check(cache.store(keys[1].hash,state,error), "repopulate malformed entry");
    PromptDiskCache foreign(root,prompt_disk_sha256("model B"),100000,3,131072);
    check(foreign.keys(ids,{{1,45}},false,{3})[0].hash!=keys[1].hash, "model identity isolates keys");
    check(!foreign.load(keys[1].hash,error), "foreign model payload rejected");
    check(cache.store(keys[1].hash,state,error), "repopulate foreign entry");
    fs::last_write_time(entry(root,keys[1].hash),fs::file_time_type::clock::now()-std::chrono::hours(73));
    check(!cache.contains(keys[1].hash), "inactivity expiry enforced before read");
    cache.prune();check(!fs::exists(entry(root,keys[1].hash)), "expired entry removed");
    check(cache.store(keys[1].hash,state,error), "repopulate expired entry");
    fs::last_write_time(entry(root,keys[1].hash),fs::file_time_type::clock::now()-std::chrono::hours(49));
    cache.touch(keys[1].hash);
    fs::last_write_time(entry(root,keys[1].hash),fs::last_write_time(entry(root,keys[1].hash))-std::chrono::hours(25));
    cache.prune();check(cache.contains(keys[1].hash), "use refreshes inactivity deadline");
    fs::last_write_time(entry(root,keys[1].hash),fs::file_time_type::clock::now()-std::chrono::hours(1));
    check(cache.store(keys[0].hash,state,error), "second entry saved");
    fs::last_write_time(entry(root,keys[0].hash),fs::file_time_type::clock::now()-std::chrono::hours(2));
    std::ofstream(root/"prompt-v1"/"unrelated.txt") << "keep";
    PromptDiskCache limited(root,identity,bytes+16,3,131072);
    check(limited.contains(keys[1].hash) && !limited.contains(keys[0].hash), "byte cap evicts least recently used entry");
    check(fs::exists(root/"prompt-v1"/"unrelated.txt"), "cleanup preserves unrelated files");
    const auto abandoned=root/"prompt-v1"/(keys[0].hash+".spc.123.456.tmp");
    std::ofstream(abandoned) << "partial";
    fs::last_write_time(abandoned,fs::file_time_type::clock::now()-std::chrono::hours(73));
    limited.prune();check(!fs::exists(abandoned), "expired interrupted write removed");
    PromptDiskCache too_small(root/"small",identity,500,3,131072);
    check(!too_small.store(keys[1].hash,state,error), "oversized entry skipped");
    PromptDiskCache disabled(root/"off",identity,0,3,131072);
    check(!disabled.enabled() && !fs::exists(root/"off"), "disabled cache does not create files");
    std::ofstream(root/"artifact") << "first";
    const auto a=prompt_disk_identity({"kv=int8"},{root/"artifact"});
    std::ofstream(root/"artifact") << "other";
    const auto b=prompt_disk_identity({"kv=int8"},{root/"artifact"});
    check(!a.empty() && a!=b, "artifact changes invalidate identity");
    check(b!=prompt_disk_identity({"kv=q4"},{root/"artifact"}), "runtime settings invalidate identity");
    check(prompt_disk_identity({}, {root/"missing"}).empty(), "missing artifacts disable persistence");
    fs::remove_all(root);
    std::printf("prompt_disk_cache_test: %d checks passed\n",checks);
}
