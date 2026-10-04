#include "strata/core/chunk_cache_disk.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>

using namespace strata::core;
namespace fs = std::filesystem;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
DiskChunkState fixture(size_t payload = 257) {
    DiskChunkState state;
    for (int i = 0; i < 2; ++i) {
        DiskChunkStage stage;
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
DiskChunkBlob host_blob(std::vector<uint8_t> v, uint64_t reuse) {
    auto buf = std::make_shared<std::vector<uint8_t>>(std::move(v));
    DiskChunkBlob b; b.size = buf->size(); b.reuse = reuse;
    b.read = [buf](void* d, size_t n, size_t at) {
        if (at > buf->size() || n > buf->size() - at) return false;
        std::memcpy(d, buf->data() + at, n); return true;
    };
    return b;
}
std::vector<uint8_t> to_vec(const ConversationBuffer& b) {
    std::vector<uint8_t> v(b.size()); size_t at = 0;
    b.visit(0, b.size(), [&](const uint8_t* p, size_t n, size_t) {
        std::memcpy(v.data() + at, p, n); at += n; return true;
    });
    return v;
}
// A chained child: same fixture with 64 bytes appended to every K/V blob.
DiskChunkState fixture_child() {
    auto state = fixture();
    for (auto& s : state.stages) {
        s.kv[0].k.resize(s.kv[0].k.size() + 64, uint8_t(9));
        s.kv[0].v.resize(s.kv[0].v.size() + 64, uint8_t(9));
    }
    state.draft = state.stages.back().kv[0]; state.draft.format = 17;
    state.draft.pooled_rows = 0; state.draft.pooled.resize(0);
    return state;
}
DiskChunkFile stream_image(const DiskChunkState& st, uint64_t reuse) {
    DiskChunkFile f; f.cvec = st.cvec;
    auto kv = [&](DiskChunkKv& o, const ConversationKv& k) {
        o.format = k.format; o.cells = k.cells; o.heads = k.heads; o.head_dim = k.head_dim;
        o.page_size = k.page_size; o.pooled_rows = k.pooled_rows; o.idx_dim = k.idx_dim;
        o.data[0] = host_blob(to_vec(k.k), reuse); o.data[1] = host_blob(to_vec(k.v), reuse);
        o.data[2] = host_blob(to_vec(k.k_scale), 0); o.data[3] = host_blob(to_vec(k.v_scale), 0);
        o.data[4] = host_blob(to_vec(k.pooled), 0);
    };
    for (const auto& s : st.stages) {
        DiskChunkFileStage o; o.layer_lo = s.layer_lo; o.layer_hi = s.layer_hi;
        o.ids = s.running.ids; o.images = s.running.imgs;
        o.running[0] = host_blob(s.running.gdn, 0);
        o.running[1] = host_blob(s.running.ple, 0);
        o.running[2] = host_blob(s.running.tails, 0);
        o.running[3] = host_blob(s.running.dead, 0);
        o.running[4] = host_blob(s.running.block_pos, 0);
        o.kv.resize(1); kv(o.kv[0], s.kv[0]); f.stages.push_back(std::move(o));
    }
    kv(f.draft, st.draft);
    return f;
}
std::vector<uint8_t> read_all(const DiskChunkFile& f, const DiskChunkBlob& b) {
    std::vector<uint8_t> v(size_t(b.size));
    if (!f.read(b, v.data(), 0, v.size())) v.clear();
    return v;
}
}
int main() {
    check(chunk_cache_disk_sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "SHA-256 empty vector");
    check(chunk_cache_disk_sha256("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256 abc vector");
    check(chunk_cache_disk_sha256(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "SHA-256 multi-block vector");
    const auto root = fs::temp_directory_path() / ("strata-disk-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    const auto identity = chunk_cache_disk_sha256("model A");
    ChunkCacheDisk cache(root, identity, 100000, 3, 131072);
    std::vector<int64_t> ids{1,2,3,4,5};
    const auto keys = cache.keys(ids, {{1,45}}, false, 1, 4);
    check(keys.size() == 4 && keys[2].tokens == 3, "complete chunks hashed through prefix cap");
    const auto other = cache.keys({1,2,3,4,9}, {{1,45}}, false, 1, 4);
    check(other[2].hash == keys[2].hash, "later tokens do not invalidate earlier prefix");
    const auto changed = cache.keys({9,2,3,4,5}, {{1,45}}, false, 1, 4);
    check(changed[0].hash != keys[0].hash && changed[2].hash != keys[2].hash, "preceding message invalidates all descendants");
    check(cache.keys(ids,{{3,91}},false,1,2)[0].hash == cache.keys(ids,{},false,1,2)[0].hash, "later image does not invalidate prefix");
    check(cache.keys(ids,{{1,46}},false,1,3)[2].hash != keys[2].hash, "image content changes prefix identity");
    check(cache.keys(ids,{{1,45}},true,1,3)[2].hash != keys[2].hash, "steering mode isolates identity");
    {
        std::vector<int64_t> prompt(14052);
        for(size_t i=0;i<prompt.size();++i)prompt[i]=int64_t(i%1000);
        const auto chain=cache.keys(prompt,{},true,512,8192);
        check(chain.size()==16 && chain.front().tokens==512 && chain.back().tokens==8192,
              "512-token chunks stop at configured 8192-token prefix");
        auto extended=cache.keys(prompt,{},true,512,10240);
        check(extended.size()==20 && extended[15].hash==chain.back().hash,
              "changing prefix cap preserves existing chunk keys");
        prompt[9000]=9999;
        check(cache.keys(prompt,{},true,512,8192).back().hash==chain.back().hash,
              "changing a late system field preserves cached early chunks");
        prompt[1500]=9999;
        const auto changed_chain=cache.keys(prompt,{},true,512,8192);
        check(changed_chain[1].hash==chain[1].hash && changed_chain[2].hash!=chain[2].hash &&
              changed_chain.back().hash!=chain.back().hash,
              "changed chunk invalidates itself and descendants, not earlier chunks");
        check(cache.keys(prompt,{},true,256,8192)[1].hash!=chain.front().hash,
              "different configured chunk sizes have separate keys");
        check(cache.keys({1,2}, {}, true,512,8192).empty(), "short prompts do not create partial chunks");
        check(cache.keys(prompt,{},true,512,511).empty(), "cap below one chunk saves nothing");
        check(cache.keys(prompt,{},true,512,9000).back().tokens==8704, "non-aligned cap rounds down to full chunks");
        check(cache.keys(prompt,{},true,0,8192).empty() && cache.keys(prompt,{},true,512,0).empty(),
              "nonpositive chunk size or prefix cap disables keys");
        std::vector<int64_t> exact(513,1);
        check(cache.keys(exact,{},true,512,8192).size()==1,
              "full chunk ending before generation token is eligible");
        exact.pop_back();
        check(cache.keys(exact,{},true,512,8192).empty(), "generation token is never snapshotted");
    }
    {
        ChunkCacheDisk incremental(root,identity,100000,3,131072);
        ChunkCacheDisk reference(root,identity,100000,3,131072);
        auto full_hash = [&](const std::vector<int64_t>& tokens,
                             const std::vector<ConversationImageKey>& images) {
            reference.keys({}, {},true,512,12288);
            return reference.keys(tokens,images,true,512,12288).back().hash;
        };
        std::vector<int64_t> prompt(14052);
        for(size_t i=0;i<prompt.size();++i)prompt[i]=int64_t(i%1000);
        const auto original=incremental.keys(prompt,{},true,512,12288);
        check(original.size()==24 && incremental.hashed_chunks()==24 && incremental.reused_chunks()==0,
              "first OpenCode-sized prefix hashes all 24 chunks");
        prompt.push_back(999);
        const auto continued=incremental.keys(prompt,{},true,512,12288);
        check(incremental.hashed_chunks()==0 && incremental.reused_chunks()==24 &&
              continued.back().hash==original.back().hash, "appended request reuses 24 hashes with zero rehashing");
        prompt[2700]=9999;
        const auto changed=incremental.keys(prompt,{},true,512,12288);
        check(incremental.reused_chunks()==5 && incremental.hashed_chunks()==19,
              "mismatch inside chunk six rehashes only that chunk and descendants");
        check(changed.back().hash==full_hash(prompt,{}),
              "incremental chain matches independent full hash after token change");
        auto pictured=incremental.keys(prompt,{{2700,42}},true,512,12288);
        check(incremental.reused_chunks()==5 && incremental.hashed_chunks()==19 &&
              pictured.back().hash==full_hash(prompt,{{2700,42}}),
              "image insertion invalidates its chunk and descendants");
        auto replaced=incremental.keys(prompt,{{2700,43}},true,512,12288);
        check(incremental.reused_chunks()==5 && incremental.hashed_chunks()==19 &&
              replaced.back().hash==full_hash(prompt,{{2700,43}}),
              "changed image invalidates hashes despite identical token IDs");
        auto removed=incremental.keys(prompt,{},true,512,12288);
        check(incremental.reused_chunks()==5 && incremental.hashed_chunks()==19 &&
              removed.back().hash==changed.back().hash, "image removal restores original token-only keys");
        incremental.keys(prompt,{{13000,99}},true,512,12288);
        check(incremental.hashed_chunks()==0, "image outside disk prefix does not invalidate any chunk");
        incremental.keys(prompt,{},false,512,12288);
        check(incremental.hashed_chunks()==24, "steering change rehashes every chunk");
        incremental.keys(prompt,{},false,256,12288);
        check(incremental.hashed_chunks()==48, "chunk-size change rehashes every chunk");
        incremental.keys(prompt,{},true,512,1024);
        incremental.keys(prompt,{},true,512,2048);
        check(incremental.reused_chunks()==2 && incremental.hashed_chunks()==2,
              "larger prefix cap hashes only additional chunks");
        prompt.resize(1024);
        incremental.keys(prompt,{},true,512,12288);
        check(incremental.reused_chunks()==1 && incremental.hashed_chunks()==0,
              "shortened request reuses complete chunks before generation token");
        prompt.push_back(9);
        const auto growing=incremental.keys(prompt,{},true,512,12288);
        check(incremental.reused_chunks()==1 && incremental.hashed_chunks()==1 &&
              growing.back().hash==full_hash(prompt,{}),
              "short conversation growth hashes only newly completed chunk");
        incremental.keys({}, {},true,512,12288);
        check(incremental.hashed_chunks()==0 && incremental.reused_chunks()==0,
              "empty input clears hash chain safely");
    }
    std::string error;
    auto state = fixture();
    check(cache.store(keys[2].hash,state,error), "two-device state published");
    check(cache.contains(keys[2].hash) && !cache.contains(keys[0].hash),
          "deep complete snapshot works without an earlier chunk file");
    const auto bytes = cache.file_bytes(keys[2].hash);
    {
        ChunkCacheDisk restarted(root,identity,100000,3,131072);
        auto loaded = restarted.load(keys[2].hash,error);
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
        auto streamed = cache.load_stream(keys[2].hash,error);
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
    fs::remove(entry(root,keys[2].hash));
    check(cache.contains(keys[2].hash), "chunk lookup uses memory index without probing each file");
    check(!cache.load_stream(keys[2].hash,error) && !cache.contains(keys[2].hash),
          "selected missing file becomes a miss and leaves the index");
    check(cache.store(keys[2].hash,state,error), "save updates index after missing file");
    {
        std::fstream f(entry(root,keys[2].hash),std::ios::binary|std::ios::in|std::ios::out);
        f.seekp(300);f.put(char(255));
    }
    check(!cache.load(keys[2].hash,error) && !cache.contains(keys[2].hash) && !error.empty(), "corrupt bytes rejected and removed");
    check(cache.store(keys[2].hash,state,error), "repopulate corrupt entry");
    fs::resize_file(entry(root,keys[2].hash),bytes-1);
    check(!cache.load(keys[2].hash,error), "truncated file rejected");
    check(cache.store(keys[2].hash,state,error), "repopulate truncated entry");
    {
        std::fstream f(entry(root,keys[2].hash),std::ios::binary|std::ios::in|std::ios::out);
        f.seekp(160); // first stage begins after header/identity/key/mode/count; then layer range and token count
        for(int i=0;i<8;++i)f.put(char(255));
    }
    check(!cache.load(keys[2].hash,error), "invalid allocation length safely rejected");
    check(cache.store(keys[2].hash,state,error), "repopulate malformed entry");
    ChunkCacheDisk foreign(root,chunk_cache_disk_sha256("model B"),100000,3,131072);
    check(foreign.keys(ids,{{1,45}},false,1,3)[2].hash!=keys[2].hash, "model identity isolates keys");
    check(!foreign.load(keys[2].hash,error), "foreign model payload rejected");
    check(cache.store(keys[2].hash,state,error), "repopulate foreign entry");
    fs::last_write_time(entry(root,keys[2].hash),fs::file_time_type::clock::now()-std::chrono::hours(73));
    check(cache.contains(keys[2].hash), "running session keeps index without per-request expiry checks");
    {
        ChunkCacheDisk restarted(root,identity,100000,3,131072);
        check(!restarted.contains(keys[2].hash) && !fs::exists(entry(root,keys[2].hash)),
              "startup removes expired entry and excludes it from restored index");
    }
    check(cache.store(keys[2].hash,state,error), "repopulate expired entry");
    fs::last_write_time(entry(root,keys[2].hash),fs::file_time_type::clock::now()-std::chrono::hours(49));
    const auto before_touch=fs::last_write_time(entry(root,keys[2].hash));
    cache.touch(keys[2].hash);
    check(fs::last_write_time(entry(root,keys[2].hash))==before_touch,
          "touch records last use in RAM without writing file metadata");
    cache.flush_touches();
    const auto flushed=fs::last_write_time(entry(root,keys[2].hash));
    check(flushed>before_touch, "idle flush persists RAM last use time");
    cache.flush_touches();
    check(fs::last_write_time(entry(root,keys[2].hash))==flushed, "idle flush does not rewrite clean timestamps");
    {
        ChunkCacheDisk restarted(root,identity,100000,3,131072);
        check(restarted.contains(keys[2].hash), "startup restores flushed last use time");
    }
    const auto unindexed=chunk_cache_disk_sha256("unindexed file");
    fs::copy_file(entry(root,keys[2].hash),entry(root,unindexed));
    check(cache.store(keys[0].hash,state,error) && !cache.contains(unindexed),
          "saving updates only its indexed entry without rescanning directory");
    fs::remove(entry(root,unindexed));
    fs::last_write_time(entry(root,keys[2].hash),fs::file_time_type::clock::now()-std::chrono::hours(1));
    check(cache.store(keys[0].hash,state,error), "second entry saved");
    fs::last_write_time(entry(root,keys[0].hash),fs::file_time_type::clock::now()-std::chrono::hours(2));
    std::ofstream(root/"prompt-v1"/"unrelated.txt") << "keep";
    ChunkCacheDisk limited(root,identity,bytes+16,3,131072);
    check(limited.contains(keys[2].hash) && !limited.contains(keys[0].hash), "byte cap evicts least recently used entry");
    check(fs::exists(root/"prompt-v1"/"unrelated.txt"), "cleanup preserves unrelated files");
    const auto abandoned=root/"prompt-v1"/(keys[0].hash+".spc.123.456.tmp");
    std::ofstream(abandoned) << "partial";
    fs::last_write_time(abandoned,fs::file_time_type::clock::now()-std::chrono::hours(73));
    {
        ChunkCacheDisk restarted(root,identity,bytes+16,3,131072);
        check(!fs::exists(abandoned), "startup removes expired interrupted write");
    }
    const auto eviction_root=root/"eviction";
    ChunkCacheDisk eviction(eviction_root,identity,bytes*2+16,3,131072);
    check(eviction.store(keys[0].hash,state,error) && eviction.store(keys[1].hash,state,error),
          "two snapshots fit incremental byte budget");
    eviction.touch(keys[0].hash);
    check(eviction.store(keys[2].hash,state,error) && eviction.contains(keys[0].hash) &&
          eviction.contains(keys[2].hash) && !eviction.contains(keys[1].hash) &&
          !fs::exists(entry(eviction_root,keys[1].hash)),
          "save enforces byte cap and honors RAM last use before timestamp flush");
    fs::remove(entry(eviction_root,keys[2].hash));
    eviction.touch(keys[2].hash);
    check(eviction.contains(keys[2].hash), "touch does not probe missing file on request path");
    eviction.flush_touches();
    check(!eviction.contains(keys[2].hash), "idle flush removes missing entry from index");
    {
        ChunkCacheDisk chain(root / "chain", identity, 100000, 3, 131072);
        auto parent = stream_image(state, 0);
        check(chain.store_stream(keys[0].hash, parent, "", error), "chained root saved");
        const auto root_bytes = chain.file_bytes(keys[0].hash);
        auto child = stream_image(fixture_child(), 257);
        check(chain.store_stream(keys[1].hash, child, keys[0].hash, error), "chained child saved on parent");
        const auto child_bytes = chain.file_bytes(keys[1].hash);
        check(child_bytes > root_bytes && child_bytes < root_bytes + 2 * 64 + 4096,
              "chained child stores only appended bytes, not a second full snapshot");
        auto loaded = chain.load_stream(keys[1].hash, error);
        check(bool(loaded) && loaded->stages.size() == 2 && !loaded->cvec, "chained child loads");
        auto want = fixture_child();
        check(read_all(*loaded, loaded->stages[0].kv[0].data[0]) == to_vec(want.stages[0].kv[0].k) &&
              read_all(*loaded, loaded->stages[1].kv[0].data[1]) == to_vec(want.stages[1].kv[0].v) &&
              read_all(*loaded, loaded->draft.data[0]) == to_vec(want.draft.k),
              "chained child restores parent prefix and appended bytes");
        check(read_all(*loaded, loaded->stages[0].running[0]) == state.stages[0].running.gdn,
              "chained child restores running state");
        {
            std::fstream f(entry(root / "chain", keys[1].hash), std::ios::binary | std::ios::in | std::ios::out);
            f.seekp(10); f.put(char(255));
        }
        check(!chain.load_stream(keys[1].hash, error) && !chain.contains(keys[1].hash),
              "corrupt chained child rejected and removed");
        check(bool(chain.load_stream(keys[0].hash, error)), "chained root survives child removal");
        check(chain.store_stream(keys[1].hash, child, keys[0].hash, error), "repopulate chained child");
        fs::remove(entry(root / "chain", keys[0].hash));
        auto orphan = chain.load_stream(keys[1].hash, error);
        check(bool(orphan) && read_all(*orphan, orphan->stages[0].kv[0].data[0]) == to_vec(want.stages[0].kv[0].k),
              "chained child loads after parent eviction: file is self-contained");
        auto stranded = stream_image(fixture_child(), 257);
        check(!chain.store_stream(keys[2].hash, stranded, "", error) && !error.empty(),
              "reuse without parent fails instead of storing wrong bytes");
        ChunkCacheDiskWriter writer(chain);
        auto queued = stream_image(state, 0);
        check(writer.queue(keys[2].hash, keys[1].hash, std::move(queued), error), "writer accepts save");
        writer.drain();
        check(chain.contains(keys[2].hash) && bool(chain.load_stream(keys[2].hash, error)),
              "background writer publishes readable entry");
    }
    ChunkCacheDisk too_small(root/"small",identity,500,3,131072);
    check(!too_small.store(keys[2].hash,state,error), "oversized entry skipped");
    ChunkCacheDisk disabled(root/"off",identity,0,3,131072);
    check(!disabled.enabled() && !fs::exists(root/"off"), "disabled cache does not create files");
    std::ofstream(root/"artifact") << "first";
    const auto a=chunk_cache_disk_identity({"kv=int8"},{root/"artifact"});
    std::ofstream(root/"artifact") << "other";
    const auto b=chunk_cache_disk_identity({"kv=int8"},{root/"artifact"});
    check(!a.empty() && a!=b, "artifact changes invalidate identity");
    check(b!=chunk_cache_disk_identity({"kv=q4"},{root/"artifact"}), "runtime settings invalidate identity");
    check(chunk_cache_disk_identity({}, {root/"missing"}).empty(), "missing artifacts disable persistence");
    fs::remove_all(root);
    std::printf("chunk_cache_disk_test: %d checks passed\n",checks);
}
