#pragma once

#include "strata/core/conversation_cache.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace strata::core {

// Each device owns its running state and positional K/V; the last device owns
// the MTP K/V too. A disk entry is a complete cumulative prompt prefix.
struct DiskChunkStage {
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint running;
    std::vector<ConversationKv> kv;
};
struct DiskChunkState {
    std::vector<DiskChunkStage> stages;
    ConversationKv draft;
    bool cvec = true;
    size_t bytes() const;
};

struct DiskChunkKey { int64_t tokens = 0; std::string hash; };

// Streaming codec keeps payloads on disk. A chained entry stores only the bytes
// a chunk adds to its parent's payload; the rest are file spans of earlier chunks.
// Capture callbacks fill at most 64 KiB; restore spans reference an open file.
struct DiskChunkSpan { uint64_t offset = 0, size = 0; };
struct DiskChunkBlob {
    uint64_t offset = 0, size = 0;
    std::function<bool(void*, size_t, size_t)> read;
    // Capture: bytes [0, reuse) are identical to the parent chunk's payload and
    // are not transferred again; read() still reports logical offsets [0, size).
    uint64_t reuse = 0;
    // Load: file locations of the logical bytes [0, size); empty means one span
    // at offset (the whole payload contiguous).
    std::vector<DiskChunkSpan> spans;
};
struct DiskChunkKv {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::array<DiskChunkBlob, 5> data;
};
struct DiskChunkFileStage {
    int64_t layer_lo = 0, layer_hi = 0;
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> images;
    std::array<DiskChunkBlob, 5> running;
    std::vector<DiskChunkKv> kv;
};
struct DiskChunkFile {
    bool cvec = true;
    std::vector<DiskChunkFileStage> stages;
    DiskChunkKv draft;
    std::shared_ptr<std::ifstream> file;
    size_t bytes() const;
    bool read(const DiskChunkBlob& blob, void* target, size_t offset, size_t count) const;
};

// CPU-only codec and retention policy. Failed reads/writes are cache misses.
// Files are versioned, checksummed, bounded before allocation and published by
// rename. Only this cache's 64-hex .spc files are candidates for removal.
class ChunkCacheDisk {
public:
    ChunkCacheDisk(std::filesystem::path root, std::string identity,
                    uint64_t budget, int days, int64_t max_tokens);
    bool enabled() const { return !root_.empty() && budget_ != 0; }
    const std::vector<DiskChunkKey>& keys(const std::vector<int64_t>& ids,
                                   const std::vector<ConversationImageKey>& images,
                                   bool cvec, int64_t chunk_tokens, int64_t prefix_tokens);
    size_t hashed_chunks() const { return hashed_chunks_; }
    size_t reused_chunks() const { return keys_.size() - hashed_chunks_; }
    bool contains(const std::string& key) const;
    uint64_t file_bytes(const std::string& key) const;
    std::optional<DiskChunkState> load(const std::string& key, std::string& error);
    bool store(const std::string& key, const DiskChunkState& state, std::string& error);
    std::optional<DiskChunkFile> load_stream(const std::string& key, std::string& error);
    // `parent` (a stored key or empty) selects the chained format: the entry
    // stores only bytes its blobs add beyond the parent's payload. Blobs with
    // reuse > 0 require a parent; without one the save fails.
    bool store_stream(const std::string& key, const DiskChunkFile& state,
                      const std::string& parent, std::string& error);
    void touch(const std::string& key);
    // Called only while idle or at orderly shutdown; touch itself does no I/O.
    void flush_touches();
    void discard(const std::string& key);

private:
    void startup_cleanup();
    std::filesystem::path path(const std::string& key) const;
    std::filesystem::path root_;
    std::string identity_;
    uint64_t budget_;
    int days_;
    int64_t max_tokens_;
    void published(const std::string& key, uint64_t bytes);
    // Callers hold index_mu_; public methods lock it themselves.
    void forget_locked(const std::string& key);
    struct Entry {
        std::filesystem::file_time_type used;
        uint64_t bytes;
        bool dirty = false;
    };
    // Startup scans once. Saves/discards update the index and byte total directly.
    // The background writer publishes while request threads look up, so the
    // index and byte total sit behind this mutex.
    mutable std::mutex index_mu_;
    std::unordered_map<std::string, Entry> cached_;
    uint64_t cached_bytes_ = 0;
    // Hash identity belongs to this exact token/image prefix, independently of
    // whether its inference state still lives in VRAM (or a request was cancelled).
    std::vector<int64_t> key_ids_;
    std::vector<ConversationImageKey> key_images_;
    std::vector<DiskChunkKey> keys_;
    int64_t key_chunk_tokens_ = 0;
    bool key_cvec_ = true;
    size_t hashed_chunks_ = 0;
};

// Engine/build/options identity plus canonical artifact paths, sizes and mtimes.
// Small artifact files are content hashed too. Failure disables persistence.
std::string chunk_cache_disk_identity(const std::vector<std::string>& options,
                                 const std::vector<std::filesystem::path>& artifacts);
std::string chunk_cache_disk_sha256(const std::string& value);

// Replace capture callbacks with host-memory copies of the bytes a save adds,
// so the request thread finishes GPU transfers before generation moves on.
bool chunk_cache_disk_stage(DiskChunkFile& image, std::string& error);

// Background writer: request threads stage a capture and queue it; this thread
// writes files in queue order, so a chained child follows its parent. Saves are
// best effort: a full queue skips the save, and errors only reach the log.
class ChunkCacheDiskWriter {
public:
    explicit ChunkCacheDiskWriter(ChunkCacheDisk& cache);
    ~ChunkCacheDiskWriter();
    bool queue(std::string key, std::string parent, DiskChunkFile image, std::string& error);
    bool pending(const std::string& key) const;
    void drain();

private:
    void run();
    struct Item { std::string key, parent; DiskChunkFile image; };
    ChunkCacheDisk& cache_;
    std::thread thread_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Item> items_;
    std::unordered_set<std::string> keys_;
    bool stop_ = false, busy_ = false;
};

} // namespace strata::core
