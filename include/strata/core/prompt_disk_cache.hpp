#pragma once

#include "strata/core/conversation_cache.hpp"

#include <filesystem>
#include <functional>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

namespace strata::core {

// Each device owns its running state and positional K/V; the last device owns
// the MTP K/V too. A disk entry is a complete cumulative prompt prefix.
struct DiskPromptStage {
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint running;
    std::vector<ConversationKv> kv;
};
struct DiskPromptState {
    std::vector<DiskPromptStage> stages;
    ConversationKv draft;
    bool cvec = true;
    size_t bytes() const;
};

struct DiskPromptKey { int64_t tokens = 0; std::string hash; };

// Streaming codec uses the same file schema, with payloads kept on disk.
// Capture callbacks fill at most 64 KiB; restore spans reference an open file.
struct DiskPromptBlob {
    uint64_t offset = 0, size = 0;
    std::function<bool(void*, size_t, size_t)> read;
};
struct DiskPromptKv {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::array<DiskPromptBlob, 5> data;
};
struct DiskPromptFileStage {
    int64_t layer_lo = 0, layer_hi = 0;
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> images;
    std::array<DiskPromptBlob, 5> running;
    std::vector<DiskPromptKv> kv;
};
struct DiskPromptFile {
    bool cvec = true;
    std::vector<DiskPromptFileStage> stages;
    DiskPromptKv draft;
    std::shared_ptr<std::ifstream> file;
    size_t bytes() const;
    bool read(const DiskPromptBlob& blob, void* target, size_t offset, size_t count) const;
};

// CPU-only codec and retention policy. Failed reads/writes are cache misses.
// Files are versioned, checksummed, bounded before allocation and published by
// rename. Only this cache's 64-hex .spc files are candidates for removal.
class PromptDiskCache {
public:
    PromptDiskCache(std::filesystem::path root, std::string identity,
                    uint64_t budget, int days, int64_t max_tokens);
    bool enabled() const { return !root_.empty() && budget_ != 0; }
    std::vector<DiskPromptKey> keys(const std::vector<int64_t>& ids,
                                   const std::vector<ConversationImageKey>& images,
                                   bool cvec, const std::vector<int64_t>& boundaries) const;
    bool contains(const std::string& key) const;
    uint64_t file_bytes(const std::string& key) const;
    std::optional<DiskPromptState> load(const std::string& key, std::string& error);
    bool store(const std::string& key, const DiskPromptState& state, std::string& error);
    std::optional<DiskPromptFile> load_stream(const std::string& key, std::string& error);
    bool store_stream(const std::string& key, const DiskPromptFile& state, std::string& error);
    void touch(const std::string& key);
    void discard(const std::string& key);
    void prune();

private:
    std::filesystem::path path(const std::string& key) const;
    std::filesystem::path root_;
    std::string identity_;
    uint64_t budget_;
    int days_;
    int64_t max_tokens_;
};

// Engine/build/options identity plus canonical artifact paths, sizes and mtimes.
// Small artifact files are content hashed too. Failure disables persistence.
std::string prompt_disk_identity(const std::vector<std::string>& options,
                                 const std::vector<std::filesystem::path>& artifacts);
std::string prompt_disk_sha256(const std::string& value);

} // namespace strata::core
