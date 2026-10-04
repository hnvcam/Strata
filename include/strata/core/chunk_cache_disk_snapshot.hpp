#pragma once
#include "strata/core/chunk_cache_disk.hpp"
#include "strata/core/conversation_snapshot.hpp"

namespace strata::core {
struct DiskChunkTarget { int device; SessionState* session; };
bool chunk_cache_disk_stream_source(DiskChunkFile& image, const std::vector<DiskChunkTarget>& targets,
                               const ModelGeometry& g, const QsaState& draft,
                               const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                               bool cvec, std::string& error);
bool chunk_cache_disk_stream_validate(const DiskChunkFile& image, const std::vector<DiskChunkTarget>& targets,
                                 const ModelGeometry& g, const QsaState& draft, std::string& error);
ConversationRestore chunk_cache_disk_stream_restore(const DiskChunkFile& image,
                                               const std::vector<DiskChunkTarget>& targets,
                                               const ModelGeometry& g, const QsaState& draft, std::string& error);
// All stages must be at the same consumed prefix. These functions preserve the
// caller's current device; the final target's device owns the MTP state.
bool chunk_cache_disk_snapshot_bytes(const std::vector<DiskChunkTarget>& targets, const ModelGeometry& g,
                                const QsaState& draft, int64_t tokens, size_t images,
                                size_t& bytes, std::string& error);
bool chunk_cache_disk_snapshot_save(DiskChunkState& image, const std::vector<DiskChunkTarget>& targets,
                               const ModelGeometry& g, const QsaState& draft,
                               const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                               bool cvec, std::string& error);
bool chunk_cache_disk_snapshot_validate(const DiskChunkState& image, const std::vector<DiskChunkTarget>& targets,
                                   const ModelGeometry& g, const QsaState& draft, std::string& error);
ConversationRestore chunk_cache_disk_snapshot_restore(const DiskChunkState& image,
                                                 const std::vector<DiskChunkTarget>& targets,
                                                 const ModelGeometry& g, const QsaState& draft, std::string& error);
} // namespace strata::core
