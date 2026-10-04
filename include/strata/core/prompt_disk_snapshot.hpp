#pragma once
#include "strata/core/prompt_disk_cache.hpp"
#include "strata/core/conversation_snapshot.hpp"

namespace strata::core {
struct DiskPromptTarget { int device; SessionState* session; };
bool prompt_disk_stream_source(DiskPromptFile& image, const std::vector<DiskPromptTarget>& targets,
                               const ModelGeometry& g, const QsaState& draft,
                               const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                               bool cvec, std::string& error);
bool prompt_disk_stream_validate(const DiskPromptFile& image, const std::vector<DiskPromptTarget>& targets,
                                 const ModelGeometry& g, const QsaState& draft, std::string& error);
ConversationRestore prompt_disk_stream_restore(const DiskPromptFile& image,
                                               const std::vector<DiskPromptTarget>& targets,
                                               const ModelGeometry& g, const QsaState& draft, std::string& error);
// All stages must be at the same consumed prefix. These functions preserve the
// caller's current device; the final target's device owns the MTP state.
bool prompt_disk_snapshot_bytes(const std::vector<DiskPromptTarget>& targets, const ModelGeometry& g,
                                const QsaState& draft, int64_t tokens, size_t images,
                                size_t& bytes, std::string& error);
bool prompt_disk_snapshot_save(DiskPromptState& image, const std::vector<DiskPromptTarget>& targets,
                               const ModelGeometry& g, const QsaState& draft,
                               const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                               bool cvec, std::string& error);
bool prompt_disk_snapshot_validate(const DiskPromptState& image, const std::vector<DiskPromptTarget>& targets,
                                   const ModelGeometry& g, const QsaState& draft, std::string& error);
ConversationRestore prompt_disk_snapshot_restore(const DiskPromptState& image,
                                                 const std::vector<DiskPromptTarget>& targets,
                                                 const ModelGeometry& g, const QsaState& draft, std::string& error);
} // namespace strata::core
