#include "strata/core/prompt_disk_snapshot.hpp"
#include "strata/core/on_device.hpp"
#include "conversation_checked.hpp"

namespace strata::core {
namespace {
bool fail(std::string& error, const char* message) { error = message; return false; }
bool targets_valid(const std::vector<DiskPromptTarget>& targets, const ModelGeometry& g, std::string& error) {
    int64_t at = 0;
    if (targets.empty() || targets.size() > 16) return fail(error, "invalid disk snapshot stage count");
    for (const auto& t : targets) {
        if (!t.session || t.device < 0 || t.session->layer_lo != at || t.session->layer_hi <= at)
            return fail(error, "invalid disk snapshot stage range");
        at = t.session->layer_hi;
    }
    return at == g.n_layers || fail(error, "incomplete disk snapshot layer ranges");
}
bool sync(std::string& error) {
    const auto status = cudaDeviceSynchronize();
    return status == cudaSuccess || fail(error, cudaGetErrorString(status));
}
}
namespace {
DiskPromptBlob source_blob(int device, const void* src, size_t bytes, std::string& error) {
    return {0, bytes, [device, src, &error](void* dst, size_t n, size_t at) {
        const OnDevice on(device);
        const auto e = cudaMemcpy(dst, static_cast<const uint8_t*>(src)+at, n, cudaMemcpyDefault);
        return e == cudaSuccess || fail(error, cudaGetErrorString(e));
    }};
}
std::array<void*,5> running_pointers(const SessionState& s, size_t j) {
    const auto* q = s.qsa_alloc ? &s.qsa_states[s.qsa_ord0+j] : nullptr;
    return {s.gdn_state, s.ple_hist, q ? q->idx_tail : nullptr, q ? q->idx_dead : nullptr, q ? q->idx_block_pos : nullptr};
}
bool kv_metadata(const DiskPromptKv& k, const ConversationKvLayout& l) {
    if(k.format!=l.format || k.cells!=l.cells || k.heads!=l.heads || k.head_dim!=l.head_dim ||
       k.page_size!=l.page_size || k.pooled_rows!=l.pooled_rows || k.idx_dim!=l.idx_dim)return false;
    for(size_t i=0;i<5;++i)if(k.data[i].size!=l.sizes[i])return false;
    return true;
}
bool image_metadata(const std::vector<ConversationImageKey>& images, size_t tokens) {
    int64_t previous = -1;
    for(const auto& i:images) {
        if(i.start<=previous || i.start<0 || uint64_t(i.start)>=tokens)return false;
        previous=i.start;
    }
    return true;
}
}
bool prompt_disk_stream_source(DiskPromptFile& image, const std::vector<DiskPromptTarget>& targets,
                               const ModelGeometry& g, const QsaState& draft,
                               const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                               bool cvec, std::string& error) {
    size_t estimate=0;
    if(!prompt_disk_snapshot_bytes(targets,g,draft,int64_t(ids.size()),images.size(),estimate,error))return false;
    DiskPromptFile result;result.cvec=cvec;result.stages.resize(targets.size());
    auto describe=[&](DiskPromptKv& out,const QsaState& st,bool index,int device) {
        ConversationKvLayout l;
        if(!conversation_kv_layout(st,g,int64_t(ids.size()),index,l,error))return false;
        out.format=l.format;out.cells=l.cells;out.heads=l.heads;out.head_dim=l.head_dim;
        out.page_size=l.page_size;out.pooled_rows=l.pooled_rows;out.idx_dim=l.idx_dim;
        for(size_t j=0;j<5;++j)out.data[j]=source_blob(device,l.pools[j],l.sizes[j],error);
        return true;
    };
    for(size_t i=0;i<targets.size();++i) {
        const OnDevice on(targets[i].device);
        const auto& s=*targets[i].session;
        if(!sync(error))return false;
        auto& out=result.stages[i];out.layer_lo=s.layer_lo;out.layer_hi=s.layer_hi;out.ids=ids;out.images=images;
        ConversationStateSizes z;
        if(!conversation_session_sizes(g,s,z,error))return false;
        const auto pointers=running_pointers(s,0);
        out.running[0]=source_blob(targets[i].device,pointers[0],z.gdn,error);
        out.running[1]=source_blob(targets[i].device,pointers[1],s.ple_hist?z.ple:0,error);
        // Indexer tails are not contiguous between layers. Their file payloads
        // concatenate each owned layer's row, matching the ordinary snapshot codec.
        for(size_t kind=2;kind<5;++kind) {
            const size_t per=kind==2?z.tail:kind==3?z.dead:z.block_pos;
            out.running[kind]={0,size_t(s.qsa_alloc)*per,[&,target=targets[i],kind,per](void* dst,size_t n,size_t at) {
                const OnDevice active(target.device);
                while(n) {
                    const size_t j=at/per,off=at%per,count=std::min(n,per-off);
                    const auto ptr=running_pointers(*target.session,j)[kind];
                    const auto e=cudaMemcpy(dst,static_cast<const uint8_t*>(ptr)+off,count,cudaMemcpyDefault);
                    if(e!=cudaSuccess)return fail(error,cudaGetErrorString(e));
                    dst=static_cast<uint8_t*>(dst)+count;at+=count;n-=count;
                }
                return true;
            }};
        }
        out.kv.resize(size_t(s.qsa_alloc));
        for(size_t j=0;j<out.kv.size();++j)
            if(!describe(out.kv[j],s.qsa_states[s.qsa_ord0+j],true,targets[i].device))return false;
    }
    if(!describe(result.draft,draft,false,targets.back().device))return false;
    image=std::move(result);return true;
}
bool prompt_disk_stream_validate(const DiskPromptFile& image, const std::vector<DiskPromptTarget>& targets,
                                 const ModelGeometry& g, const QsaState& draft, std::string& error) {
    if(!targets_valid(targets,g,error) || image.stages.size()!=targets.size())return false;
    const auto& first=image.stages.front();
    if(first.ids.empty() || !image_metadata(first.images,first.ids.size()) ||
       std::any_of(first.ids.begin(),first.ids.end(),[](auto id){return id<0;}))return fail(error,"invalid streaming prompt metadata");
    const auto L=int64_t(first.ids.size());
    for(size_t i=0;i<targets.size();++i) {
        const auto& s=*targets[i].session;const auto& part=image.stages[i];ConversationStateSizes z;
        if(part.layer_lo!=s.layer_lo || part.layer_hi!=s.layer_hi || part.ids!=first.ids || part.images!=first.images ||
           L>s.max_cells || part.kv.size()!=size_t(s.qsa_alloc) || !conversation_session_sizes(g,s,z,error))return false;
        const std::array<size_t,5> sizes{z.gdn,s.ple_hist?z.ple:0,size_t(s.qsa_alloc)*z.tail,
                                       size_t(s.qsa_alloc)*z.dead,size_t(s.qsa_alloc)*z.block_pos};
        for(size_t j=0;j<5;++j) {
            if(part.running[j].size!=sizes[j])return fail(error,"invalid streaming running-state size");
            if(j<2 && sizes[j] && !running_pointers(s,0)[j])return false;
        }
        for(size_t j=0;j<part.kv.size();++j) {
            const auto& st=s.qsa_states[s.qsa_ord0+j];ConversationKvLayout l;
            if(!st.idx_tail || !st.idx_dead || !st.idx_block_pos ||
               !conversation_kv_layout(st,g,L,true,l,error) || !kv_metadata(part.kv[j],l))return false;
        }
    }
    ConversationKvLayout l;
    return conversation_kv_layout(draft,g,L,false,l,error) && kv_metadata(image.draft,l);
}
ConversationRestore prompt_disk_stream_restore(const DiskPromptFile& image,
                                               const std::vector<DiskPromptTarget>& targets,
                                               const ModelGeometry& g, const QsaState& draft, std::string& error) {
    if(!image.file || !prompt_disk_stream_validate(image,targets,g,draft,error))return ConversationRestore::invalid;
    std::array<uint8_t,65536> scratch;
    auto transfer=[&](const DiskPromptBlob& blob,void* dst,size_t off,size_t bytes) {
        for(size_t at=0;at<bytes;) {
            const size_t n=std::min(scratch.size(),bytes-at);
            if(!image.read(blob,scratch.data(),off+at,n))return fail(error,"streaming snapshot read failed");
            const auto e=cudaMemcpy(static_cast<uint8_t*>(dst)+at,scratch.data(),n,cudaMemcpyDefault);
            if(e!=cudaSuccess)return fail(error,cudaGetErrorString(e));
            at+=n;
        }
        return true;
    };
    const auto L=int64_t(image.stages[0].ids.size());
    auto kv_restore=[&](const DiskPromptKv& k,const QsaState& st,bool index) {
        ConversationKvLayout l;
        if(!conversation_kv_layout(st,g,L,index,l,error))return false;
        for(size_t j=0;j<5;++j)if(!transfer(k.data[j],l.pools[j],0,l.sizes[j]))return false;
        return conversation_kv_restore_finish(st,g,L,error);
    };
    for(size_t i=0;i<targets.size();++i) {
        const OnDevice on(targets[i].device);auto& s=*targets[i].session;
        if(!sync(error))return ConversationRestore::transfer_failed;
        ConversationStateSizes z;
        if(!conversation_session_sizes(g,s,z,error))return ConversationRestore::transfer_failed;
        const auto ptr=running_pointers(s,0);
        if(!transfer(image.stages[i].running[0],ptr[0],0,z.gdn) ||
           !transfer(image.stages[i].running[1],ptr[1],0,s.ple_hist?z.ple:0))return ConversationRestore::transfer_failed;
        for(size_t j=0;j<size_t(s.qsa_alloc);++j) {
            const auto p=running_pointers(s,j);
            for(size_t kind=2;kind<5;++kind) {
                const auto per=kind==2?z.tail:kind==3?z.dead:z.block_pos;
                if(!transfer(image.stages[i].running[kind],p[kind],j*per,per))return ConversationRestore::transfer_failed;
            }
            const auto& st=s.qsa_states[s.qsa_ord0+j];
            if(!kv_restore(image.stages[i].kv[j],st,true))return ConversationRestore::transfer_failed;
            const auto e=cudaMemcpy(st.idx_pooled+(L/strata::kernels::qsa_real_shapes().idx_block)*g.idx_key_dim,
                                    st.idx_dead,z.dead,cudaMemcpyDefault);
            if(e!=cudaSuccess){fail(error,cudaGetErrorString(e));return ConversationRestore::transfer_failed;}
        }
        s.ple_prev[0]=L>=2?image.stages[i].ids[size_t(L-2)]:-1;
        s.ple_prev[1]=image.stages[i].ids[size_t(L-1)];
        if(!sync(error))return ConversationRestore::transfer_failed;
    }
    const OnDevice on(targets.back().device);
    if(!kv_restore(image.draft,draft,false) || !sync(error))return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}
bool prompt_disk_snapshot_bytes(const std::vector<DiskPromptTarget>& targets, const ModelGeometry& g,
                                const QsaState& draft, int64_t tokens, size_t images,
                                size_t& bytes, std::string& error) {
    bytes = 0;
    if (!targets_valid(targets, g, error) || tokens <= 0) return false;
    for (const auto& t : targets) {
        const auto& s = *t.session;
        ConversationStateSizes z;
        if (tokens > s.max_cells || !conversation_session_sizes(g, s, z, error)) return false;
        size_t ids = 0, imgs = 0, qsa = 0;
        if (!conversation_detail::product(ids, {uint64_t(tokens), sizeof(int32_t)}) ||
            !conversation_detail::product(imgs, {images, sizeof(ConversationImageKey)}) ||
            !conversation_detail::product(qsa, {uint64_t(s.qsa_alloc), z.tail + z.dead + z.block_pos + sizeof(ConversationKv)}))
            return fail(error, "disk snapshot size overflow");
        for (const auto n : {ids, imgs, qsa, z.gdn, s.ple_hist ? z.ple : 0, sizeof(DiskPromptStage)})
            if (!conversation_detail::add(bytes, n)) return fail(error, "disk snapshot size overflow");
        for (int64_t j = 0; j < s.qsa_alloc; ++j) {
            const auto n = conversation_kv_bytes(s.qsa_states[s.qsa_ord0+j], g, tokens, true);
            if (!n || !conversation_detail::add(bytes, n)) return fail(error, "invalid disk snapshot K/V extent");
        }
    }
    const auto n = conversation_kv_bytes(draft, g, tokens, false);
    return (n && conversation_detail::add(bytes, n)) || fail(error, "invalid disk snapshot draft extent");
}
bool prompt_disk_snapshot_save(DiskPromptState& image, const std::vector<DiskPromptTarget>& targets,
                               const ModelGeometry& g, const QsaState& draft,
                               const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                               bool cvec, std::string& error) {
    size_t bytes = 0;
    if (!prompt_disk_snapshot_bytes(targets, g, draft, int64_t(ids.size()), images.size(), bytes, error)) return false;
    DiskPromptState result;
    result.cvec = cvec; result.stages.resize(targets.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        const OnDevice on(targets[i].device);
        const auto& s = *targets[i].session;
        auto& part = result.stages[i];
        part.layer_lo = s.layer_lo; part.layer_hi = s.layer_hi;
        part.running.ids = ids; part.running.imgs = images;
        if (!sync(error) || !conversation_checkpoint_save(part.running, s, g, error)) return false;
        part.kv.resize(size_t(s.qsa_alloc));
        for (size_t j = 0; j < part.kv.size(); ++j)
            if (!conversation_kv_save(part.kv[j], s.qsa_states[s.qsa_ord0+j], g, int64_t(ids.size()), true, error)) return false;
    }
    {
        const OnDevice on(targets.back().device);
        if (!conversation_kv_save(result.draft, draft, g, int64_t(ids.size()), false, error)) return false;
    }
    image = std::move(result);
    return true;
}
bool prompt_disk_snapshot_validate(const DiskPromptState& image, const std::vector<DiskPromptTarget>& targets,
                                   const ModelGeometry& g, const QsaState& draft, std::string& error) {
    if (!targets_valid(targets, g, error) || image.stages.size() != targets.size()) return false;
    const auto& first = image.stages.front().running;
    if (first.ids.empty() || std::any_of(first.ids.begin(), first.ids.end(), [](auto id) { return id < 0; }))
        return fail(error, "invalid disk snapshot token IDs");
    for (size_t i = 0; i < targets.size(); ++i) {
        const auto& s = *targets[i].session;
        const auto& part = image.stages[i];
        if (part.layer_lo != s.layer_lo || part.layer_hi != s.layer_hi || !part.running.stage_parts.empty() ||
            part.running.ids != first.ids || part.running.imgs != first.imgs || part.kv.size() != size_t(s.qsa_alloc) ||
            !conversation_checkpoint_validate(part.running, s, g, error)) return fail(error, "invalid disk snapshot running state");
        for (size_t j = 0; j < part.kv.size(); ++j)
            if (!conversation_kv_validate(part.kv[j], s.qsa_states[s.qsa_ord0+j], g, int64_t(first.ids.size()), true, error)) return false;
    }
    return conversation_kv_validate(image.draft, draft, g, int64_t(first.ids.size()), false, error);
}
ConversationRestore prompt_disk_snapshot_restore(const DiskPromptState& image,
                                                 const std::vector<DiskPromptTarget>& targets,
                                                 const ModelGeometry& g, const QsaState& draft, std::string& error) {
    // Validate every device before applying anything to the first one.
    if (!prompt_disk_snapshot_validate(image, targets, g, draft, error)) return ConversationRestore::invalid;
    const auto tokens = int64_t(image.stages[0].running.ids.size());
    for (size_t i = 0; i < targets.size(); ++i) {
        const OnDevice on(targets[i].device);
        auto& s = *targets[i].session;
        if (!sync(error)) return ConversationRestore::transfer_failed;
        for (size_t j = 0; j < image.stages[i].kv.size(); ++j)
            if (!conversation_kv_restore(image.stages[i].kv[j], s.qsa_states[s.qsa_ord0+j], g, tokens, true, error))
                return ConversationRestore::transfer_failed;
        if (!conversation_checkpoint_restore(image.stages[i].running, s, g, error)) return ConversationRestore::transfer_failed;
    }
    const OnDevice on(targets.back().device);
    if (!conversation_kv_restore(image.draft, draft, g, tokens, false, error) || !sync(error)) return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}
} // namespace strata::core
