#include "strata/core/chunk_cache_disk.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#if defined(__linux__)
#include <fcntl.h>
#include <linux/fs.h>
#include <sys/ioctl.h>
#endif
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace strata::core {
namespace {
// Streaming SHA-256; keys and payload integrity use the same portable encoding.
class Sha256 {
    std::array<uint32_t, 8> h_{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                              0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::array<uint8_t, 64> block_{};
    uint64_t bytes_ = 0;
    size_t used_ = 0;
    static uint32_t rot(uint32_t x, int n) { return (x >> n) | (x << (32-n)); }
    void compress() {
        static constexpr uint32_t k[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t w[64];
        for (size_t i=0;i<16;++i) w[i]=(uint32_t(block_[4*i])<<24)|(uint32_t(block_[4*i+1])<<16)|
                                      (uint32_t(block_[4*i+2])<<8)|block_[4*i+3];
        for (size_t i=16;i<64;++i) {
            const auto a=w[i-15],b=w[i-2];
            w[i]=w[i-16]+(rot(a,7)^rot(a,18)^(a>>3))+w[i-7]+(rot(b,17)^rot(b,19)^(b>>10));
        }
        auto a=h_[0],b=h_[1],c=h_[2],d=h_[3],e=h_[4],f=h_[5],g=h_[6],h=h_[7];
        for (size_t i=0;i<64;++i) {
            const auto t=h+(rot(e,6)^rot(e,11)^rot(e,25))+((e&f)^(~e&g))+k[i]+w[i];
            const auto u=(rot(a,2)^rot(a,13)^rot(a,22))+((a&b)^(a&c)^(b&c));
            h=g;g=f;f=e;e=d+t;d=c;c=b;b=a;a=t+u;
        }
        h_[0]+=a;h_[1]+=b;h_[2]+=c;h_[3]+=d;h_[4]+=e;h_[5]+=f;h_[6]+=g;h_[7]+=h;
    }
public:
    void update(const void* data, size_t count) {
        const auto* p=static_cast<const uint8_t*>(data);
        bytes_+=count;
        while(count) {
            const size_t n=std::min(count,64-used_);
            std::memcpy(block_.data()+used_,p,n);used_+=n;p+=n;count-=n;
            if(used_==64) {compress();used_=0;}
        }
    }
    void number(uint64_t v) {
        uint8_t b[8];for(int i=0;i<8;++i) b[i]=uint8_t(v>>(i*8));update(b,8);
    }
    void text(const std::string& s) { number(s.size());update(s.data(),s.size()); }
    // A chained save continues its parent's payload digest without re-reading
    // the shared prefix: 112 bytes of mid-stream state travel in the footer.
    void store(uint8_t* out) const {
        for(int i=0;i<8;++i) for(int b=0;b<4;++b) out[4*i+b]=uint8_t(h_[i]>>(8*b));
        std::memcpy(out+32,block_.data(),64);
        for(int b=0;b<8;++b){out[96+b]=uint8_t(bytes_>>(8*b));out[104+b]=uint8_t(used_>>(8*b));}
    }
    static Sha256 restore(const uint8_t* in) {
        Sha256 s;
        for(int i=0;i<8;++i){uint32_t v=0;for(int b=0;b<4;++b)v|=uint32_t(in[4*i+b])<<(8*b);s.h_[i]=v;}
        std::memcpy(s.block_.data(),in+32,64);
        for(int b=0;b<8;++b){s.bytes_|=uint64_t(in[96+b])<<(8*b);s.used_|=uint64_t(in[104+b])<<(8*b);}
        return s;
    }
    uint64_t used() const { return used_; }
    uint64_t bytes() const { return bytes_; }
    std::string finish() const {
        auto copy=*this;
        const uint64_t bits=bytes_*8;
        uint8_t pad[128]{};pad[0]=0x80;
        copy.update(pad,used_<56 ? 56-used_ : 120-used_);
        for(int i=0;i<8;++i) pad[i]=uint8_t(bits>>((7-i)*8));
        copy.update(pad,8);
        std::string out;
        static constexpr char hex[]="0123456789abcdef";
        for(auto v:copy.h_) for(int shift=28;shift>=0;shift-=4) out+=hex[(v>>shift)&15];
        return out;
    }
};

bool valid_key(const std::string& key) {
    return key.size()==64 && std::all_of(key.begin(),key.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});
}
bool managed_temp(const std::string& name) {
    if(name.size()<75 || !valid_key(name.substr(0,64)) || name.substr(64,5)!=".spc." || !name.ends_with(".tmp"))return false;
    const auto suffix=name.substr(69,name.size()-73);
    const auto dot=suffix.find('.');
    if(dot==std::string::npos || dot==0 || dot+1==suffix.size())return false;
    for(size_t i=0;i<suffix.size();++i)
        if(i!=dot && (suffix[i]<'0' || suffix[i]>'9'))return false;
    return true;
}

struct Writer {
    std::ofstream out;
    Sha256 digest;
    uint64_t left;
    Writer(const std::filesystem::path& p,uint64_t cap):out(p,std::ios::binary|std::ios::trunc),left(cap) {}
    void raw(const void* p,size_t n) {
        if(!n)return;
        if(n>left) throw std::runtime_error("entry exceeds disk budget");
        out.write(static_cast<const char*>(p),std::streamsize(n));
        if(!out) throw std::runtime_error("cache write failed");
        left-=n;digest.update(p,n);
    }
    void num(uint64_t v) { uint8_t b[8];for(int i=0;i<8;++i)b[i]=uint8_t(v>>(8*i));raw(b,8); }
    template<class T> void vec(const std::vector<T>& v) { num(v.size());raw(v.data(),v.size()*sizeof(T)); }
    void buffer(const ConversationBuffer& b) {
        num(b.size());b.visit(0,b.size(),[&](const uint8_t* p,size_t n,size_t){raw(p,n);return true;});
    }
};
struct Reader {
    std::ifstream in;
    Sha256 digest;
    uint64_t left;
    Reader(const std::filesystem::path& p,uint64_t n):in(p,std::ios::binary),left(n) {}
    void raw(void* p,size_t n) {
        if(!n)return;
        if(n>left) throw std::runtime_error("truncated cache entry");
        in.read(static_cast<char*>(p),std::streamsize(n));
        if(!in) throw std::runtime_error("cache read failed");
        left-=n;digest.update(p,n);
    }
    uint64_t num() { uint8_t b[8];raw(b,8);uint64_t v=0;for(int i=0;i<8;++i)v|=uint64_t(b[i])<<(8*i);return v; }
    size_t count(size_t width,uint64_t cap=UINT64_MAX) {
        const auto n=num();
        if(n>cap || n>left/width || n>SIZE_MAX/width) throw std::runtime_error("invalid cache allocation length");
        return size_t(n);
    }
    template<class T> void vec(std::vector<T>& v,uint64_t cap=UINT64_MAX) {
        v.resize(count(sizeof(T),cap));raw(v.data(),v.size()*sizeof(T));
    }
    void buffer(ConversationBuffer& b) {
        b.resize(count(1));b.visit(0,b.size(),[&](uint8_t* p,size_t n,size_t){raw(p,n);return true;});
    }
};
void write_checkpoint(Writer& w,const ConversationCheckpoint& c) {
    w.vec(c.ids);w.num(c.imgs.size());
    for(const auto& i:c.imgs){w.num(uint64_t(i.start));w.num(i.hash);}
    w.vec(c.gdn);w.vec(c.ple);w.vec(c.tails);w.vec(c.dead);w.vec(c.block_pos);
}
void read_checkpoint(Reader& r,ConversationCheckpoint& c,int64_t max_tokens) {
    r.vec(c.ids,uint64_t(max_tokens));c.imgs.resize(r.count(16,uint64_t(max_tokens)));
    for(auto& i:c.imgs){i.start=int64_t(r.num());i.hash=r.num();}
    r.vec(c.gdn);r.vec(c.ple);r.vec(c.tails);r.vec(c.dead);r.vec(c.block_pos);
}
void write_kv(Writer& w,const ConversationKv& k) {
    for(auto v:{int64_t(k.format),k.cells,k.heads,k.head_dim,k.page_size,k.pooled_rows,k.idx_dim})w.num(uint64_t(v));
    for(const auto* b:{&k.k,&k.v,&k.k_scale,&k.v_scale,&k.pooled})w.buffer(*b);
}
void read_kv(Reader& r,ConversationKv& k) {
    const auto format=r.num();
    if(format>3 && format!=16 && format!=17)throw std::runtime_error("invalid K/V format");
    k.format=int(format);
    for(auto* v:{&k.cells,&k.heads,&k.head_dim,&k.page_size,&k.pooled_rows,&k.idx_dim}) {
        const auto n=r.num();if(n>uint64_t(INT64_MAX))throw std::runtime_error("invalid K/V extent");*v=int64_t(n);
    }
    for(auto* b:{&k.k,&k.v,&k.k_scale,&k.v_scale,&k.pooled})r.buffer(*b);
}
constexpr char magic[8]={'S','T','R','P','K','V','0','1'};

// v2 chained entries: [payload][metadata][512-byte footer]. The payload holds
// every chunk's added bytes in append order; a child reflinks or copies its
// parent's payload and appends only what changed. Metadata (rewritten per save)
// maps each blob to its spans; the footer carries the payload's SHA-256 state,
// so a child extends the digest without re-hashing the shared prefix.
constexpr char magic2[8]={'S','T','R','P','K','V','0','2'};
constexpr size_t footer_size=512,state_size=112;

std::vector<DiskChunkBlob*> blobs_of(DiskChunkFile& f) {
    std::vector<DiskChunkBlob*> out;
    for(auto& s:f.stages){for(auto& b:s.running)out.push_back(&b);for(auto& k:s.kv)for(auto& b:k.data)out.push_back(&b);}
    for(auto& b:f.draft.data)out.push_back(&b);
    return out;
}
std::vector<const DiskChunkBlob*> blobs_of(const DiskChunkFile& f) {
    std::vector<const DiskChunkBlob*> out;
    for(const auto& s:f.stages){for(const auto& b:s.running)out.push_back(&b);for(const auto& k:s.kv)for(const auto& b:k.data)out.push_back(&b);}
    for(const auto& b:f.draft.data)out.push_back(&b);
    return out;
}
std::vector<DiskChunkSpan> truncate_spans(const std::vector<DiskChunkSpan>& in,uint64_t keep) {
    std::vector<DiskChunkSpan> out;uint64_t at=0;
    for(const auto& s:in) {
        if(at>=keep)break;
        const auto take=std::min(s.size,keep-at);
        out.push_back({s.offset,take});at+=take;
    }
    return out;
}

struct BufWriter {
    std::string s;
    void raw(const void* p,size_t n){s.append(static_cast<const char*>(p),n);}
    void num(uint64_t v){char b[8];for(int i=0;i<8;++i)b[i]=char(v>>(8*i));raw(b,8);}
    template<class T> void vec(const std::vector<T>& v){num(v.size());raw(v.data(),v.size()*sizeof(T));}
};
struct BufReader {
    const char* p;size_t n,at=0;
    void raw(void* d,size_t c){if(c>n-at)throw std::runtime_error("truncated chunk metadata");std::memcpy(d,p+at,c);at+=c;}
    uint64_t num(){uint8_t b[8];raw(b,8);uint64_t v=0;for(int i=0;i<8;++i)v|=uint64_t(b[i])<<(8*i);return v;}
    size_t count(size_t width,uint64_t cap){
        const auto c=num();
        if(c>cap||c>(n-at)/width)throw std::runtime_error("invalid chunk metadata length");
        return size_t(c);
    }
};

std::string build_metadata(const std::string& identity,const std::string& key,const DiskChunkFile& f,
                           const std::vector<std::vector<DiskChunkSpan>>& spans) {
    BufWriter w;
    w.raw(identity.data(),64);w.raw(key.data(),64);w.num(f.cvec);w.num(f.stages.size());
    size_t at=0;
    auto blob=[&](const DiskChunkBlob& b){
        w.num(b.size);w.num(spans[at].size());
        for(const auto& s:spans[at++]){w.num(s.offset);w.num(s.size);}
    };
    auto kv=[&](const DiskChunkKv& k){
        for(auto v:{int64_t(k.format),k.cells,k.heads,k.head_dim,k.page_size,k.pooled_rows,k.idx_dim})w.num(uint64_t(v));
        for(size_t j=0;j<5;++j)blob(k.data[j]);
    };
    for(const auto& s:f.stages) {
        w.num(uint64_t(s.layer_lo));w.num(uint64_t(s.layer_hi));w.vec(s.ids);w.num(s.images.size());
        for(const auto& i:s.images){w.num(uint64_t(i.start));w.num(i.hash);}
        for(size_t j=0;j<5;++j)blob(s.running[j]);
        w.num(s.kv.size());for(const auto& k:s.kv)kv(k);
    }
    kv(f.draft);
    return w.s;
}
// Fills every blob's size and spans; spans must tile [0, payload_end) per blob.
bool parse_metadata(const std::string& meta,const std::string& identity,const std::string& key,
                    int64_t max_tokens,uint64_t payload_end,DiskChunkFile& f,std::string& error) {
    try {
        BufReader r{meta.data(),meta.size()};
        char id[64],k[64];r.raw(id,64);r.raw(k,64);
        if(std::string(id,64)!=identity||std::string(k,64)!=key)throw std::runtime_error("cache identity mismatch");
        const auto cvec=r.num();if(cvec>1)throw std::runtime_error("invalid steering mode");
        f.cvec=cvec!=0;f.stages.resize(r.count(16,16));
        if(f.stages.empty())throw std::runtime_error("missing device stages");
        auto blob=[&](DiskChunkBlob& b){
            b.size=r.num();b.offset=0;b.spans.resize(r.count(16,1u<<20));
            uint64_t total=0;
            for(auto& s:b.spans) {
                s.offset=r.num();s.size=r.num();
                if(s.size>payload_end||s.offset>payload_end-s.size||total>payload_end-s.size)
                    throw std::runtime_error("chunk span outside payload");
                total+=s.size;
            }
            if(total!=b.size||b.size>payload_end)throw std::runtime_error("chunk span size mismatch");
        };
        auto kv=[&](DiskChunkKv& k){
            const auto format=r.num();
            if(format>3&&format!=16&&format!=17)throw std::runtime_error("invalid K/V format");
            k.format=int(format);
            for(auto* v:{&k.cells,&k.heads,&k.head_dim,&k.page_size,&k.pooled_rows,&k.idx_dim}) {
                const auto n=r.num();if(n>uint64_t(INT64_MAX))throw std::runtime_error("invalid K/V extent");*v=int64_t(n);
            }
            for(auto& b:k.data)blob(b);
        };
        for(auto& s:f.stages) {
            s.layer_lo=int64_t(r.num());s.layer_hi=int64_t(r.num());
            s.ids.resize(r.count(4,uint64_t(max_tokens)));r.raw(s.ids.data(),s.ids.size()*sizeof(int32_t));
            s.images.resize(r.count(16,uint64_t(max_tokens)));
            for(auto& i:s.images){i.start=int64_t(r.num());i.hash=r.num();}
            for(auto& b:s.running)blob(b);
            s.kv.resize(r.count(96,128));for(auto& k:s.kv)kv(k);
        }
        kv(f.draft);
        return true;
    }catch(const std::exception& e){error=e.what();return false;}
}

struct Footer {
    uint64_t meta_off=0,meta_size=0,payload_end=0;
    uint8_t state[state_size]{};
    char meta_digest[64]{},parent[64]{};
};
bool read_footer(std::ifstream& in,uint64_t size,Footer& f) {
    if(size<footer_size)return false;
    char buf[footer_size];
    in.seekg(std::streamoff(size-footer_size));in.read(buf,footer_size);
    if(!in||std::memcmp(buf,magic2,8))return false;
    auto u64=[&](size_t at){uint64_t v=0;for(int i=0;i<8;++i)v|=uint64_t(uint8_t(buf[at+i]))<<(8*i);return v;};
    f.meta_off=u64(8);f.meta_size=u64(16);f.payload_end=u64(24);
    std::memcpy(f.state,buf+32,state_size);
    std::memcpy(f.meta_digest,buf+144,64);
    std::memcpy(f.parent,buf+208,64);
    const auto used=uint64_t(f.state[104])|(uint64_t(f.state[105])<<8)|(uint64_t(f.state[106])<<16)|(uint64_t(f.state[107])<<24);
    uint64_t bytes=0;for(int i=0;i<8;++i)bytes|=uint64_t(f.state[96+i])<<(8*i);
    return used<64&&bytes==f.payload_end&&f.meta_off==f.payload_end&&
           f.meta_size>=64&&f.payload_end<=size-footer_size-f.meta_size+1&&
           f.payload_end+f.meta_size+footer_size==size;
}
// A chain parent must yield its spans before its child can be written.
bool read_parent(const std::filesystem::path& p,const std::string& identity,const std::string& key,
                 int64_t max_tokens,Footer& f,DiskChunkFile& img,std::string& error) {
    std::error_code ec;const auto size=std::filesystem::file_size(p,ec);
    if(ec||size<footer_size+64)return false;
    std::ifstream in(p,std::ios::binary);
    if(!in||!read_footer(in,size,f))return false;
    std::string meta(size_t(f.meta_size),'\0');
    in.seekg(std::streamoff(f.meta_off));in.read(meta.data(),std::streamsize(meta.size()));
    if(!in)return false;
    Sha256 md;md.update(meta.data(),meta.size());
    if(md.finish()!=std::string(f.meta_digest,64))return false;
    return parse_metadata(meta,identity,key,max_tokens,f.payload_end,img,error);
}
// Copy-on-write clone where the filesystem supports it; the caller falls back
// to a byte copy, which is correct but pays the full rewrite again.
bool reflink_prefix(const std::filesystem::path& src,const std::filesystem::path& dst) {
#if defined(__linux__)
    const int from=::open(src.c_str(),O_RDONLY|O_CLOEXEC);
    if(from<0)return false;
    const int to=::open(dst.c_str(),O_RDWR|O_CLOEXEC|O_CREAT|O_TRUNC,0600);
    if(to<0){::close(from);return false;}
    const bool ok=::ioctl(to,FICLONE,from)==0;
    ::close(from);::close(to);
    return ok;
#else
    (void)src;(void)dst;return false;
#endif
}
}

size_t DiskChunkState::bytes() const {
    size_t n=stages.capacity()*sizeof(DiskChunkStage)+draft.bytes();
    for(const auto& s:stages){n+=s.running.bytes()+s.kv.capacity()*sizeof(ConversationKv);for(const auto& k:s.kv)n+=k.bytes();}
    return n;
}
std::string chunk_cache_disk_sha256(const std::string& value) { Sha256 h;h.update(value.data(),value.size());return h.finish(); }

ChunkCacheDisk::ChunkCacheDisk(std::filesystem::path root,std::string identity,uint64_t budget,int days,int64_t max_tokens)
    :identity_(std::move(identity)),budget_(budget),days_(days),max_tokens_(max_tokens) {
    if(root.empty()||!valid_key(identity_)||!budget||days<=0||max_tokens<=0)return;
    std::error_code ec;
    root_=std::move(root)/"prompt-v1";
    std::filesystem::create_directories(root_,ec);
    if(ec){root_.clear();return;}
    // Prompt tokens and model state are private to this machine/user.
    std::filesystem::permissions(root_,std::filesystem::perms::owner_all,std::filesystem::perm_options::replace,ec);
    startup_cleanup();
}
std::filesystem::path ChunkCacheDisk::path(const std::string& key) const {
    return enabled()&&valid_key(key) ? root_/(key+".spc") : std::filesystem::path{};
}
const std::vector<DiskChunkKey>& ChunkCacheDisk::keys(const std::vector<int64_t>& ids,
        const std::vector<ConversationImageKey>& images,bool cvec,int64_t chunk_tokens,int64_t prefix_tokens) {
    hashed_chunks_=0;
    if(!enabled()||chunk_tokens<=0||prefix_tokens<=0||ids.empty()) {
        keys_.clear();key_ids_.clear();key_images_.clear();return keys_;
    }
    // The final prompt token always starts generation; persist full chunks only.
    const auto limit=std::min({prefix_tokens,max_tokens_,int64_t(ids.size()-1)});
    const auto covered=limit/chunk_tokens*chunk_tokens;
    int64_t common=0;
    if(cvec==key_cvec_ && chunk_tokens==key_chunk_tokens_) {
        const auto compare=std::min(covered,int64_t(key_ids_.size()));
        while(common<compare && ids[size_t(common)]==key_ids_[size_t(common)])++common;
        // Images affect the chunk containing their first token and all descendants.
        size_t old=0,fresh=0;
        while(old<key_images_.size() || fresh<images.size()) {
            const auto before=old<key_images_.size() ? key_images_[old].start : INT64_MAX;
            const auto after=fresh<images.size() ? images[fresh].start : INT64_MAX;
            const auto at=std::min(before,after);
            if(at>=common)break;
            if(before!=after || key_images_[old].hash!=images[fresh].hash) {common=at;break;}
            ++old;++fresh;
        }
    }
    const auto reused=size_t(common/chunk_tokens);
    keys_.resize(reused);
    size_t pos=reused*size_t(chunk_tokens),image=0;
    while(image<images.size() && images[image].start<int64_t(pos))++image;
    std::string parent;
    if(int64_t(pos)<covered) {
        if(reused)parent=keys_.back().hash;
        else {
            Sha256 seed;seed.text("strata-chunk-v1");seed.text(identity_);seed.number(cvec);seed.number(chunk_tokens);
            parent=seed.finish();
        }
    }
    for(int64_t end=int64_t(pos)+chunk_tokens;end<=covered;end+=chunk_tokens) {
        Sha256 h;h.text(parent);h.number(chunk_tokens);
        for(;pos<size_t(end);++pos) {
            h.number(uint64_t(ids[pos]));
            if(image<images.size() && images[image].start==int64_t(pos)) {
                h.number(1);h.number(images[image++].hash);
            }else h.number(0);
        }
        parent=h.finish();keys_.push_back({end,parent});++hashed_chunks_;
    }
    // Preserve the token allocation on the usual unchanged-prefix path.
    if(common!=covered || int64_t(key_ids_.size())!=covered)
        key_ids_.assign(ids.begin(),ids.begin()+covered);
    key_images_.clear();
    for(const auto& im:images)if(im.start<covered)key_images_.push_back(im);
    key_chunk_tokens_=chunk_tokens;key_cvec_=cvec;
    return keys_;
}
uint64_t ChunkCacheDisk::file_bytes(const std::string& key) const {
    const auto p=path(key);if(p.empty())return 0;
    std::error_code ec;
    if(!std::filesystem::is_regular_file(std::filesystem::symlink_status(p,ec))||ec)return 0;
    const auto size=std::filesystem::file_size(p,ec);
    return ec||size<200||size>budget_ ? 0 : size;
}
bool ChunkCacheDisk::contains(const std::string& key) const {
    std::lock_guard<std::mutex> lock(index_mu_);
    return cached_.find(key)!=cached_.end();
}
void ChunkCacheDisk::touch(const std::string& key) {
    std::lock_guard<std::mutex> lock(index_mu_);
    const auto it=cached_.find(key);if(it==cached_.end())return;
    it->second.used=std::filesystem::file_time_type::clock::now();it->second.dirty=true;
}
void ChunkCacheDisk::flush_touches() {
    std::lock_guard<std::mutex> lock(index_mu_);
    for(auto it=cached_.begin();it!=cached_.end();) {
        if(!it->second.dirty){++it;continue;}
        std::error_code ec;std::filesystem::last_write_time(path(it->first),it->second.used,ec);
        if(ec) {
            // A removed file is a miss; retry other errors on the next idle flush.
            if(ec==std::errc::no_such_file_or_directory) {
                cached_bytes_-=it->second.bytes;it=cached_.erase(it);continue;
            }
        }else it->second.dirty=false;
        ++it;
    }
}
void ChunkCacheDisk::forget_locked(const std::string& key) {
    const auto it=cached_.find(key);
    if(it!=cached_.end()){cached_bytes_-=it->second.bytes;cached_.erase(it);}
}
void ChunkCacheDisk::discard(const std::string& key) {
    const auto p=path(key);if(p.empty())return;
    std::error_code ec;std::filesystem::remove(p,ec);
    if(!ec){std::lock_guard<std::mutex> lock(index_mu_);forget_locked(key);}
}
void ChunkCacheDisk::published(const std::string& key,uint64_t bytes) {
    std::lock_guard<std::mutex> lock(index_mu_);
    forget_locked(key);
    // Enforce the cap from the in-memory index, without another directory scan.
    while(cached_bytes_>budget_-bytes && !cached_.empty()) {
        auto oldest=cached_.end();
        for(auto it=cached_.begin();it!=cached_.end();++it)
            if(oldest==cached_.end() || it->second.used<oldest->second.used)oldest=it;
        const auto victim=oldest->first;
        std::error_code ec;std::filesystem::remove(path(victim),ec);
        if(ec) {
            // Do not accept a new entry if the cap cannot be enforced.
            std::filesystem::remove(path(key),ec);return;
        }
        forget_locked(victim);
    }
    cached_[key]={std::filesystem::file_time_type::clock::now(),bytes,false};cached_bytes_+=bytes;
}
void ChunkCacheDisk::startup_cleanup() {
    if(!enabled())return;
    struct StartupEntry { std::filesystem::path p;std::filesystem::file_time_type at;uint64_t size; };
    std::vector<StartupEntry> entries;uint64_t total=0;std::error_code ec;
    cached_.clear();cached_bytes_=0;
    const auto now=std::filesystem::file_time_type::clock::now();
    for(std::filesystem::directory_iterator it(root_,ec),end;!ec&&it!=end;it.increment(ec)) {
        const auto p=it->path();const auto name=p.filename().string();
        const bool temporary=managed_temp(name);
        if(!temporary && (name.size()!=68||name.substr(64)!=".spc"||!valid_key(name.substr(0,64))))continue;
        std::error_code e;
        if(!std::filesystem::is_regular_file(it->symlink_status(e))||e)continue;
        const auto at=it->last_write_time(e);if(e)continue;
        const auto size=it->file_size(e);if(e)continue;
        if(temporary) {
            if(std::chrono::duration_cast<std::chrono::hours>(now-at).count()>=int64_t(days_)*24)
                std::filesystem::remove(p,e);
            continue;
        }
        if(std::chrono::duration_cast<std::chrono::hours>(now-at).count()>=int64_t(days_)*24||size<200||size>budget_||size>UINT64_MAX-total) {
            std::filesystem::remove(p,e);continue;
        }
        total+=size;entries.push_back({p,at,size});
    }
    std::sort(entries.begin(),entries.end(),[](const auto& a,const auto& b){return a.at<b.at;});
    for(const auto& e:entries) {
        if(total>budget_) {
            std::error_code ignored;
            if(std::filesystem::remove(e.p,ignored)){total-=e.size;continue;}
        }
        cached_.emplace(e.p.stem().string(),Entry{e.at,e.size,false});cached_bytes_+=e.size;
    }
}
std::optional<DiskChunkState> ChunkCacheDisk::load(const std::string& key,std::string& error) {
    error.clear();const auto size=file_bytes(key);
    if(!size){std::lock_guard<std::mutex> lock(index_mu_);forget_locked(key);return std::nullopt;}
    try {
        Reader r(path(key),size-64);char head[8];r.raw(head,8);
        if(std::memcmp(head,magic,8))throw std::runtime_error("unsupported prompt cache schema");
        char identity[64],stored_key[64];r.raw(identity,64);r.raw(stored_key,64);
        if(std::string(identity,64)!=identity_||std::string(stored_key,64)!=key)throw std::runtime_error("cache identity mismatch");
        const auto cvec=r.num();if(cvec>1)throw std::runtime_error("invalid steering mode");
        DiskChunkState state;state.cvec=cvec!=0;state.stages.resize(r.count(16,16));
        if(state.stages.empty())throw std::runtime_error("missing device stages");
        for(auto& s:state.stages) {
            s.layer_lo=int64_t(r.num());s.layer_hi=int64_t(r.num());read_checkpoint(r,s.running,max_tokens_);
            s.kv.resize(r.count(96,128));for(auto& k:s.kv)read_kv(r,k);
        }
        read_kv(r,state.draft);
        if(r.left)throw std::runtime_error("trailing cache payload");
        char checksum[64];r.in.read(checksum,64);
        if(!r.in||r.digest.finish()!=std::string(checksum,64))throw std::runtime_error("cache checksum mismatch");
        return state;
    }catch(const std::exception& e){error=e.what();discard(key);return std::nullopt;}
}
bool ChunkCacheDisk::store(const std::string& key,const DiskChunkState& state,std::string& error) {
    error.clear();const auto target=path(key);if(target.empty())return false;
    std::filesystem::path temp;
    try {
        // A snapshot larger than the cap is refused before opening any file.
        if(state.bytes()>budget_)return false;
        temp=target;temp+="."+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+
                           "."+std::to_string(std::random_device{}())+".tmp";
        Writer w(temp,budget_-std::min<uint64_t>(budget_,64));
        w.raw(magic,8);w.raw(identity_.data(),64);w.raw(key.data(),64);w.num(state.cvec);w.num(state.stages.size());
        for(const auto& s:state.stages) {
            w.num(uint64_t(s.layer_lo));w.num(uint64_t(s.layer_hi));write_checkpoint(w,s.running);
            w.num(s.kv.size());for(const auto& k:s.kv)write_kv(w,k);
        }
        write_kv(w,state.draft);const auto sum=w.digest.finish();w.out.write(sum.data(),64);w.out.flush();
        if(!w.out)throw std::runtime_error("cache flush failed");
        w.out.close();
        if(!w.out)throw std::runtime_error("cache close failed");
        std::error_code ec;
        std::filesystem::permissions(temp,std::filesystem::perms::owner_read|std::filesystem::perms::owner_write,
                                     std::filesystem::perm_options::replace,ec);
        std::filesystem::rename(temp,target,ec);
        if(ec)throw std::runtime_error("cache publish failed: "+ec.message());
        published(key,budget_-w.left);return contains(key);
    }catch(const std::exception& e){error=e.what();std::error_code ec;if(!temp.empty())std::filesystem::remove(temp,ec);return false;}
}
bool DiskChunkFile::read(const DiskChunkBlob& blob, void* target, size_t offset, size_t count) const {
    if(!file || offset>blob.size || count>blob.size-offset)return false;
    std::vector<DiskChunkSpan> single;
    const std::vector<DiskChunkSpan>* spans=&blob.spans;
    if(spans->empty()){single.push_back({blob.offset,blob.size});spans=&single;}
    auto* dst=static_cast<uint8_t*>(target);
    uint64_t logical=0,left=count;
    for(const auto& s:*spans) {
        if(!left)break;
        const uint64_t end=logical+s.size;
        if(end>offset&&logical<offset+count) {
            const auto from=std::max(logical,offset),to=std::min(end,offset+count);
            file->clear();file->seekg(std::streamoff(s.offset+(from-logical)));
            file->read(reinterpret_cast<char*>(dst+(from-offset)),std::streamsize(to-from));
            if(!*file)return false;
            left-=size_t(to-from);
        }
        logical=end;
    }
    return left==0;
}
size_t DiskChunkFile::bytes() const {
    size_t n=stages.capacity()*sizeof(DiskChunkFileStage);
    for(const auto& s:stages)n+=s.ids.capacity()*sizeof(int32_t)+s.images.capacity()*sizeof(ConversationImageKey)+
                              s.kv.capacity()*sizeof(DiskChunkKv);
    return n;
}
namespace {
void read_blob(Reader& r,DiskChunkBlob& b) {
    b.size=r.count(1);b.offset=uint64_t(r.in.tellg());std::array<uint8_t,65536> scratch;
    for(uint64_t at=0;at<b.size;) {
        const auto n=size_t(std::min<uint64_t>(scratch.size(),b.size-at));r.raw(scratch.data(),n);at+=n;
    }
}
void read_stream_kv(Reader& r,DiskChunkKv& k) {
    const auto format=r.num();
    if(format>3 && format!=16 && format!=17)throw std::runtime_error("invalid K/V format");
    k.format=int(format);
    for(auto* v:{&k.cells,&k.heads,&k.head_dim,&k.page_size,&k.pooled_rows,&k.idx_dim}) {
        const auto n=r.num();if(n>uint64_t(INT64_MAX))throw std::runtime_error("invalid K/V extent");*v=int64_t(n);
    }
    for(auto& b:k.data)read_blob(r,b);
}
}
std::optional<DiskChunkFile> ChunkCacheDisk::load_stream(const std::string& key,std::string& error) {
    error.clear();const auto size=file_bytes(key);
    if(!size){std::lock_guard<std::mutex> lock(index_mu_);forget_locked(key);return std::nullopt;}
    try {
        std::ifstream probe(path(key),std::ios::binary);
        Footer f;
        if(probe&&read_footer(probe,size,f)) {
            // v2 chained entry: verify metadata, then the payload against the
            // digest state the footer carries (the shared prefix included).
            std::string meta(size_t(f.meta_size),'\0');
            probe.seekg(std::streamoff(f.meta_off));probe.read(meta.data(),std::streamsize(meta.size()));
            if(!probe)throw std::runtime_error("cache read failed");
            Sha256 md;md.update(meta.data(),meta.size());
            if(md.finish()!=std::string(f.meta_digest,64))throw std::runtime_error("cache checksum mismatch");
            DiskChunkFile image;
            if(!parse_metadata(meta,identity_,key,max_tokens_,f.payload_end,image,error))
                throw std::runtime_error(error.empty()?"invalid chunk metadata":error);
            const auto expected=Sha256::restore(f.state).finish();
            Sha256 fresh;std::array<uint8_t,65536> scratch;uint64_t left=f.payload_end;
            std::ifstream in(path(key),std::ios::binary);
            while(left) {
                const auto n=size_t(std::min<uint64_t>(scratch.size(),left));
                in.read(reinterpret_cast<char*>(scratch.data()),std::streamsize(n));
                if(!in)throw std::runtime_error("cache read failed");
                fresh.update(scratch.data(),n);left-=n;
            }
            if(fresh.finish()!=expected)throw std::runtime_error("cache checksum mismatch");
            image.file=std::make_shared<std::ifstream>(path(key),std::ios::binary);
            if(!*image.file)throw std::runtime_error("cache open failed");
            return image;
        }
        Reader r(path(key),size-64);char head[8];r.raw(head,8);
        if(std::memcmp(head,magic,8))throw std::runtime_error("unsupported prompt cache schema");
        char identity[64],stored_key[64];r.raw(identity,64);r.raw(stored_key,64);
        if(std::string(identity,64)!=identity_||std::string(stored_key,64)!=key)throw std::runtime_error("cache identity mismatch");
        const auto mode=r.num();if(mode>1)throw std::runtime_error("invalid steering mode");
        DiskChunkFile image;image.cvec=mode!=0;image.stages.resize(r.count(16,16));
        if(image.stages.empty())throw std::runtime_error("missing device stages");
        for(auto& s:image.stages) {
            s.layer_lo=int64_t(r.num());s.layer_hi=int64_t(r.num());r.vec(s.ids,uint64_t(max_tokens_));
            s.images.resize(r.count(16,uint64_t(max_tokens_)));
            for(auto& i:s.images){i.start=int64_t(r.num());i.hash=r.num();}
            for(auto& b:s.running)read_blob(r,b);
            s.kv.resize(r.count(96,128));for(auto& k:s.kv)read_stream_kv(r,k);
        }
        read_stream_kv(r,image.draft);
        if(r.left)throw std::runtime_error("trailing cache payload");
        char checksum[64];r.in.read(checksum,64);
        if(!r.in || r.digest.finish()!=std::string(checksum,64))throw std::runtime_error("cache checksum mismatch");
        image.file=std::make_shared<std::ifstream>(std::move(r.in));return image;
    }catch(const std::exception& e){error=e.what();discard(key);return std::nullopt;}
}
bool ChunkCacheDisk::store_stream(const std::string& key,const DiskChunkFile& state,const std::string& parent,std::string& error) {
    error.clear();const auto target=path(key);if(target.empty())return false;
    std::filesystem::path temp;
    try {
        const auto blobs=blobs_of(state);
        // A usable parent contributes its payload prefix, its spans and its
        // running payload digest; without one every blob must be captured whole.
        DiskChunkFile pimg;Footer pf;std::vector<const DiskChunkBlob*> pblobs;
        uint64_t payload_end=0;bool chained=false;
        if(!parent.empty()&&parent!=key&&contains(parent)) {
            std::string pe;
            if(read_parent(path(parent),identity_,parent,max_tokens_,pf,pimg,pe)) {
                pblobs=blobs_of(std::as_const(pimg));
                chained=pblobs.size()==blobs.size();
                for(size_t i=0;chained&&i<blobs.size();++i)
                    chained=pblobs[i]->reuse==0&&pblobs[i]->spans.size()<= (1u<<20)&&pblobs[i]->size>=blobs[i]->reuse;
                if(chained)payload_end=pf.payload_end;
            }
        }
        if(!chained) {
            pblobs.clear();
            for(const auto* b:blobs)if(b->reuse)throw std::runtime_error("chain parent unavailable");
        }
        uint64_t delta=0;
        for(const auto* b:blobs)delta+=b->size-(chained?b->reuse:0);
        if(delta>budget_||payload_end+footer_size>budget_-delta)throw std::runtime_error("entry exceeds disk budget");
        temp=target;temp+="."+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+
                           "."+std::to_string(std::random_device{}())+".tmp";
        Sha256 payload;
        if(chained) {
            payload=Sha256::restore(pf.state);
            std::error_code ec;
            if(!reflink_prefix(path(parent),temp)) {
                std::ifstream in(path(parent),std::ios::binary);
                std::ofstream copy(temp,std::ios::binary|std::ios::trunc);
                std::array<uint8_t,1<<20> buf;uint64_t left=pf.payload_end;
                while(left) {
                    const auto n=size_t(std::min<uint64_t>(buf.size(),left));
                    in.read(reinterpret_cast<char*>(buf.data()),std::streamsize(n));
                    copy.write(reinterpret_cast<const char*>(buf.data()),std::streamsize(n));
                    if(!in||!copy)throw std::runtime_error("chain copy failed");
                    left-=n;
                }
                copy.close();
            }
            std::filesystem::resize_file(temp,pf.payload_end,ec);
            if(ec)throw std::runtime_error("chain truncate failed: "+ec.message());
        }
        // `app` (not `ate`): libstdc++ maps ate|out to fopen "wb", which would
        // truncate the cloned parent payload before anything is appended.
        std::ofstream out(temp,std::ios::binary|std::ios::app);
        if(!out)throw std::runtime_error("cache write failed");
        uint64_t left=budget_-payload_end-footer_size,total=payload_end;
        auto raw=[&](const void* p,size_t n){
            if(!n)return;
            if(n>left)throw std::runtime_error("entry exceeds disk budget");
            out.write(static_cast<const char*>(p),std::streamsize(n));
            if(!out)throw std::runtime_error("cache write failed");
            left-=n;total+=n;
        };
        std::vector<std::vector<DiskChunkSpan>> spans(blobs.size());
        std::array<uint8_t,65536> scratch;
        for(size_t i=0;i<blobs.size();++i) {
            const auto* b=blobs[i];
            if(chained)spans[i]=truncate_spans(pblobs[i]->spans,b->reuse);
            const uint64_t from=chained?b->reuse:0;
            if(b->size>from) {
                const auto at=payload_end;
                for(uint64_t off=from;off<b->size;) {
                    const auto n=size_t(std::min<uint64_t>(scratch.size(),b->size-off));
                    if(!b->read||!b->read(scratch.data(),n,size_t(off)))throw std::runtime_error("snapshot transfer failed");
                    raw(scratch.data(),n);payload.update(scratch.data(),n);off+=n;
                }
                spans[i].push_back({at,b->size-from});payload_end+=b->size-from;
            }
        }
        const auto meta=build_metadata(identity_,key,state,spans);
        raw(meta.data(),meta.size());
        Sha256 md;md.update(meta.data(),meta.size());
        std::array<char,footer_size> foot{};
        std::memcpy(foot.data(),magic2,8);
        auto put=[&](size_t at,uint64_t v){for(int i=0;i<8;++i)foot[at+i]=char(v>>(8*i));};
        put(8,payload_end);put(16,meta.size());put(24,payload_end);
        payload.store(reinterpret_cast<uint8_t*>(foot.data())+32);
        std::memcpy(foot.data()+144,md.finish().data(),64);
        if(chained)std::memcpy(foot.data()+208,parent.data(),64);
        raw(foot.data(),foot.size());
        out.flush();if(!out)throw std::runtime_error("cache flush failed");
        out.close();if(!out)throw std::runtime_error("cache close failed");
        std::error_code ec;
        std::filesystem::permissions(temp,std::filesystem::perms::owner_read|std::filesystem::perms::owner_write,
                                     std::filesystem::perm_options::replace,ec);
        std::filesystem::rename(temp,target,ec);
        if(ec)throw std::runtime_error("cache publish failed: "+ec.message());
        published(key,total);return contains(key);
    }catch(const std::exception& e){error=e.what();std::error_code ec;if(!temp.empty())std::filesystem::remove(temp,ec);return false;}
}

std::string chunk_cache_disk_identity(const std::vector<std::string>& options,const std::vector<std::filesystem::path>& artifacts) {
    try {
        Sha256 h;h.text("Strata prompt snapshot v1; little endian host payload");
        const uint32_t endian=1;h.update(&endian,sizeof endian);h.number(sizeof(size_t));
        for(const auto& s:options)h.text(s);
        std::vector<std::string> environment;
#if defined(_WIN32)
        char** vars = _environ;
#else
        char** vars = ::environ;
#endif
        for (; vars && *vars; ++vars) {
            const std::string value = *vars;
            if (value.starts_with("STRATA_") || value.starts_with("CUDA_VISIBLE_DEVICES=") ||
                value.starts_with("HIP_VISIBLE_DEVICES=") || value.starts_with("ROCR_VISIBLE_DEVICES="))
                environment.push_back(value);
        }
        std::sort(environment.begin(), environment.end());
        for (const auto& value : environment) h.text(value);
        std::vector<std::filesystem::path> files;
        for(const auto& p:artifacts) {
            if(p.empty())continue;
            const auto canonical=std::filesystem::canonical(p);h.text(canonical.string());
            if(std::filesystem::is_directory(canonical)) {
                for(const auto& e:std::filesystem::recursive_directory_iterator(canonical))
                    if(e.is_regular_file())files.push_back(e.path());
            }else files.push_back(canonical);
        }
        std::sort(files.begin(),files.end());files.erase(std::unique(files.begin(),files.end()),files.end());
        std::array<char,65536> b;
        for(const auto& p:files) {
            h.text(p.string());const auto size=std::filesystem::file_size(p);h.number(size);
            h.number(uint64_t(std::filesystem::last_write_time(p).time_since_epoch().count()));
            // Large weights are identified by path/size/mtime, without rereading
            // tens of GB. Manifests, configs and custom vocab lists get content hashes.
            if(size<=1024*1024) {
                std::ifstream in(p,std::ios::binary);
                if(!in)throw std::runtime_error("unreadable artifact");
                while(in){in.read(b.data(),b.size());h.update(b.data(),size_t(in.gcount()));}
                if(!in.eof())throw std::runtime_error("artifact read failed");
            }
        }
        return h.finish();
    }catch(const std::exception&){return {};}
}

bool chunk_cache_disk_stage(DiskChunkFile& image,std::string& error) {
    try {
        for(auto* b:blobs_of(image)) {
            if(b->size<=b->reuse)continue;
            const auto from=b->reuse,n=size_t(b->size-from);
            if(n>size_t(INT64_MAX))throw std::runtime_error("chunk too large to stage");
            auto buf=std::make_shared<std::string>();buf->resize(n);
            for(size_t off=0;off<n;) {
                const auto take=std::min<size_t>(65536,n-off);
                if(!b->read||!b->read(buf->data()+off,take,size_t(from+off)))
                    throw std::runtime_error("snapshot transfer failed");
                off+=take;
            }
            b->read=[buf,from](void* d,size_t count,size_t off){
                if(off<from||off-from>buf->size()||count>buf->size()-(off-from))return false;
                std::memcpy(d,buf->data()+(off-from),count);return true;
            };
        }
        return true;
    }catch(const std::exception& e){error=e.what();return false;}
}

ChunkCacheDiskWriter::ChunkCacheDiskWriter(ChunkCacheDisk& cache):cache_(cache) {
    thread_=std::thread([this]{run();});
}
ChunkCacheDiskWriter::~ChunkCacheDiskWriter() {
    drain();
    {std::lock_guard<std::mutex> lock(mu_);stop_=true;}
    cv_.notify_all();
    if(thread_.joinable())thread_.join();
}
bool ChunkCacheDiskWriter::queue(std::string key,std::string parent,DiskChunkFile image,std::string& error) {
    if(!cache_.enabled())return true;
    if(!chunk_cache_disk_stage(image,error))return false;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if(stop_)return false;
        if(items_.size()>=8||!keys_.insert(key).second)return true;
        items_.push_back({std::move(key),std::move(parent),std::move(image)});
    }
    cv_.notify_one();return true;
}
bool ChunkCacheDiskWriter::pending(const std::string& key) const {
    std::lock_guard<std::mutex> lock(mu_);return keys_.count(key)!=0;
}
void ChunkCacheDiskWriter::drain() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock,[this]{return (items_.empty()&&!busy_)||stop_;});
}
void ChunkCacheDiskWriter::run() {
    std::unique_lock<std::mutex> lock(mu_);
    for(;;) {
        cv_.wait(lock,[this]{return stop_||!items_.empty();});
        if(items_.empty())return;
        auto item=std::move(items_.front());items_.pop_front();busy_=true;
        lock.unlock();
        std::string error;
        if(!cache_.store_stream(item.key,item.image,item.parent,error))
            std::cerr<<"disk chunk cache: save failed: "<<error<<"\n";
        lock.lock();
        keys_.erase(item.key);busy_=false;cv_.notify_all();
    }
}
} // namespace strata::core
