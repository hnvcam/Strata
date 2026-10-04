#include "strata/core/prompt_disk_cache.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <cstdlib>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#include <fstream>
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
}

size_t DiskPromptState::bytes() const {
    size_t n=stages.capacity()*sizeof(DiskPromptStage)+draft.bytes();
    for(const auto& s:stages){n+=s.running.bytes()+s.kv.capacity()*sizeof(ConversationKv);for(const auto& k:s.kv)n+=k.bytes();}
    return n;
}
std::string prompt_disk_sha256(const std::string& value) { Sha256 h;h.update(value.data(),value.size());return h.finish(); }

PromptDiskCache::PromptDiskCache(std::filesystem::path root,std::string identity,uint64_t budget,int days,int64_t max_tokens)
    :identity_(std::move(identity)),budget_(budget),days_(days),max_tokens_(max_tokens) {
    if(root.empty()||!valid_key(identity_)||!budget||days<=0||max_tokens<=0)return;
    std::error_code ec;
    root_=std::move(root)/"prompt-v1";
    std::filesystem::create_directories(root_,ec);
    if(ec){root_.clear();return;}
    // Prompt tokens and model state are private to this machine/user.
    std::filesystem::permissions(root_,std::filesystem::perms::owner_all,std::filesystem::perm_options::replace,ec);
    prune();
}
std::filesystem::path PromptDiskCache::path(const std::string& key) const {
    return enabled()&&valid_key(key) ? root_/(key+".spc") : std::filesystem::path{};
}
std::vector<DiskPromptKey> PromptDiskCache::keys(const std::vector<int64_t>& ids,
        const std::vector<ConversationImageKey>& images,bool cvec,const std::vector<int64_t>& boundaries) const {
    std::vector<DiskPromptKey> out;
    if(!enabled())return out;
    Sha256 h;h.text(identity_);h.number(cvec);size_t pos=0,image=0;
    for(auto end:boundaries) {
        if(end<=int64_t(pos)||end>=int64_t(ids.size()))continue;
        for(;pos<size_t(end);++pos) {
            h.number(uint64_t(ids[pos]));
            if(image<images.size()&&images[image].start==int64_t(pos)) {
                h.number(1);h.number(images[image++].hash);
            } else h.number(0);
        }
        out.push_back({end,h.finish()});
    }
    return out;
}
uint64_t PromptDiskCache::file_bytes(const std::string& key) const {
    const auto p=path(key);if(p.empty())return 0;
    std::error_code ec;
    if(!std::filesystem::is_regular_file(std::filesystem::symlink_status(p,ec))||ec)return 0;
    const auto at=std::filesystem::last_write_time(p,ec);
    if(ec||std::chrono::duration_cast<std::chrono::hours>(std::filesystem::file_time_type::clock::now()-at).count()>=int64_t(days_)*24)return 0;
    const auto size=std::filesystem::file_size(p,ec);
    return ec||size<200||size>budget_ ? 0 : size;
}
bool PromptDiskCache::contains(const std::string& key) const { return file_bytes(key)!=0; }
void PromptDiskCache::touch(const std::string& key) {
    const auto p=path(key);if(p.empty())return;
    std::error_code ec;std::filesystem::last_write_time(p,std::filesystem::file_time_type::clock::now(),ec);
}
void PromptDiskCache::discard(const std::string& key) {
    const auto p=path(key);if(p.empty())return;std::error_code ec;std::filesystem::remove(p,ec);
}
void PromptDiskCache::prune() {
    if(!enabled())return;
    struct Entry { std::filesystem::path p;std::filesystem::file_time_type at;uint64_t size; };
    std::vector<Entry> entries;uint64_t total=0;std::error_code ec;
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
        if(std::chrono::duration_cast<std::chrono::hours>(now-at).count()>=int64_t(days_)*24||size>budget_||size>UINT64_MAX-total) {
            std::filesystem::remove(p,e);continue;
        }
        total+=size;entries.push_back({p,at,size});
    }
    std::sort(entries.begin(),entries.end(),[](const auto& a,const auto& b){return a.at<b.at;});
    for(const auto& e:entries) {
        if(total<=budget_)break;
        std::error_code ignored;if(std::filesystem::remove(e.p,ignored))total-=e.size;
    }
}
std::optional<DiskPromptState> PromptDiskCache::load(const std::string& key,std::string& error) {
    error.clear();const auto size=file_bytes(key);if(!size)return std::nullopt;
    try {
        Reader r(path(key),size-64);char head[8];r.raw(head,8);
        if(std::memcmp(head,magic,8))throw std::runtime_error("unsupported prompt cache schema");
        char identity[64],stored_key[64];r.raw(identity,64);r.raw(stored_key,64);
        if(std::string(identity,64)!=identity_||std::string(stored_key,64)!=key)throw std::runtime_error("cache identity mismatch");
        const auto cvec=r.num();if(cvec>1)throw std::runtime_error("invalid steering mode");
        DiskPromptState state;state.cvec=cvec!=0;state.stages.resize(r.count(16,16));
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
bool PromptDiskCache::store(const std::string& key,const DiskPromptState& state,std::string& error) {
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
        prune();return contains(key);
    }catch(const std::exception& e){error=e.what();std::error_code ec;if(!temp.empty())std::filesystem::remove(temp,ec);return false;}
}
bool DiskPromptFile::read(const DiskPromptBlob& blob, void* target, size_t offset, size_t count) const {
    if(!file || offset>blob.size || count>blob.size-offset || blob.offset>uint64_t(INT64_MAX)-offset)return false;
    file->clear();file->seekg(std::streamoff(blob.offset+offset));
    file->read(static_cast<char*>(target),std::streamsize(count));
    return bool(*file);
}
size_t DiskPromptFile::bytes() const {
    size_t n=stages.capacity()*sizeof(DiskPromptFileStage);
    for(const auto& s:stages)n+=s.ids.capacity()*sizeof(int32_t)+s.images.capacity()*sizeof(ConversationImageKey)+
                              s.kv.capacity()*sizeof(DiskPromptKv);
    return n;
}
namespace {
void write_blob(Writer& w,const DiskPromptBlob& b) {
    w.num(b.size);std::array<uint8_t,65536> scratch;
    for(uint64_t at=0;at<b.size;) {
        const auto n=size_t(std::min<uint64_t>(scratch.size(),b.size-at));
        if(!b.read || !b.read(scratch.data(),n,size_t(at)))throw std::runtime_error("snapshot transfer failed");
        w.raw(scratch.data(),n);at+=n;
    }
}
void read_blob(Reader& r,DiskPromptBlob& b) {
    b.size=r.count(1);b.offset=uint64_t(r.in.tellg());std::array<uint8_t,65536> scratch;
    for(uint64_t at=0;at<b.size;) {
        const auto n=size_t(std::min<uint64_t>(scratch.size(),b.size-at));r.raw(scratch.data(),n);at+=n;
    }
}
void write_stream_kv(Writer& w,const DiskPromptKv& k) {
    for(auto v:{int64_t(k.format),k.cells,k.heads,k.head_dim,k.page_size,k.pooled_rows,k.idx_dim})w.num(uint64_t(v));
    for(const auto& b:k.data)write_blob(w,b);
}
void read_stream_kv(Reader& r,DiskPromptKv& k) {
    const auto format=r.num();
    if(format>3 && format!=16 && format!=17)throw std::runtime_error("invalid K/V format");
    k.format=int(format);
    for(auto* v:{&k.cells,&k.heads,&k.head_dim,&k.page_size,&k.pooled_rows,&k.idx_dim}) {
        const auto n=r.num();if(n>uint64_t(INT64_MAX))throw std::runtime_error("invalid K/V extent");*v=int64_t(n);
    }
    for(auto& b:k.data)read_blob(r,b);
}
}
std::optional<DiskPromptFile> PromptDiskCache::load_stream(const std::string& key,std::string& error) {
    error.clear();const auto size=file_bytes(key);if(!size)return std::nullopt;
    try {
        Reader r(path(key),size-64);char head[8];r.raw(head,8);
        if(std::memcmp(head,magic,8))throw std::runtime_error("unsupported prompt cache schema");
        char identity[64],stored_key[64];r.raw(identity,64);r.raw(stored_key,64);
        if(std::string(identity,64)!=identity_||std::string(stored_key,64)!=key)throw std::runtime_error("cache identity mismatch");
        const auto mode=r.num();if(mode>1)throw std::runtime_error("invalid steering mode");
        DiskPromptFile image;image.cvec=mode!=0;image.stages.resize(r.count(16,16));
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
bool PromptDiskCache::store_stream(const std::string& key,const DiskPromptFile& state,std::string& error) {
    error.clear();const auto target=path(key);if(target.empty())return false;
    std::filesystem::path temp;
    try {
        temp=target;temp+="."+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+
                           "."+std::to_string(std::random_device{}())+".tmp";
        Writer w(temp,budget_-std::min<uint64_t>(budget_,64));
        w.raw(magic,8);w.raw(identity_.data(),64);w.raw(key.data(),64);w.num(state.cvec);w.num(state.stages.size());
        for(const auto& s:state.stages) {
            w.num(uint64_t(s.layer_lo));w.num(uint64_t(s.layer_hi));w.vec(s.ids);w.num(s.images.size());
            for(const auto& i:s.images){w.num(uint64_t(i.start));w.num(i.hash);}
            for(const auto& b:s.running)write_blob(w,b);
            w.num(s.kv.size());for(const auto& k:s.kv)write_stream_kv(w,k);
        }
        write_stream_kv(w,state.draft);const auto sum=w.digest.finish();w.out.write(sum.data(),64);w.out.flush();
        if(!w.out)throw std::runtime_error("cache flush failed");
        w.out.close();if(!w.out)throw std::runtime_error("cache close failed");
        std::error_code ec;
        std::filesystem::permissions(temp,std::filesystem::perms::owner_read|std::filesystem::perms::owner_write,
                                     std::filesystem::perm_options::replace,ec);
        std::filesystem::rename(temp,target,ec);
        if(ec)throw std::runtime_error("cache publish failed: "+ec.message());
        prune();return contains(key);
    }catch(const std::exception& e){error=e.what();std::error_code ec;if(!temp.empty())std::filesystem::remove(temp,ec);return false;}
}

std::string prompt_disk_identity(const std::vector<std::string>& options,const std::vector<std::filesystem::path>& artifacts) {
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
} // namespace strata::core
