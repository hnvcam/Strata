// Model-free Qwen3.6 cache checks: quantization, prompt gather, causal attention and batch session selection.
#include "q36_attention.cuh"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::qwen35::attention;
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); } }
template <class T> T* alloc(size_t n) {
    T* p = nullptr; ck(cudaMalloc(&p, n * sizeof(T))); ck(cudaMemset(p, 0, n * sizeof(T))); return p;
}
template <class T> void put(T* p, const std::vector<T>& v) { ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); }
template <class T> std::vector<T> get(const T* p, size_t n) {
    std::vector<T> v(n); ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}
}

int main() {
    constexpr int H = 4, HK = 2, D = 256, C = 300, R = 6, NC = 2;
    const size_t values = HK * C * D;
    std::vector<float> qg(C * H * 2 * D, 0.f), k(C * HK * D), v(k.size()), w(D, 1.f);
    for (int t = 0; t < C; ++t) {
        const float scale = t == 0 ? 0.f : t % 19 == 0 ? 40.f : t % 17 == 0 ? .001f : 1.f;
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d) qg[((t * H + h) * 2) * D + d] = std::sin(float(t + h + d) * .13f);
        for (int i = 0; i < HK * D; ++i) {
            k[t * HK * D + i] = scale * std::sin(float(t * 7 + i) * .17f);
            v[t * HK * D + i] = scale * std::cos(float(t * 3 + i) * .11f);
        }
    }
    auto dqg = alloc<float>(qg.size()), dk = alloc<float>(k.size()), dv = alloc<float>(v.size()), dw = alloc<float>(D);
    put(dqg, qg); put(dk, k); put(dv, v); put(dw, w);
    auto q = alloc<float>(C * H * D), gate = alloc<float>(C * H * D);
    auto pos = alloc<int32_t>(1);
    auto fk = alloc<float>(values), fv = alloc<float>(values);
    auto qk = alloc<KvQ8Block>(values / kKvGroup), qv = alloc<KvQ8Block>(values / kKvGroup);
    k_attn_prep<false><<<dim3(H + HK, C), 256>>>(dqg, dk, dv, dw, dw, q, gate, fk, fv, H, HK, D, 64, pos, C, 1e7f, 1e-6f, nullptr, nullptr, {});
    k_attn_prep<true><<<dim3(H + HK, C), 256>>>(dqg, dk, dv, dw, dw, q, gate, qk, qv, H, HK, D, 64, pos, C, 1e7f, 1e-6f, nullptr, nullptr, {});
    ck(cudaDeviceSynchronize());
    auto hk = get(fk, values), hv = get(fv, values), hq = get(q, C * H * D);
    auto bk = get(qk, values / kKvGroup), bv = get(qv, values / kKvGroup);
    int errors = 0;
    std::vector<float> rk(values), rv(values);
    for (int kv = 0; kv < 2; ++kv) {
        const auto& f = kv ? hv : hk; const auto& b = kv ? bv : bk; auto& ref = kv ? rv : rk;
        for (size_t g = 0; g < b.size(); ++g) {
            float amax = 0.f;
            for (int j = 0; j < kKvGroup; ++j) amax = std::max(amax, std::fabs(f[g * kKvGroup + j]));
            const float s = __half2float(__float2half_rn(amax / 127.f));
            if (__half2float(b[g].scale) != s) ++errors;
            for (int j = 0; j < kKvGroup; ++j) {
                const int code = s > 0 ? std::clamp((int) std::nearbyint(f[g * kKvGroup + j] / s), -127, 127) : 0;
                if (b[g].codes[j] != code) ++errors;
                ref[g * kKvGroup + j] = code * s;
            }
        }
    }
    // P != capacity verifies that prompt gather uses the cache's head stride, not its output's stride.
    constexpr int P = 287;
    auto k16 = alloc<__half>(values), v16 = alloc<__half>(values);
    k_kv_f16<true><<<(HK * P * D + 255) / 256, 256>>>(qk, qv, k16, v16, C, P, D, HK);
    auto gk = get(k16, HK * P * D), gv = get(v16, HK * P * D);
    for (int h = 0; h < HK; ++h) for (int i = 0; i < P * D; ++i) {
        if (__half2float(gk[h * P * D + i]) != __half2float(__float2half_rn(rk[h * C * D + i]))) ++errors;
        if (__half2float(gv[h * P * D + i]) != __half2float(__float2half_rn(rv[h * C * D + i]))) ++errors;
    }
    // Distinct session pools, irregular cells, both sides of the 256-cell split boundary.
    auto qk2 = alloc<KvQ8Block>(bk.size()), qv2 = alloc<KvQ8Block>(bv.size());
    put(qk2, bk);
    for (auto& b : bv) b.scale = __float2half_rn(__half2float(b.scale) * .5f);
    put(qv2, bv);
    auto kt = alloc<void*>(2), vt = alloc<void*>(2);
    put(kt, std::vector<void*>{qk, qk2}); put(vt, std::vector<void*>{qv, qv2});
    std::vector<int32_t> cells{0, 63, 64, 255, 256, 299}, slots{0, 1, 0, 1, 0, 1};
    auto dc = alloc<int32_t>(R), ds = alloc<int32_t>(R); put(dc, cells); put(ds, slots);
    RowSel rs; rs.cell = dc; rs.slot = ds; rs.kc = kt; rs.vc = vt;
    auto pa = alloc<float>(R * H * NC * D), pm = alloc<float>(R * H * NC), pl = alloc<float>(R * H * NC);
    auto out = alloc<float>(R * H * D);
    cudaStream_t st; ck(cudaStreamCreate(&st));
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    k_attn_part<true><<<dim3(NC, R * H), 256, 0, st>>>(q, nullptr, nullptr, pos, 0, H, HK, C, 1.f / 16, NC, pa, pm, pl, rs);
    k_attn_merge<<<R * H, D, 0, st>>>(pa, pm, pl, gate, pos, 0, H, NC, out, rs);
    cudaGraph_t graph; cudaGraphExec_t exec;
    ck(cudaStreamEndCapture(st, &graph)); ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    ck(cudaGraphLaunch(exec, st)); ck(cudaStreamSynchronize(st));
    auto actual = get(out, R * H * D);
    double max_err = 0;
    for (int t = 0; t < R; ++t) for (int h = 0; h < H; ++h) {
        const int kh = h / (H / HK), n = cells[t] + 1;
        std::vector<double> scores(n); double mx = -INFINITY, sum = 0;
        for (int p = 0; p < n; ++p) {
            double dot = 0;
            for (int d = 0; d < D; ++d) dot += double(hq[(t * H + h) * D + d]) * rk[(kh * C + p) * D + d];
            mx = std::max(mx, scores[p] = dot / 16);
        }
        for (auto& s : scores) sum += (s = std::exp(s - mx));
        for (int d = 0; d < D; ++d) {
            double expected = 0;
            for (int p = 0; p < n; ++p) expected += scores[p] * rv[(kh * C + p) * D + d] * (slots[t] ? .5 : 1.);
            expected = expected / sum * .5;   // sigmoid(0) output gate
            const double e = std::fabs(actual[(t * H + h) * D + d] - expected);
            max_err = std::max(max_err, e);
            if (!std::isfinite(e) || e > 3e-4 * std::max(1., std::fabs(expected))) ++errors;
        }
    }
    std::printf("q36 Q8 KV: %d errors, attention max absolute error %.9g\n", errors, max_err);
    ck(cudaGraphExecDestroy(exec)); ck(cudaGraphDestroy(graph)); ck(cudaStreamDestroy(st));
    return errors ? 1 : 0;
}
