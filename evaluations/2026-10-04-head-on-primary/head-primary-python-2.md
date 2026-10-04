This is a review of the preceding proposal from the perspective of a Strata maintainer. The proposal contains **critical architectural misunderstandings** regarding the current MTP runtime's coupling to the Q2_0 format and the specific hardware constraints of the target PC. It also proposes changes that are not supported by the provided source code excerpts.

### 1. Critical Mistake: Misunderstanding the "Native Expert" Path in MTP

**Proposal Claim:** *"The sidecar's Q4_K experts are compatible with Strata's Native Expert kernels... Modify `src/core/mtp.cpp` to... Use `native_expert_grouped` for experts."*

**Source Fact:**
*   `src/core/mtp.cpp` lines 530–534 explicitly call `moe_group_resident` and `moe_grouped_s2`.
*   `moe_grouped_s2` is a **specialized kernel** for the **Q2_0** format (see `include/strata/kernels/cpu/expert.hpp` lines 35–45, which defines `BLOB` specifically for Q2_0).
*   `native_expert_grouped` (defined in `include/strata/kernels/iq_kernels.hpp` line 59) is used for the **main model's** experts, not the MTP drafter.
*   The MTP drafter (`src/core/mtp.cpp`) does **not** currently link to or call `native_expert_grouped`. It uses a custom, highly optimized path for Q2_0 that assumes the specific interleaved layout and scale/code separation defined in `expert.hpp`.

**Correction:**
You cannot simply "switch" the MTP loader to use `native_expert_grouped` without rewriting the entire MTP forward pass in `src/core/mtp.cpp`. The current MTP code is hard-coded to:
1.  Allocate memory based on `strata::kernels::cpu::BLOB` (Q2_0 size).
2.  Call `moe_group_resident` which expects the Q2_0 blob layout.
3.  Call `moe_grouped_s2` which expects Q2_0 scales/codes.

**Required Change:**
To support Q4_K, you must either:
*   **Option A (Heavy):** Rewrite `src/core/mtp.cpp` to use `native_expert_grouped` and `iq_mmvq` for dense layers, effectively replacing the custom MTP kernel path with the generic IQ kernel path. This loses the specific Q2_0 optimizations (VNNI, etc.) and may be slower.
*   **Option B (Conversion):** Convert Q4_K experts **back** to Q2_0 during the `mtp_rt.py` step. This is lossy and defeats the purpose of using a higher-precision sidecar.
*   **Option C (New Kernel):** Write a new `moe_grouped_s2_q4k` kernel. This is a massive engineering effort not supported by the current codebase structure.

**Verdict:** The proposal underestimates the work required. The MTP runtime is **not** modular enough to swap expert formats easily.

### 2. Critical Mistake: Dense Tensor Quantization Support

**Proposal Claim:** *"Modify `src/core/mtp.cpp` to... Support native quantization types for dense tensors... `iq_mmvq` supports Q4_K/Q5_K/Q8_0 for dense matrix-vector products."*

**Source Fact:**
*   `src/core/mtp.cpp` lines 201–204 hard-check for `q8_0` types:
    ```cpp
    const char* required[] = {"fc_embedding.weight", ...};
    for (const char* n : required) if (!q8(n)) { err = std::string("mtp: ") + n + " is missing (q8_0)"; return false; }
    ```
*   The `q8()` helper (implied) likely checks for `kind == "q8_0"`.
*   The MTP forward pass (lines 522–553) uses `bf16_gemv_fp32_mmvf` and `shared_expert` with hardcoded `GGML_Q8_0` types (line 536).
*   There is **no** call to `iq_mmvq` in `src/core/mtp.cpp`. The MTP path uses different kernels (`bf16_gemv_fp32_mmvf`, `shared_expert`) that expect BF16 or Q8_0.

**Correction:**
The MTP runtime **does not** support Q4_K/Q6_K/Q5_0 for dense tensors. To support them, you must:
1.  Replace `bf16_gemv_fp32_mmvf` calls with `iq_mmvq` calls.
2.  Replace `shared_expert` calls with a version that accepts native types.
3.  Update the loader to parse `q4_k`, `q6_k`, etc., from `dense.txt`.

**Verdict:** The proposal assumes the MTP runtime is generic. It is not. It is a specialized, optimized path for Q2_0/Q8_0/BF16.

### 3. Missing Requirement: Hyper-Connection (HC) Tensor Handling

**Proposal Claim:** *"The sidecar uses Q4_K/Q6_K for HC down/inject and Q5_0 for up. The current runtime likely expects BF16 or Q8_0. The loader must support these types for HC tensors."*

**Source Fact:**
*   `tools/mtp_rt.py` lines 92–96:
    ```python
    if short in Q8:
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        raw, kind = q8_0(f32), "q8_0"
    else:
        raw, kind = u16.tobytes(), "bf16"
    ```
*   The current converter **only** outputs `q8_0` or `bf16`. It does not support Q4_K/Q5_0.
*   `src/core/mtp.cpp` lines 258–259 allocate buffers for HC tensors (`inj_`, `inj2_`, `lo_`, `rs_`, `bo_`). These are likely FP32 or BF16.
*   The MTP forward pass (lines 550–551) uses `gr_write` and other kernels that expect specific formats.

**Correction:**
If the sidecar provides HC tensors in Q4_K/Q5_0, the converter must **dequantize** them to BF16 or FP32, OR the runtime must be extended to support these types. Given the complexity of HC kernels, dequantizing to BF16 in the converter is the safer, less invasive path.

**Verdict:** The proposal ignores the fact that the current converter **cannot** output Q4_K/Q5_0 for HC tensors. It must either dequantize them or the runtime must be rewritten.

### 4. Memory Implications: Underestimation of VRAM Overhead

**Proposal Claim:** *"Total VRAM: ~0.8–0.9 GB... well within the 8 GB limit."*

**Source Fact:**
*   `src/core/mtp.cpp` lines 168–178: Allocates `dense_` buffer.
*   `src/core/mtp.cpp` lines 180–200: Allocates `experts_` buffer.
*   `src/core/mtp.cpp` lines 206–222: Allocates `state_arena_` for K/V.
*   `src/core/mtp.cpp` lines 258–284: Allocates `arena_` for intermediate buffers (`mixed_`, `inj_`, `qfull_`, etc.).
*   `src/program/generate.cpp` line 2299: `const int64_t kDrafterMib = 1000; // the MTP drafter (839 MiB) + the head`.

**Correction:**
The current Q2_0 drafter uses ~839 MiB.
*   Q4_K experts are larger than Q2_0.
*   Q4_K dense tensors are larger than Q8_0/BF16.
*   Intermediate buffers (`arena_`) are allocated based on `max_t` and model dimensions, not quantization. They remain large.
*   The **total** VRAM for the drafter will likely exceed 1 GB, possibly approaching 1.5–2 GB depending on `max_t` and context window.
*   On an RTX 4060 (8 GB), if the main model is split, the drafter must fit alongside the last layer's KV cache and the head. 2 GB is significant but likely manageable. However, the proposal's estimate of 0.9 GB is **too optimistic** because it ignores the `arena_` and `state_arena_` overheads which are substantial.

**Verdict:** The memory estimate is incomplete. It must account for the full `arena_` and `state_arena_` allocations, not just weights.

### 5. Missing Requirement: Norm Offset Handling

**Proposal Claim:** *"The sidecar norms have +1 offset... The new converter must **not** add +1.0."*

**Source Fact:**
*   `tools/mtp_rt.py` line 86: `arr = (data.astype(np.float32) + 1.0)`.
*   Product facts: *"source GGUF norm scales already include their Gemma +1 offset."*

**Correction:**
This is correct. The converter must **skip** the `+1.0` addition. However, the proposal does not mention that the **runtime** (`src/core/mtp.cpp`) might also expect the offset to be applied.
*   `src/core/mtp.cpp` does not show norm application logic in the excerpt.
*   If the runtime expects raw weights and applies `+1` internally, and the sidecar already has `+1`, we get `+2`.
*   If the runtime expects pre-offset weights, and the sidecar has `+1`, we are correct.
*   **Verification Needed:** Check `src/core/mtp.cpp` for `rms_norm` or similar calls to see if they add `1.0`.

**Verdict:** The proposal correctly identifies the converter change but fails to verify the runtime's expectation.

### 6. Missing Requirement: Tensor Name Mapping

**Proposal Claim:** *"Create a mapping from sidecar names (`blk.48.attn_q_norm`) to runtime names (`self_attn.q_norm`)."*

**Source Fact:**
*   `tools/mtp_rt.py` lines 82–83:
    ```python
    short = name[len("mtp."):]
    short = short[len("layers.0."):] if short.startswith("layers.0.") else short
    ```
*   The sidecar uses `blk.48.*`. The current converter expects `mtp.layers.0.*`.

**Correction:**
The converter must handle `blk.48.*` names. This is a simple string replacement, but it must be done correctly for all tensors (attention, experts, shared expert, HC, router).

**Verdict:** This is a minor implementation detail, but critical for the converter to work.

### 7. Untested Expectation: Kernel Performance

**Proposal Claim:** *"Verify `native_expert_supported` returns true for `gu_type=Q4_K`... Verify `iq_mmvq` supports Q4_K."*

**Source Fact:**
*   `include/strata/kernels/iq_kernels.hpp` line 2: *"Q4_K / Q5_K / Q5_1 / Q8_0 for Unsloth's UD-Q4_K_XL"*.
*   `src/kernels/cuda/iq_kernels.cu` line 1244: `case 13: dq_q5_k...` (Q5_K).
*   `src/kernels/cuda/iq_kernels.cu` line 1246: `case 8: dq_q8_0...`.
*   **Missing:** There is no explicit `case 12: dq_q4_k` in the `dq_dispatch` excerpt (lines 1244–1249). It might be elsewhere, but the excerpt does not confirm Q4_K dequantization is implemented in the same way.
*   `native_expert_supported` (line 1417) checks `is_iq(gu_type)`. `is_iq` (line 1268) includes `t == 12` (Q4_K). So it is supported.

**Correction:**
While Q4_K is supported by the IQ kernels, the **MTP runtime** does not use these kernels. The proposal assumes we can just plug them in. We cannot. We must rewrite the MTP runtime to use them.

**Verdict:** The kernel support exists, but the integration path does not.

### Revised Proposal Summary

1.  **Abandon the idea of "just switching" to native experts.** The MTP runtime is a specialized Q2_0 engine.
2.  **Option 1 (Recommended):** Write a new MTP runtime path (`mtp_native.cpp`) that uses `native_expert_grouped` and `iq_mmvq`. This is a large development effort.
3.  **Option 2 (Quick & Dirty):** Convert the Q4_K sidecar **back** to Q2_0 in `mtp_rt.py`. This is lossy but requires no runtime changes.
4.  **Option 3 (Hybrid):** Convert Q4_K experts to Q2_0, but keep dense tensors in Q8_0/BF16. This is what the current system does. The sidecar's Q4_K experts are not directly usable without conversion.

**Conclusion:** The preceding proposal is **invalid** because it assumes the MTP runtime is modular and supports native IQ formats. It is not. The MTP runtime is hard-coded to Q2_0. To use the Q4_K sidecar, you must either convert the sidecar to Q2_0 (lossy) or rewrite the MTP runtime to use the generic IQ kernels (heavy).
