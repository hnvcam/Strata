Based on the provided source code and product facts, here is the proposal for integrating the external Q4_K_M MTP sidecar into Strata.

### 1. Current Incompatibility Analysis

The current Strata MTP runtime (`tools/mtp_rt.py`, `src/core/mtp.cpp`) is hard-coded to ingest a specific intermediate format derived from BF16 checkpoints, not arbitrary GGUFs. The external sidecar `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` is incompatible for three primary reasons:

**A. Expert Format and Layout Mismatch**
*   **Current State:** `tools/mtp_rt.py` expects experts in `gate_up_proj` and `down_proj` tensors quantized to **Q2_0** (Type 42). It explicitly asserts shapes `(NE, 2*FF, 720)` and `(NE, H, 180)` (lines 70-72). It converts these into a custom `experts.bin` blob layout defined in `include/strata/kernels/cpu/expert.hpp` (interleaved gate/up rows, planar codes/scales).
*   **Sidecar State:** The sidecar uses **Q4_K** (Type 12) for gate/up and **Q8_0** (Type 8) for down. The shapes are `[2560, 640, 512]` and `[640, 2560, 512]` (innermost axis first).
*   **Conflict:** The runtime loader in `src/core/mtp.cpp` (lines 180-200) reads `experts.bin` assuming a fixed size `g.n_expert * strata::kernels::cpu::BLOB` (where `BLOB` is derived from Q2_0 geometry). It cannot read Q4_K blocks. Furthermore, the GPU kernels in `src/kernels/cuda/iq_kernels.cu` for the *legacy* MTP path (`moe_grouped_s2`) are specialized for the Q2_0 blob layout, not native GGUF Q4_K blocks. While `native_expert_grouped` exists in `iq_kernels.hpp` (lines 59-61), the current MTP runtime does not invoke it; it invokes `moe_grouped_s2` (line 533 in `mtp.cpp`).

**B. Dense Tensor Quantization and Naming**
*   **Current State:** `tools/mtp_rt.py` converts specific dense tensors (attention, shared expert) to **Q8_0** (lines 29-31, 92-94) and others to **BF16** or **F32**. It writes a `dense.txt` index with kinds `q8_0`, `bf16`, `f32`. The runtime (`mtp.cpp` lines 201-204) strictly requires specific tensors to be `q8_0` (e.g., `self_attn.q_proj.weight`).
*   **Sidecar State:** The sidecar uses mixed quantization: **Q4_K**, **Q6_K**, **Q8_0**, **Q5_0**, and **F32**. For example, `ffn_gate_exps` are Q4_K, but `ffn_down_exps` are Q8_0. Hyper-connection up matrices are Q5_0.
*   **Conflict:** The runtime's `q8()` helper (implied by line 204) likely fails or produces incorrect results if it encounters a Q4_K tensor where it expects Q8_0. The `dense.bin` loader does not support dynamic dispatch for Q4_K/Q5_0/Q6_K in the current MTP path; it assumes the pre-processed Q8_0/BF16/F32 mix.

**C. Architecture and Tensor Naming**
*   **Current State:** The packer (`mtp_pack.py`) and runtime expect tensor names like `mtp.layers.0.mlp.experts.gate_up_proj`.
*   **Sidecar State:** The sidecar uses `blk.48.ffn_gate_exps.weight`, `blk.48.nextn.eh_proj.weight`, etc.
*   **Conflict:** The runtime looks for specific short names (e.g., `fc_embedding.weight`, `self_attn.q_proj.weight`) in `dense.txt`. The sidecar's naming convention (`blk.48.*`) does not match the expected keys in `mtp.cpp` (lines 201-203).

### 2. Proposed Conversion and Runtime Changes

To support the sidecar without breaking existing Q2 support, a **dual-path ingestion strategy** is required.

#### Phase 1: New Conversion Tool (`tools/mtp_rt_native.py`)
A new tool is needed to convert the external GGUF directly into Strata's runtime format, or a new runtime format. Given the complexity of Q4_K, **converting to the existing Q8_0/BF16/F32 `dense.bin` + a new `experts_native.bin`** is safer than modifying the core runtime to parse GGUF on the fly.

1.  **Expert Handling:**
    *   Read `blk.48.ffn_gate_exps.weight` (Q4_K) and `blk.48.ffn_down_exps.weight` (Q8_0).
    *   **Option A (Preferred for Performance):** Dequantize Q4_K/Q8_0 experts to BF16/FP32, then re-quantize to **Q2_0** using the existing `q2_0` logic in `mtp_pack.py`. This allows reusing the *entire* existing runtime (`mtp_rt.py` output format) and kernels (`moe_grouped_s2`). This is the lowest-risk path.
    *   **Option B (Native Q4_K):** If Q4_K accuracy is required, the runtime must be modified to use `native_expert_grouped` from `iq_kernels.hpp`. This requires:
        *   Creating a new `experts.bin` layout compatible with `NativeExpertLayout` (lines 41-47 in `iq_kernels.hpp`).
        *   Modifying `src/core/mtp.cpp` to detect the expert type and switch from `moe_grouped_s2` to `native_expert_grouped`.
        *   *Recommendation:* Start with Option A (Re-quantize to Q2_0) to validate the sidecar's structural integrity. Option B is a performance optimization for later.

2.  **Dense Tensor Handling:**
    *   Map sidecar names to Strata names:
        *   `blk.48.nextn.eh_proj.weight` -> `fc_embedding.weight` (Note: Sidecar shape `[5120, 2560]` vs Strata expectation. Strata expects `fc_embedding` and `fc_hidden` separately or combined? `mtp_rt.py` line 29 lists `fc_embedding.weight` and `fc_hidden.weight` as separate Q8_0 tensors. The sidecar has a single `eh_proj`. **Conversion must split this tensor** into embedding and hidden parts if the architecture allows, or modify the runtime to accept a combined projection. Given `eh_proj` concatenates embedding and hidden inputs, splitting is likely required to match `fc_embedding` and `fc_hidden` expectations).
        *   `blk.48.attn_q_proj.weight` -> `self_attn.q_proj.weight`.
        *   `blk.48.ffn_gate_shexp.weight` -> `mlp.shared_expert.gate_proj.weight`.
    *   **Quantization Normalization:**
        *   Convert all Q4_K/Q6_K/Q5_0 dense tensors to **Q8_0** (using `q8_0` function from `mtp_rt.py`) to satisfy the runtime's `q8()` checks.
        *   Keep F32 norms as F32.
        *   Keep BF16 tensors as BF16.
    *   **Norm Offset:** The sidecar norms already include the Gemma +1 offset. `mtp_rt.py` line 86 *adds* 1.0 to F32 norms. **The new converter must NOT add 1.0** if the source already has it, or it must subtract it first. This is a critical correctness check.

#### Phase 2: Runtime Modifications (`src/core/mtp.cpp`)
If Option A (Re-quantize to Q2_0) is chosen, **no runtime changes are needed**. The output of the new converter will be identical in format to `tools/mtp_rt.py`.

If Option B (Native Q4_K) is chosen:
1.  **Loader:** Modify `mtp.cpp` to read a metadata flag (e.g., `strata.mtp.expert_format`) from `dense.txt` or a new `meta.txt`.
2.  **Kernel Dispatch:**
    *   If format is `q2_0`, use `moe_grouped_s2` (current path).
    *   If format is `q4_k`, use `native_expert_grouped`.
    *   Update `native_expert_supported` checks (line 1417 in `iq_kernels.cu`) to ensure Q4_K/Q8_0 pairs are enabled for the MTP dimensions (H=2560, FF=640).
3.  **Memory:** `native_expert_grouped` requires scratch buffers (`native_expert_scratch_bytes`). Ensure `arena_` allocation in `mtp.cpp` accounts for this.

### 3. Reusable Helpers and Kernels

*   **`tools/mtp_pack.py::q2_0`**: Reuse this to re-quantize sidecar experts to Q2_0 if Option A is chosen.
*   **`tools/mtp_rt.py::q8_0`**: Reuse to convert sidecar dense tensors (Q4_K/Q6_K) to Q8_0.
*   **`tools/gguf_reader.py`**: Use this to parse the sidecar header and verify tensor types/shapes before conversion.
*   **`include/strata/kernels/iq_kernels.hpp`**: Contains `native_expert_grouped` and `iq_mmvq` which are essential if native Q4_K support is implemented.
*   **`src/core/mtp.cpp`**: The `dense.txt` parsing logic (lines 154-165) is robust and can be reused if the new converter outputs the same index format.

### 4. Compatibility Checks

1.  **Tensor Name Mapping:** Verify that every tensor required by `mtp.cpp` (lines 201-203) exists in the sidecar after renaming. Specifically, check if `eh_proj` can be split into `fc_embedding` and `fc_hidden` or if the runtime needs a patch to handle a combined input projection.
2.  **Norm Offset Verification:** Compare a sample norm weight from the sidecar (F32) with the expected value. If `sidecar_val == expected_val - 1.0`, the offset is missing. If `sidecar_val == expected_val`, the offset is present. The converter must adjust accordingly.
3.  **Shape Consistency:** Ensure `blk.48.ffn_gate_exps` (Q4_K) dequantizes to `[512, 2560, 640]` (Strata order) or `[512, 640, 2560]` (GGUF order). The converter must handle the innermost-axis-first GGUF convention correctly.
4.  **Vocabulary:** The sidecar is "shared," meaning it relies on the target model's embedding/output. Ensure the draft vocabulary (`draft_vocab.bin`) matches the target model's vocabulary subset. The sidecar itself contains no vocabulary.

### 5. Memory Implications

*   **Current Q2 MTP:** ~0.71 GB for experts (Q2_0) + ~0.18 GB for dense = ~0.9 GB VRAM.
*   **Sidecar Q4_K (Native):**
    *   Experts: Q4_K is ~4.5 bits/weight. Q2_0 is ~2.25 bits/weight. Expert size will roughly **double** to ~1.4 GB.
    *   Dense: Q4_K/Q6_K are larger than Q8_0? No, Q8_0 is 8 bits. Q4_K is ~4.5 bits. So dense tensors might be **smaller** if converted to Q4_K, but the current runtime forces Q8_0. If we keep Q8_0, size is similar.
    *   **Total VRAM:** Likely **1.2 - 1.5 GB** for the drafter.
*   **Impact on RTX 5070 Ti (16 GB):** The drafter is loaded on the last stage. `kDrafterMib` is currently 1000 MiB (line 2299 in `generate.cpp`). If the new drafter is 1.5 GB, this constant must be increased, or the layer split logic will underestimate VRAM usage, potentially causing OOM during cache sizing.
*   **Impact on RTX 4060 (8 GB):** If the drafter is on the 4060, 1.5 GB is a significant portion of 8 GB. The `--split-skip-if-fits` logic (line 2139) must account for the larger drafter size.

### 6. Validation Needed

1.  **Bitwise Accuracy (Option A):** If re-quantizing to Q2_0, compare the output `experts.bin` of the new converter against a reference `mtp_rt.py` run on the original BF16 checkpoint (if available). They should be identical if the source weights are identical.
2.  **Acceptance Rate:** Run the drafter on a standard prompt set. Compare the draft acceptance rate against the legacy Q2 MTP. Q4_K sidecar *should* have higher acceptance if the original Q2 was lossy, but re-quantizing Q4_K to Q2_0 might introduce double quantization error.
3.  **Memory Profile:** Monitor VRAM usage during startup. Ensure `kDrafterMib` is updated if the drafter size exceeds 1000 MiB.
4.  **Norm Correctness:** Generate a single token and compare logits with a reference implementation (e.g., llama.cpp) using the same sidecar. Discrepancies in the first token indicate norm or projection errors.

### 7. Uncertainties

*   **`eh_proj` Splitting:** It is unclear if `fc_embedding` and `fc_hidden` in Strata are strictly separate or if the runtime can handle a combined `[5120, 2560]` projection. If they are separate, the converter must split the sidecar's `eh_proj` tensor.
*   **Norm Offset:** The exact state of the Gemma +1 offset in the sidecar's F32 norms is assumed based on the prompt ("source GGUF norm scales already include their Gemma +1 offset"). This must be verified empirically.
*   **Q4_K Kernel Performance:** If native Q4_K is used, the performance of `native_expert_grouped` for the specific MTP dimensions (H=2560, FF=640) on the RTX 5070 Ti is unknown. It may be slower than the highly optimized Q2_0 `moe_grouped_s2` kernel.

### Conclusion

The most robust path is to **convert the sidecar to the existing Q2_0 format** using a new tool that reuses `mtp_pack.py`'s quantizers. This requires no runtime changes, preserves memory profiles, and leverages existing validated kernels. Native Q4_K support is a larger engineering effort requiring runtime kernel dispatch changes and VRAM budget adjustments.
