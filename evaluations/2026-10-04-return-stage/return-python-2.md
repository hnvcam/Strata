**Review: Skeptical Maintainer Assessment**

I have reviewed the proposal for integrating the Q4_K_M MTP sidecar. While the high-level direction (using native expert kernels) is technically sound, the proposal contains **critical factual errors regarding the current codebase’s capabilities** and **dangerous assumptions about memory and kernel dispatch** that would lead to immediate runtime failures or incorrect results.

Here are the specific corrections and missing requirements:

### 1. Critical Error: `mtp.cpp` Does Not Support Native Expert Kernels
**Assumption in Proposal:** "Modify `src/core/mtp.cpp`... Dispatch to `native_expert_grouped`... The kernels already exist."
**Source Fact:** `src/core/mtp.cpp` (lines 530-534) **only** calls `moe_group_resident` and `moe_grouped_s2`. These are legacy kernels hardcoded for the Q2_0 blob layout (`strata::kernels::cpu::BLOB`).
*   `moe_grouped_s2` expects the specific interleaved Q2_0 structure defined in `expert.hpp`.
*   There is **no code path** in `mtp.cpp` that calls `native_expert_grouped` (defined in `iq_kernels.hpp`).
*   **Correction:** The proposal underestimates the work. We cannot just "wire" it. We must refactor `mtp.cpp` to:
    1.  Detect the expert format.
    2.  If Native, allocate `NativeExpertLayout` structures.
    3.  Implement the call sequence for `native_expert_grouped`, which requires different argument preparation (group pointers, start indices) than the current `moe_group_resident` flow.
    4.  **Risk:** The current `mtp.cpp` logic for grouping experts (lines 530-531) is tightly coupled to the Q2_0 memory layout. Reusing this logic for Native layout is not a simple "if/else"; it requires a new grouping implementation or a significant abstraction layer.

### 2. Critical Error: Dense Tensor Kernel Dispatch is Missing
**Assumption in Proposal:** "Update the forward pass... to dispatch to `iq_mmvq` for Q4_K tensors."
**Source Fact:** `src/core/mtp.cpp` (lines 526, 543) uses `bf16_gemv_fp32_mmvf` and `shared_expert` with hardcoded `GGML_Q8_0` types (lines 536-538).
*   The `shared_expert` call explicitly sets `nsw.gate_type = GGML_Q8_0`.
*   There is **no dispatch logic** in `mtp.cpp` to handle `q4_k` or `q6_k` for dense tensors. The `iq_mmvq` kernel exists in `iq_kernels.cu`, but `mtp.cpp` does not know how to call it.
*   **Correction:** We must implement a new dense tensor execution path in `mtp.cpp` that:
    1.  Reads the `kind` from `dense.txt`.
    2.  Calls `iq_mmvq` with the correct `ggml_type` ID.
    3.  Handles the activation quantization (`q8_1`) required by `iq_mmvq` (see `iq_kernels.hpp` line 22). The current `mtp.cpp` uses `quantize_q8_0_scaled` (line 532), which produces Q8_0, not Q8_1. **Q8_0 and Q8_1 are different formats.** `iq_mmvq` expects Q8_1. This is a blocker.

### 3. Missing Requirement: Activation Quantization Mismatch
**Assumption in Proposal:** Implicitly assumes activations can be reused.
**Source Fact:**
*   `mtp.cpp` line 532: `quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, ...)` produces **Q8_0** activations.
*   `iq_kernels.hpp` line 22: `iq_mmvq` expects **q8_1** activations (`x_q8_1`).
*   `iq_kernels.hpp` line 23: `quantize_q8_1_rows` is the helper to produce Q8_1.
*   **Correction:** The proposal fails to address that the **entire activation pipeline** for the MTP layer must change if we use Native kernels. We cannot feed Q8_0 activations into `iq_mmvq`. We must replace `quantize_q8_0_scaled` with `quantize_q8_1_rows` for all dense and expert inputs when in Native mode. This affects `mixed_`, `hit_xq_`, and potentially the shared expert inputs.

### 4. Memory Implication Error: `kDrafterMib` is Hardcoded
**Assumption in Proposal:** "Update `generate.cpp` to increase `kDrafterMib`."
**Source Fact:** `src/program/generate.cpp` line 2299: `const int64_t kDrafterMib = 1000;` is a **compile-time constant**.
*   It is not a runtime configuration variable.
*   **Correction:** We cannot simply "update" it at runtime. We must either:
    1.  Make `kDrafterMib` a runtime variable derived from the loaded MTP format (requires plumbing through `generate.cpp`).
    2.  Increase the constant to the worst-case (e.g., 3000 MiB), which wastes VRAM on Q2_0 installs.
    3.  **Recommendation:** Option 1 is required for correctness. The proposal ignores the static nature of this constant.

### 5. Sidecar Specimen Analysis: "Shared" vs. "Full"
**Assumption in Proposal:** The sidecar is a drop-in replacement for the MTP layer.
**Source Fact:** The sidecar is described as "shared" and "the target model supplies the embedding and output projection."
*   `mtp.cpp` (lines 201-204) requires `fc_embedding.weight` and `fc_hidden.weight` to be present in `dense.txt` as Q8_0.
*   The sidecar spec lists `blk.48.nextn.eh_proj.weight` (Q4_K) but does **not** explicitly list `fc_embedding` or `fc_hidden` in the "Dense attention/shared-expert projections" list, though it mentions "Shared-expert matrices use ffn_gate/up/down_shexp".
*   **Missing Requirement:** We must verify if the sidecar contains `fc_embedding` and `fc_hidden`. If it does not, the converter must **synthesize** them from the main model's embedding table (which is not available in the sidecar file alone). The proposal assumes the sidecar is self-contained for the MTP layer, but "shared" implies dependencies on the main model's weights that are not in the sidecar file. **This is a major blocker.** If the sidecar lacks these tensors, the converter needs access to the main model's GGUF to extract them, or the runtime must be modified to fetch them from the main model's memory.

### 6. Norm Offset Double-Application Risk
**Assumption in Proposal:** "Skip this step for the sidecar, as the offset is pre-applied."
**Source Fact:** `tools/mtp_rt.py` line 86: `arr = (data.astype(np.float32) + 1.0)`.
*   The sidecar spec says: "source GGUF norm scales already include their Gemma +1 offset."
*   **Correction:** This is correct, but the proposal must ensure that the **new converter** does not apply the offset. However, `mtp.cpp` does not apply the offset; it reads the raw F32. The risk is in the **converter**. If the converter uses `gguf-py` to read the tensor, it gets the raw bytes. If the raw bytes are already +1, we must write them as-is. This is a simple flag in the converter, but it must be explicit.

### 7. Kernel Availability: `native_expert_grouped` is CUDA-Only
**Assumption in Proposal:** Use `native_expert_grouped`.
**Source Fact:** `include/strata/kernels/iq_kernels.hpp` is a CUDA header. `src/kernels/cuda/iq_kernels.cu` implements it.
*   **Missing Requirement:** What about CPU fallback? The current Q2_0 path has a CPU implementation (`expert.hpp`). The Native Q4_K path **does not** have a CPU implementation in the provided sources.
*   **Correction:** If the user runs on CPU (or if the GPU is unavailable), the Native path will fail. We must either:
    1.  Implement CPU kernels for Native Q4_K/Q8_0 experts (large effort).
    2.  **Disable** the Native sidecar support on CPU-only builds.
    3.  **Fallback** to dequantizing Native experts to BF16/F32 and using the generic dense path (slow, but works). The proposal does not address CPU compatibility.

### Revised Proposal Summary

1.  **Converter (`tools/mtp_rt_q4.py`):**
    *   Must handle **missing** `fc_embedding`/`fc_hidden` by either requiring the main model path or failing with a clear error.
    *   Must **not** apply +1.0 to norms.
    *   Must output `dense.txt` with correct `kind` strings (`q4_k`, `q6_k`, etc.).
    *   Must output `experts.bin` in `NativeExpertLayout` format.

2.  **Runtime (`src/core/mtp.cpp`):**
    *   **Major Refactor:** Add a `format_` member (Q2_0 vs Native).
    *   **Activation:** Switch from `quantize_q8_0_scaled` to `quantize_q8_1_rows` for Native mode.
    *   **Dense:** Implement dispatch to `iq_mmvq` for Q4_K/Q6_K tensors.
    *   **Experts:** Implement call to `native_expert_grouped` with correct argument marshalling.
    *   **CPU:** Add a guard to prevent Native mode on CPU, or implement a slow fallback.

3.  **Memory (`src/program/generate.cpp`):**
    *   Change `kDrafterMib` from a constant to a runtime variable calculated after MTP format detection.

4.  **Validation:**
    *   Test on GPU only.
    *   Verify that `fc_embedding`/`fc_hidden` are present in the sidecar or handled correctly.
    *   Verify Q8_1 activation quantization matches `iq_mmvq` expectations.

**Conclusion:** The original proposal was too optimistic about the existing code's flexibility. The "Native" path is not just a data format change; it requires a parallel execution pipeline in `mtp.cpp` that does not currently exist. The "shared" nature of the sidecar also introduces a dependency on the main model's weights that must be resolved.
