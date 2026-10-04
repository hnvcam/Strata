### 11. Missing Requirement: Activation Quantization Contract Mismatch

**The Proposal Claims:**
> "Reuse existing helpers... `iq_mmvq`... Activations are q8_1 (32 values, fp16 scale and fp16 sum), the llama.cpp CUDA contract."

**The Reality (Source Evidence):**
*   **`include/strata/kernels/cpu/expert.hpp` lines 53–63:** The comment explicitly states:
    > "The default legacy quantizer uses FP32 scales and half-away rounding. It is not the pinned ggml CPU Q8_0 contract... `hx[k] = scale[k] * sum[k]` is the weight-independent correction the Q2_0 identity needs..."
*   **`src/core/mtp.cpp` line 532:**
    ```cpp
    quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, (int64_t) T * N, cs);
    ```
    The current runtime uses `quantize_q8_0_scaled` for the *legacy* Q2_0 experts. This function likely produces a specific activation format (possibly Q8_0 with FP32 scales or a custom Q8_1 variant) that matches the legacy Q2_0 kernel's expectations.
*   **`include/strata/kernels/iq_kernels.hpp` line 7:**
    > "Activations are q8_1 (32 values, fp16 scale and fp16 sum), the llama.cpp CUDA contract."
*   **The Conflict:** The `native_expert_grouped` kernel (used for Q4_K/Q8_0 experts) expects **Q8_1** activations (FP16 scales/sums). The current MTP runtime generates activations using `quantize_q8_0_scaled` (likely FP32 scales or different rounding).
*   **Missing Requirement:** If we switch to `native_expert_grouped`, we **must** change the activation quantization step in `src/core/mtp.cpp` from `quantize_q8_0_scaled` to `quantize_q8_1_rows` (or equivalent).
*   **Impact:** This is not just a loader change; it is a **kernel interface change**. The `hit_xq_` buffer layout and size will change (Q8_1 blocks are 34 bytes per 32 values, same as Q8_0, but the scale/sum storage differs). The `moe_grouped_s2` kernel (legacy) and `native_expert_grouped` kernel (new) are **not** drop-in replacements for each other regarding activation input.

**Correction:**
The proposal must explicitly state that `src/core/mtp.cpp` line 532 must be replaced with a call to `quantize_q8_1_rows` (defined in `iq_kernels.hpp` line 23) when in "Native Mode". The buffer `hit_xq_` must be sized for Q8_1 blocks.

### 12. Missing Requirement: Shared Expert Kernel Dispatch

**The Proposal Claims:**
> "Update kernel calls (e.g., `iq_mmvq`) to pass the correct `ggml_type` for each tensor."

**The Reality (Source Evidence):**
*   **`src/core/mtp.cpp` lines 535–545:** The shared expert is executed via `shared_expert(...)`.
*   **`src/core/mtp.cpp` lines 536–538:** The `NativeSharedWeights` struct is populated with hardcoded `GGML_Q8_0`.
*   **The Conflict:** The `shared_expert` function likely dispatches to a specific kernel based on `nsw.gate_type`. If we pass `GGML_Q4_K`, does `shared_expert` support it?
*   **`include/strata/kernels/iq_kernels.hpp` line 26:** `iq_mmvq` supports Q4_K.
*   **Missing Requirement:** We must verify that the `shared_expert` wrapper in `src/core/mtp.cpp` (or its underlying implementation) correctly routes Q4_K weights to `iq_mmvq`. If `shared_expert` is hardcoded to use a Q8_0-specific path, it will crash or produce garbage.
*   **Correction:** The proposal must include a code review step for `shared_expert` implementation to ensure it accepts dynamic types. If it does not, the conversion tool must convert shared expert weights to Q8_0 (losing the Q4_K benefit for that specific tensor) or the runtime must be patched. Given the goal is "minimal change," converting shared expert weights to Q8_0 in the conversion tool is the safer, verified path.

### 13. Missing Requirement: `mtp_fetch.py` and Source Integrity

**The Proposal Claims:**
> "Read the external sidecar... using `tools/gguf_reader.py`."

**The Reality (Source Evidence):**
*   **`tools/mtp_fetch.py` lines 29–35:** The current workflow relies on a **pinned revision** and **SHA256 verification** of the BF16 checkpoint.
*   **Sidecar Specimen:** This is an *external* file (`mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`). It is **not** fetched by `mtp_fetch.py`.
*   **Missing Requirement:** The proposal ignores the **provenance** of the sidecar.
    *   How is the sidecar validated?
    *   Does it match the target model's architecture (`qwen4exp`)?
    *   Is it compatible with the *specific* IQ4_XS target model installed?
*   **Correction:** The proposal must include a **compatibility check** step in the conversion tool:
    1.  Read `general.architecture` from the sidecar. Verify it is `qwen4exp`.
    2.  Read `general.embedding_length`, `general.feed_forward_length`, `general.expert_count`. Verify they match the target model's config (H=2560, FF=640, NE=512).
    3.  If they do not match, **fail** the conversion. Do not attempt to load mismatched weights.

### 14. Missing Requirement: Draft Vocabulary Interaction

**The Proposal Claims:**
> "Full draft vocabulary contains all 248,320 target IDs."

**The Reality (Source Evidence):**
*   **`setup.py` lines 2435–2453:** `refresh_draft_vocab` copies a subset (`cjk`, `en`, `cyrillic`) to `rt/draft_vocab.bin`.
*   **`src/core/mtp.cpp` lines 156–159:** The runtime loads `draft_vocab.bin`.
*   **The Conflict:** The sidecar is a "shared sidecar". It does not contain the output projection (LM head). The LM head is supplied by the target model.
*   **Missing Requirement:** The proposal assumes the draft vocabulary logic remains unchanged. However, if the sidecar changes the *acceptance rate* or *distribution* of drafts, the optimal draft vocabulary subset might change.
*   **Correction:** While not a code change, the validation plan must include **re-evaluating the draft vocabulary choice** (cjk vs en vs cyrillic) after switching to Q4_K_M, as the acceptance profile may shift.

### 15. Final Verdict and Actionable Plan

**Verified Source Facts:**
1.  `mtp_rt.py` hardcodes Q2_0 blob layout and Q8_0/BF16 dense conversion.
2.  `mtp.cpp` hardcodes `GGML_Q8_0` for shared experts and uses `quantize_q8_0_scaled` for activations.
3.  `iq_kernels.hpp` supports Q4_K/Q8_0 via `native_expert_grouped` and `iq_mmvq`, but expects Q8_1 activations.
4.  `gguf_reader.py` can parse Q4_K/Q8_0 headers.
5.  Sidecar uses `blk.48.` prefix, fused `eh_proj`, and pre-offset norms.

**Untested Expectations (Risks):**
1.  `native_expert_grouped` CUDA kernel correctly handles mixed Q4_K (gate/up) and Q8_0 (down) types.
2.  `iq_mmvq` has optimized paths for Q4_K at `n_embd=2560`.
3.  `shared_expert` wrapper supports dynamic types.
4.  Splitting Q4_K `eh_proj` via block slicing preserves numerical correctness.

**Corrected Implementation Plan:**

1.  **Conversion Tool (`tools/mtp_rt_native.py`):**
    *   **Input:** Sidecar GGUF.
    *   **Validation:** Check architecture and dimensions against target model config.
    *   **Experts:** Extract Q4_K gate/up and Q8_0 down. Concatenate to `[Gate][Up][Down]` blob. Write `experts_native.bin`.
    *   **Dense:**
        *   Split `eh_proj` (Q4_K) into `fc_embedding` and `fc_hidden` (Q4_K) via block slicing.
        *   Convert F32 router to BF16.
        *   Convert Q4_K/Q6_K/Q5_0 hyper-connections to BF16.
        *   Convert Shared Expert weights to Q8_0 (to avoid `shared_expert` dispatch risks).
        *   Keep Attention weights as Q4_K (if `iq_mmvq` supports it) or convert to Q8_0 (safe fallback).
        *   **Skip `+1.0`** for norms.
        *   Write `dense_native.bin` with 256-byte alignment.
    *   **Index:** Write `dense.txt` with string kinds (`q4_k`, `q8_0`, `bf16`, `f32`).

2.  **Runtime Changes (`src/core/mtp.cpp`):**
    *   **Loader:** Detect `dense_native.bin`.
    *   **Parsing:** Support new `kind` strings.
    *   **Activations:** Replace `quantize_q8_0_scaled` with `quantize_q8_1_rows` in Native Mode.
    *   **Experts:** Replace `moe_grouped_s2` with `native_expert_grouped`.
    *   **Dense:** Replace hardcoded `q8()` calls with dynamic type dispatch to `iq_mmvq`.

3.  **Validation:**
    *   **Unit:** Verify `eh_proj` split.
    *   **Integration:** Compare logits against Q8_0-converted sidecar.
    *   **Performance:** Measure acceptance rate and VRAM.
    *   **Hardware:** Force drafter to 16 GB card.

**Conclusion:**
The initial proposal was **insufficient** because it underestimated the coupling between the loader, activation quantization, and kernel dispatch. The corrected plan requires **significant runtime changes** (activation format, kernel dispatch) and **careful conversion logic** (block slicing, type conversion) to safely integrate the Q4_K_M sidecar. It is not a simple "add option" task.
