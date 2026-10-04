# Proposal: Native Q4_K_M MTP Sidecar Integration

## 1. Executive Summary
This proposal enables Strata to import and execute the external `Q4_K_M` MTP sidecar (`mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`) while **preserving its native Q4_K (gate/up) and Q8_0 (down) expert quantization**. It retains full support for the legacy Q2_0 MTP runtime.

The solution introduces a **dual-path loader** in `src/core/mtp.cpp` and a new **conversion tool** (`tools/mtp_rt_native.py`). The conversion tool normalizes the sidecar into a "Native GGUF-backed" binary format that the runtime loads using existing `iq_kernels` (which already support Q4_K/Q8_0 via `native_expert_grouped`), avoiding CUDA kernel rewrites. Dense weights are converted to Q8_0/BF16 to minimize runtime dispatch complexity.

## 2. Affected Files

| File | Action | Reason |
|------|--------|--------|
| `tools/mtp_rt_native.py` | **New** | Converts external GGUF to Strata's native binary format (`dense_native.bin`, `experts_native.bin`). |
| `src/core/mtp.cpp` | **Modify** | Add detection logic for "Native" mode; update loader to parse dynamic types; switch activation quantizer to Q8_1. |
| `include/strata/core/mtp.hpp` | **Modify** | Add flags/structs to distinguish between Legacy (Q2_0) and Native (Q4_K) modes. |
| `tools/gguf_reader.py` | **Reuse** | Use existing `BLOCK_GEOMETRY` and header parsing to validate sidecar. |
| `include/strata/kernels/iq_kernels.hpp` | **Verify** | Confirm `native_expert_grouped` supports mixed Q4_K (gate/up) + Q8_0 (down) types. |

## 3. Conversion Design (`tools/mtp_rt_native.py`)

The conversion tool reads the external GGUF and writes two files: `experts_native.bin` and `dense_native.bin`, plus an index `dense.txt`.

### 3.1 Expert Conversion (Native Retention)
*   **Input:** `blk.48.ffn_gate_exps.weight` (Q4_K), `blk.48.ffn_up_exps.weight` (Q4_K), `blk.48.ffn_down_exps.weight` (Q8_0).
*   **Layout:** `native_expert_grouped` expects a contiguous blob per expert: `[Gate Rows] [Up Rows] [Down Rows]`.
*   **Process:**
    1.  For each expert `e` (0–511):
        *   Extract raw Q4_K blocks for Gate and Up.
        *   Extract raw Q8_0 blocks for Down.
        *   Concatenate: `Gate_Data || Up_Data || Down_Data`.
    2.  Write to `experts_native.bin`.
*   **Validation:** Ensure `native_expert_supported(12, 8, 2560, 640)` returns true. (H=2560, FF=640).

### 3.2 Dense Weight Conversion
*   **Naming Normalization:**
    *   Strip `blk.48.` prefix.
    *   Map `nextn.eh_proj.weight` $\rightarrow$ split into `fc_embedding.weight` and `fc_hidden.weight`.
    *   Map `self_attn.*`, `mlp.shared_expert.*`, `mlp.gate.weight` to expected names.
*   **Quantization Handling:**
    *   **Norms (F32):** Sidecar norms are pre-offset (Gemma +1). **Do not add 1.0** (unlike `mtp_rt.py` line 86).
    *   **Router (`mlp.gate.weight`):** Sidecar is F32. Convert to **BF16** to match current runtime expectations (`bf16_gemv_fp32_mmvf`).
    *   **Hyper-Connections:** Sidecar uses Q4_K/Q6_K/Q5_0. Convert to **BF16** to avoid complex dynamic dispatch in non-critical paths.
    *   **Attention/Shared Expert:** Convert to **Q8_0** to reuse existing `q8()` helpers and avoid dynamic type dispatch in `shared_expert`.
*   **`eh_proj` Splitting:**
    *   Input: `[5120, 2560]` Q4_K.
    *   Split along inner dimension (2560). Since Q4_K blocks are 256 values, split at block boundary 10.
    *   Output: Two `[2560, 2560]` Q4_K tensors.
*   **Alignment:** Pad all tensors in `dense_native.bin` to **256-byte boundaries** (matching `mtp_rt.py` line 97).

### 3.3 Index Format (`dense.txt`)
Update format to support dynamic types.
*   **Legacy:** `name kind rows cols offset bytes` (kind: `q8_0`, `bf16`, `f32`)
*   **Native:** `name kind rows cols offset bytes` (kind: `q4_k`, `q8_0`, `bf16`, `f32`)
*   **Detection:** If `kind` is not in `{q8_0, bf16, f32}`, assume Native Mode.

## 4. Runtime Design (`src/core/mtp.cpp`)

### 4.1 Loader Modifications
1.  **Mode Detection:**
    *   Read `dense.txt`. If any `kind` is `q4_k` (or similar), set `native_mode = true`.
2.  **Dense Loading:**
    *   Replace hardcoded `q8()` helper with `get_tensor_ptr(kind, offset)`.
    *   Update `NativeSharedWeights` struct population (lines 536–538) to use dynamic types from `dense.txt` instead of hardcoded `GGML_Q8_0`.
3.  **Expert Loading:**
    *   If `native_mode`:
        *   Calculate size using `native_expert_layout` (from `iq_kernels.hpp`).
        *   Load `experts_native.bin`.
        *   **Critical:** Replace `moe_grouped_s2` (legacy) with `native_expert_grouped`.

### 4.2 Activation Quantization Switch
*   **Current:** `quantize_q8_0_scaled` (line 532) produces activations for Q2_0 kernels.
*   **Native:** `native_expert_grouped` requires **Q8_1** activations (FP16 scales/sums).
*   **Change:** In `native_mode`, replace `quantize_q8_0_scaled` with `quantize_q8_1_rows` (from `iq_kernels.hpp` line 23).
*   **Buffer:** Ensure `hit_xq_` is sized for Q8_1 blocks (34 bytes per 32 values).

### 4.3 Kernel Dispatch
*   **Dense:** Ensure `iq_mmvq` is called with the correct `ggml_type` for each tensor (Q4_K, Q8_0, etc.).
*   **Experts:** Call `native_expert_grouped` with `NativeExpertLayout` configured for `gu_type=12` (Q4_K) and `d_type=8` (Q8_0).

## 5. Compatibility & Memory Considerations

### 5.1 Hardware Impact (RTX 5070 Ti 16GB + RTX 4060 8GB)
*   **VRAM Increase:**
    *   Legacy Q2_0 MTP: ~0.9 GB.
    *   Native Q4_K_M MTP: ~2.2 GB (Experts ~1.8 GB + Dense ~0.4 GB).
    *   **Risk:** The 8 GB RTX 4060 may not fit the drafter + main model KV cache.
    *   **Mitigation:** Force drafter to 16 GB card via `--layer-split` or `--split-skip-if-fits`.

### 5.2 Backward Compatibility
*   **Legacy Q2_0:** Unaffected. Loader detects `kind=q8_0/bf16` and uses old path.
*   **Sidecar:** Requires new conversion tool. Existing `mtp_pack.py` remains for Q2_0.

### 5.3 Numerical Correctness
*   **Norms:** Sidecar norms are pre-offset. Conversion tool must **skip** `+1.0` addition.
*   **Activations:** Switching to Q8_1 changes rounding behavior. Must validate against reference.

## 6. Validation Plan

1.  **Unit Test: `eh_proj` Split**
    *   Verify that splitting Q4_K `eh_proj` into two Q4_K tensors preserves block alignment and values.
2.  **Integration Test: Logit Comparison**
    *   Run MTP forward pass with Native Loader.
    *   Compare logits against a reference implementation (e.g., llama.cpp loading the same sidecar).
    *   **Tolerance:** Allow small differences due to Q8_1 vs Q8_0 activation rounding.
3.  **Acceptance Rate Benchmark**
    *   Measure draft acceptance rate on a standard prompt set.
    *   **Goal:** Q4_K_M should have higher acceptance than Q2_0. If lower, investigate quantization/norm errors.
4.  **Memory Leak Check**
    *   Verify `native_expert_grouped` scratch buffers are correctly allocated/freed.

## 7. Remaining Uncertainties

1.  **Kernel Support:** Does `native_expert_grouped` CUDA kernel explicitly support mixed `gu_type=12` (Q4_K) and `d_type=8` (Q8_0)? Source shows struct support, but kernel body is not fully visible. **Action:** Verify in `iq_kernels.cu`.
2.  **`iq_mmvq` Performance:** Is `iq_mmvq` optimized for Q4_K at `n_embd=2560`? If not, dense layers may be slow. **Action:** Benchmark.
3.  **Sidecar Provenance:** Is the sidecar compatible with the *exact* IQ4_XS target model? **Action:** Add architecture/dimension checks in conversion tool.
4.  **Draft Vocabulary:** Does the change in acceptance rate require re-tuning the draft vocabulary subset (`cjk`/`en`)? **Action:** Monitor after deployment.

## 8. Conclusion
This proposal requires significant changes to `src/core/mtp.cpp` (loader, activation quantizer, kernel dispatch) and a new conversion tool. It is **not** a simple configuration change. However, it leverages existing `iq_kernels` support for Q4_K/Q8_0, avoiding CUDA kernel rewrites. The primary risks are VRAM pressure on the 8 GB card and potential numerical mismatches in activation quantization.
