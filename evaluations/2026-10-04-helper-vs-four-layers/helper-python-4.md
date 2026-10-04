# Review: Q4_K_M MTP Adaptation Proposal (Revised for Native Expert Retention)

This review corrects the previous proposal to strictly adhere to the requirement: **Routed experts must remain Q4_K (gate/up) and Q8_0 (down).** Re-quantization to Q2_0 is prohibited. Dense tensors may be converted.

## 1. Critical Corrections to Previous Proposal

### A. Expert Layout and Dispatch (The "Native" Path)
**Previous Error:** The previous proposal suggested converting experts or using a generic loader that might default to legacy paths.
**Correction:** We must explicitly use the **Native Expert Path** defined in `include/strata/kernels/iq_kernels.hpp`.
*   **Source Fact:** `native_expert_grouped` (line 59) accepts `NativeExpertLayout` with separate `gu_type` and `d_type`.
*   **Source Fact:** `native_expert_layout` (line 1423 in `iq_kernels.cu`) calculates offsets for `[Gate | Up | Down]`.
*   **Requirement:** The conversion tool must produce `experts_native.bin` in this exact layout.
*   **Runtime Change:** `src/core/mtp.cpp` must detect this mode and call `native_expert_grouped` instead of `moe_grouped_s2`.

### B. Activation Format Mismatch (Critical)
**Previous Error:** The previous proposal noted the mismatch but did not emphasize the *runtime* change required.
**Correction:** The legacy Q2_0 path uses `quantize_q8_0_scaled` (line 532 in `mtp.cpp`). The `native_expert_grouped` kernel expects **Q8_1** activations (FP16 scales/sums), as stated in `iq_kernels.hpp` line 7.
*   **Action:** In Native Mode, `src/core/mtp.cpp` must replace `quantize_q8_0_scaled` with `quantize_q8_1_rows` (line 23 in `iq_kernels.hpp`).
*   **Impact:** The buffer `hit_xq_` must be sized for Q8_1 blocks (34 bytes per 32 values), not the legacy format.

### C. Tensor Naming and Axis Correction
**Previous Error:** Assumed simple name mapping.
**Correction:**
*   **Sidecar Names:** `blk.48.ffn_gate_exps.weight`, `blk.48.ffn_up_exps.weight`, `blk.48.ffn_down_exps.weight`.
*   **Strata Expectation:** The runtime expects a single blob per expert.
*   **Axis Check:**
    *   Sidecar Gate/Up: `[2560, 640, 512]` (Innermost first: 2560 is inner? No, GGUF is innermost first. Shape `[2560, 640, 512]` means 512 experts, each 640x2560? Or 2560x640?
    *   **Specimen Fact:** "blk.48.ffn_gate_exps.weight... [2560,640,512]". In GGUF, shape is `[cols, rows, batch]`? No, GGUF shape is `[dim0, dim1, ...]` where dim0 is innermost.
    *   **Strata Fact:** `expert.hpp` line 49: `gu: (1280, 720) Q2_0 rows`. This implies Gate/Up are stored as rows of length `H` (2560) or `FF` (640)?
    *   **Re-evaluation:** `native_expert_layout` uses `n_embd` (2560) for Gate/Up rows. `iq_row_bytes(gu_type, n_embd)`.
    *   **Conversion:** The sidecar tensors are likely `[n_embd, n_ff, n_experts]` or similar. The conversion tool must reshape/transpose to match `native_expert_layout`'s expectation of contiguous rows of length `n_embd` for Gate/Up.

### D. Dense Weight Conversion Strategy
**Previous Error:** Suggested keeping Q4_K for dense if possible.
**Correction:** To minimize runtime changes, convert **all** dense tensors (Attention, Shared Expert, Router, Hyper-connections) to **Q8_0** or **BF16** in the conversion tool.
*   **Reason:** `src/core/mtp.cpp` currently hardcodes `GGML_Q8_0` for shared experts (line 536) and uses `bf16()` for router (line 526). Supporting dynamic types for *all* dense tensors requires significant loader refactoring. Converting dense to Q8_0/BF16 allows reusing the existing `q8()` and `bf16()` helpers with minimal changes.
*   **Exception:** If `iq_mmvq` is already used for dense in the current code, we can keep Q4_K. But `mtp.cpp` line 536 suggests hardcoded Q8_0. **Safest Path:** Convert dense to Q8_0/BF16.

## 2. Revised Implementation Plan

### 2.1 Conversion Tool (`tools/mtp_rt_native.py`)

1.  **Input:** `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`.
2.  **Experts (Native Retention):**
    *   Read `blk.48.ffn_gate_exps.weight` (Q4_K), `blk.48.ffn_up_exps.weight` (Q4_K), `blk.48.ffn_down_exps.weight` (Q8_0).
    *   For each expert `e`:
        *   Extract Q4_K blocks for Gate and Up.
        *   Extract Q8_0 blocks for Down.
        *   Concatenate: `Gate_Blocks || Up_Blocks || Down_Blocks`.
    *   Write to `experts_native.bin`.
3.  **Dense (Conversion):**
    *   **Attention/Shared Expert:** Convert Q4_K/Q6_K/Q8_0 to **Q8_0**.
    *   **Router/Hyper-connections:** Convert F32/Q4_K/Q5_0 to **BF16**.
    *   **Norms:** Keep F32. **Do not add +1.0** (sidecar is pre-offset).
    *   **`eh_proj`:** Split `[5120, 2560]` into `fc_embedding` and `fc_hidden`. Convert to Q8_0.
    *   Write to `dense_native.bin` with 256-byte alignment.
4.  **Index (`dense.txt`):**
    *   Use standard kinds: `q8_0`, `bf16`, `f32`.
    *   Add a marker line or filename convention to indicate "Native Experts".

### 2.2 Runtime Changes (`src/core/mtp.cpp`)

1.  **Detection:**
    *   Check for `experts_native.bin`. If present, set `native_experts = true`.
2.  **Expert Loading:**
    *   If `native_experts`:
        *   Calculate size using `native_expert_layout(12, 8, 2560, 640)`.
        *   Load `experts_native.bin`.
        *   **Do NOT** use `cpu::BLOB` size.
3.  **Activation Quantization:**
    *   If `native_experts`:
        *   Replace `quantize_q8_0_scaled` (line 532) with `quantize_q8_1_rows`.
        *   Ensure `hit_xq_` buffer is sized for Q8_1.
4.  **Expert Dispatch:**
    *   If `native_experts`:
        *   Replace `moe_grouped_s2` (line 533) with `native_expert_grouped`.
        *   Pass `NativeExpertLayout` with `gu_type=12` (Q4_K), `d_type=8` (Q8_0).
5.  **Dense Loading:**
    *   Since dense is converted to Q8_0/BF16, existing `q8()` and `bf16()` helpers work unchanged.

## 3. Compatibility Checks

1.  **Kernel Support:**
    *   Verify `native_expert_supported(12, 8, 2560, 640)` returns `true`.
    *   Verify `iq_mmvq` supports Q8_0 (it does, line 1246 in `iq_kernels.cu`).
2.  **Dimension Check:**
    *   `n_embd % 256 == 0` (2560 % 256 = 0). OK.
    *   `n_ff % 32 == 0` (640 % 32 = 0). OK.
3.  **Sidecar Validation:**
    *   Check `general.architecture == "qwen4exp"`.
    *   Check `general.expert_count == 512`.

## 4. Memory Estimates

*   **Experts:**
    *   Q4_K Gate/Up: $2 \times 2560 \times 640 \times \frac{4.5}{8} \approx 1.84$ MB/expert.
    *   Q8_0 Down: $2560 \times 640 \times \frac{8}{8} \approx 1.64$ MB/expert.
    *   Total per expert: ~3.48 MB.
    *   Total 512 experts: **~1.78 GB**.
*   **Dense (Converted to Q8_0/BF16):**
    *   Similar to current Q8_0/BF16 dense: **~0.2 GB**.
*   **Total MTP VRAM:** **~2.0 GB**.
*   **Impact:**
    *   **RTX 5070 Ti (16 GB):** Fits easily.
    *   **RTX 4060 (8 GB):** Tight. Main model + KV + 2 GB drafter may exceed 8 GB. **Recommendation:** Force drafter to 16 GB card.

## 5. Validation Plan

1.  **Unit Test:** Verify `native_expert_layout` byte offsets match the concatenated blob.
2.  **Integration Test:**
    *   Run MTP with Native Experts.
    *   Compare logits against a reference (llama.cpp) using the same sidecar.
    *   **Focus:** Check for NaNs or large deviations in expert outputs.
3.  **Acceptance Rate:** Measure draft acceptance. Q4_K_M should be higher than Q2_0.

## 6. Remaining Uncertainties

1.  **`native_expert_grouped` Mixed Type Support:** Does the CUDA kernel explicitly handle `gu_type != d_type`? Source shows struct support, but kernel body is not fully visible. **Action:** Inspect `iq_kernels.cu` kernel implementation.
2.  **Q8_1 Activation Overhead:** Does `quantize_q8_1_rows` have significant overhead compared to `quantize_q8_0_scaled`? **Action:** Profile.
3.  **Sidecar `eh_proj` Split:** Is the split along the correct axis? Specimen says "concatenating embedding input columns followed by hidden input columns". This implies splitting the **inner** dimension (2560) into two 2560 parts? No, `[5120, 2560]` means 5120 rows, 2560 cols. If it's `[out, in]`, then 5120 outputs. If it's `[in, out]`, then 5120 inputs.
    *   **Correction:** `fc_embedding` and `fc_hidden` are `[2560, 2560]`. `eh_proj` is `[5120, 2560]`. This implies `eh_proj` is `[out, in]`? No, 5120 is likely `2 * 2560`. So it's `[2*H, H]`. Splitting along dim 0 (rows) gives two `[2560, 2560]` tensors. This matches `fc_embedding` and `fc_hidden`.

## 7. Conclusion

This revised proposal strictly retains Q4_K/Q8_0 experts by leveraging the existing `native_expert_grouped` kernel. It simplifies dense weight handling by converting them to Q8_0/BF16, minimizing runtime changes. The primary risk is VRAM pressure on the 8 GB card and the need to verify mixed-type expert kernel support.
