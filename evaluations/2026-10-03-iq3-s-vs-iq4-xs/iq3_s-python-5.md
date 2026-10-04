# Final Proposal: Native Q4_K/Q8_0 MTP Expert Support with Legacy Q2 Retention

## 1. Executive Summary
To support the external `Q4_K_M` sidecar while retaining legacy `Q2_0` support, Strata requires a **dual-path MTP runtime**. The current hard-coded `Q2_0` blob loader in `src/core/mtp.cpp` must be extended to detect and load native GGUF expert blocks (`Q4_K` gate/up, `Q8_0` down) and dispatch them to the existing `native_expert_grouped` kernel. Dense weights will be converted to engine-supported formats (`Q8_0`/`BF16`) to minimize C++ changes.

## 2. Affected Files

### New Files
*   `tools/mtp_rt_native.py`: A converter that reads the sidecar GGUF, extracts raw expert blocks, and generates a native layout index.

### Modified Files
*   `src/core/mtp.cpp`:
    *   **Loader:** Add logic to detect `experts_native.txt` (or a flag in `dense.txt`) and load raw GGUF blocks into `experts_`.
    *   **Dispatch:** Switch between `moe_grouped_s2` (legacy Q2) and `native_expert_grouped` (native Q4_K/Q8_0) based on the loaded format.
    *   **Activation:** Ensure `Q8_1` activation quantization is used for the native path (required by `native_expert_grouped`).
*   `include/strata/core/mtp.hpp`:
    *   Add `strata::kernels::NativeExpertLayout native_layout_` member.
    *   Add `bool is_native_experts_` flag.

### Unchanged Files
*   `src/kernels/cuda/iq_kernels.cu`: `native_expert_grouped` already supports `Q4_K` (type 12) and `Q8_0` (type 8) as verified by `is_iq` (line 1269) and `native_expert_supported` (line 1417).
*   `include/strata/kernels/cpu/expert.hpp`: Legacy Q2 blob layout remains unchanged.

## 3. Converter Design (`tools/mtp_rt_native.py`)

This tool converts the sidecar into a format compatible with the new native runtime path.

### A. Expert Extraction (No Requantization)
*   **Input:** `blk.48.ffn_gate_exps.weight` (Q4_K), `blk.48.ffn_up_exps.weight` (Q4_K), `blk.48.ffn_down_exps.weight` (Q8_0).
*   **Process:**
    1.  **Read Raw Blocks:** Extract raw bytes from the GGUF tensors.
    2.  **Layout Transformation:** `native_expert_grouped` expects a contiguous blob per expert: `[Gate Rows | Up Rows | Down Rows]`.
        *   **Gate/Up:** Q4_K blocks. GGUF shape `[2560, 640, 512]` (H, FF, NE). The kernel expects rows of `n_embd` (2560). The raw GGUF blocks are already row-major for the innermost axis (H).
        *   **Down:** Q8_0 blocks. GGUF shape `[640, 2560, 512]` (FF, H, NE). The kernel expects rows of `n_ff` (640).
    3.  **Concatenate:** For each expert `e` (0..511), concatenate:
        *   `gate_blocks[e]`
        *   `up_blocks[e]`
        *   `down_blocks[e]`
    4.  **Output:** `experts_native.bin` (512 concatenated blobs).
    5.  **Index:** `experts_native.txt` containing:
        *   `gu_type=12` (Q4_K)
        *   `d_type=8` (Q8_0)
        *   `n_embd=2560`
        *   `n_ff=640`

### B. Dense Weight Conversion (Engine-Supported Formats)
*   **Input:** `blk.48.*` dense tensors.
*   **Process:**
    1.  **Split `eh_proj`:**
        *   Read `blk.48.nextn.eh_proj.weight` (Q4_K, `[5120, 2560]`).
        *   Split into `fc_embedding.weight` (first 2560 rows) and `fc_hidden.weight` (last 2560 rows).
        *   **Quantize:** Convert to `Q8_0` using `q8_0()` logic from `tools/mtp_rt.py` (lines 34-45).
    2.  **Attention/Shared Expert:**
        *   Rename `blk.48.self_attn.*` → `self_attn.*`.
        *   Rename `blk.48.mlp.shared_expert.*` → `mlp.shared_expert.*`.
        *   **Quantize:** Convert Q4_K/Q6_K to `Q8_0` for `dense.bin`.
    3.  **Hyper-Connections:**
        *   Rename `blk.48.hc_*` → `hc_*`.
        *   **Dequantize:** Convert Q4_K/Q6_K to `BF16` for `dense.bin`.
    4.  **Norms:**
        *   Read `blk.48.nextn.*norm`.
        *   **No Offset:** Do not add +1.0 (sidecar includes it). Store as F32.
    5.  **Output:** `dense.bin` and `dense.txt` (same format as current `mtp_rt.py`).

## 4. Runtime Design (`src/core/mtp.cpp`)

### A. Loader Logic
*   **Detection:** Check for `experts_native.txt` in `rt_dir`.
*   **Native Path:**
    *   Read `experts_native.txt` to get `gu_type`, `d_type`, `n_embd`, `n_ff`.
    *   Call `strata::kernels::native_expert_layout(gu_type, d_type, n_embd, n_ff)` to get `NativeExpertLayout L`.
    *   Read `experts_native.bin` into `experts_`.
    *   Set `is_native_experts_ = true`.
*   **Legacy Path:**
    *   Read `experts.bin` (Q2_0 blob) as currently done (lines 180-200).
    *   Set `is_native_experts_ = false`.

### B. Dispatch Logic
*   **Current:** `moe_grouped_s2(...)` (line 533).
*   **Proposed:**
    ```cpp
    if (is_native_experts_) {
        // Ensure activations are Q8_1
        quantize_q8_1_rows(mixed_, hit_xq_, T, N, cs); // New call
        native_expert_grouped(native_layout_, grp_ptr_, grp_start_, grp_counts_, 
                              hit_dst_, hit_slot_, T*K, T*K, hit_xq_, 
                              hit_scratch_, parts_, cs);
    } else {
        // Legacy Q2 path
        quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, T*N, cs);
        moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, 
                       T*K, T*K, hit_xq_, hit_xs_, hit_scratch_, parts_, cs);
    }
    ```

### C. Activation Format Correction
*   **Fact:** `native_expert_grouped` expects `Q8_1` activations (see `iq_kernels.hpp` line 7: "Activations are q8_1").
*   **Current:** `moe_grouped_s2` uses `hit_xq_` which is likely `Q8_0` (based on `quantize_q8_0_scaled` in line 532).
*   **Change:** The native path must use `quantize_q8_1_rows` (from `iq_kernels.hpp` line 23) to produce `Q8_1` blocks. The `hit_xq_` buffer size must be adjusted to accommodate `Q8_1` (34 bytes per 32 values vs 18 bytes for Q8_0).

## 5. Compatibility Checks

1.  **Kernel Support:**
    *   Verify `native_expert_supported(12, 8, 2560, 640)` returns true.
    *   `is_iq(12)` (Q4_K) is true (line 1269).
    *   `is_iq(8)` (Q8_0) is true (line 1269).
    *   `n_embd % 256 == 0` (2560 % 256 = 0).
    *   `n_ff % 32 == 0` (640 % 32 = 0).
2.  **Tensor Names:**
    *   Ensure `dense.txt` has `fc_embedding.weight`, `fc_hidden.weight`, `self_attn.q_proj.weight`, etc.
3.  **Norm Offsets:**
    *   Verify `nextn.enorm` values are ~1.0 (offset included) vs ~0.0 (offset not included).

## 6. Memory Implications

*   **Experts (Native Q4_K/Q8_0):**
    *   Gate/Up (Q4_K): 512 * 2 * (2560*640/256) * 144 bytes = 1.47 GB.
    *   Down (Q8_0): 512 * (640*2560/32) * 34 bytes = 0.87 GB.
    *   Total Experts: **~2.34 GB**.
*   **Dense (Q8_0/BF16):** ~0.2 GB.
*   **Total MTP VRAM:** **~2.5 GB**.
*   **Comparison:** Current Q2 MTP is ~0.9 GB. This is a **2.7x increase**.
*   **Impact:**
    *   **RTX 5070 Ti (16 GB):** Safe.
    *   **RTX 4060 (8 GB):** Risky. If the main model uses >5.5 GB, the MTP drafter will cause OOM. The `--split-skip-if-fits` logic (line 2139) may force the drafter to the 5070 Ti if the 4060 is too full.

## 7. Validation Plan

1.  **Static Check:**
    *   Run `tools/mtp_rt_native.py`.
    *   Verify `experts_native.bin` size is `512 * (921600*2 + 1740800)`.
2.  **Kernel Smoke Test:**
    *   Load with native flag.
    *   Check `native_expert_supported` log output.
3.  **Numerical Check:**
    *   Compare output logits of native Q4_K MTP vs. legacy Q2 MTP. Expect higher precision (lower error) for native.
4.  **Performance:**
    *   Measure decode speed. Native Q4_K may be slower due to larger memory bandwidth requirements.

## 8. Remaining Uncertainties

1.  **Activation Format Mismatch:** The current `moe_grouped_s2` path uses `hit_xq_` which might be Q8_0. `native_expert_grouped` requires Q8_1. **Critical:** If `moe_grouped_s2` uses Q8_0, the MTP runtime must be modified to use `quantize_q8_1_rows` for the native path. This is a significant change to the activation pipeline.
2.  **Scratch Memory:** `native_expert_scratch_bytes` (line 1437) calculates scratch size. Ensure the MTP arena (`arena_`) is large enough.
3.  **Kernel Optimization:** `native_expert_grouped` may not be as optimized for the specific `n_embd=2560, n_ff=640` dimensions as the custom `moe_grouped_s2`. Performance regression is possible.

## 9. Conclusion

To meet the requirement of retaining Q4_K/Q8_0 experts, **C++ changes are mandatory**. The proposal involves:
1.  A new conversion tool to extract raw GGUF blocks.
2.  Modifying `src/core/mtp.cpp` to detect native layout and call `native_expert_grouped`.
3.  Ensuring activation quantization uses Q8_1 for the native path.

This path is complex and carries performance risks but satisfies the precision requirement. The alternative (requantizing to Q2) is rejected per instructions.
