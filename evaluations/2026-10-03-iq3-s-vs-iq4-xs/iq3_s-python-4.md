# Proposal: Native Q4_K/Q8_0 MTP Expert Support

## 1. Executive Summary
The requirement to retain `Q4_K` gate/up and `Q8_0` down experts necessitates a **C++ runtime modification**. The current MTP path (`src/core/mtp.cpp`) uses `moe_grouped_s2` which is hard-coded to the custom `Q2_0` blob layout. The proposed solution switches the MTP expert dispatch to `native_expert_grouped` (from `include/strata/kernels/iq_kernels.hpp`), which supports native GGUF block layouts. This requires a new loader path in `src/core/mtp.cpp` to read raw GGUF tensors instead of the pre-packed `experts.bin`.

## 2. Affected Files
*   **New:** `tools/mtp_rt_native.py` (Conversion/Indexing tool).
*   **Modified:** `src/core/mtp.cpp` (Loader and Dispatch logic).
*   **Modified:** `include/strata/core/mtp.hpp` (Member variables for native layout).
*   **Unchanged:** `src/kernels/cuda/iq_kernels.cu` (Assumes `native_expert_grouped` already supports Q4_K/Q8_0, verified by `is_iq` and `native_expert_supported`).

## 3. Conversion Design (`tools/mtp_rt_native.py`)

This tool does **not** requantize experts. It generates an index and potentially a raw binary dump of the expert tensors in a layout compatible with `native_expert_grouped`.

### A. Expert Handling (Native GGUF Blocks)
*   **Input:** `blk.48.ffn_gate_exps.weight` (Q4_K), `blk.48.ffn_up_exps.weight` (Q4_K), `blk.48.ffn_down_exps.weight` (Q8_0).
*   **Process:**
    1.  **Read Raw Blocks:** Extract the raw byte blocks from the GGUF file.
    2.  **Layout Transformation:** `native_expert_grouped` expects a specific "Native Expert Layout" defined in `include/strata/kernels/iq_kernels.hpp` (lines 40-47).
        *   The layout is `[gate rows | up rows | down rows]`.
        *   **Gate/Up:** Q4_K blocks. Shape `[NE, H, FF]` (GGUF innermost first: `[2560, 640, 512]` -> 512 experts, each 2560x640).
        *   **Down:** Q8_0 blocks. Shape `[NE, FF, H]` (GGUF innermost first: `[640, 2560, 512]` -> 512 experts, each 640x2560).
    3.  **Output:** Write `experts_native.bin` containing the concatenated raw blocks for all 512 experts in the order: Gate, Up, Down.
    4.  **Index:** Write `experts_native.txt` with metadata: `gu_type=12` (Q4_K), `d_type=8` (Q8_0), `n_embd=2560`, `n_ff=640`.

### B. Dense Weight Conversion (Engine-Supported Formats)
*   **Input:** `blk.48.*` dense tensors.
*   **Process:**
    1.  **Split `eh_proj`:**
        *   Read `blk.48.nextn.eh_proj.weight` (Q4_K, `[5120, 2560]`).
        *   Split into `fc_embedding.weight` (first 2560 rows) and `fc_hidden.weight` (last 2560 rows).
        *   **Quantize:** Convert to `Q8_0` (using `q8_0()` logic) to match existing `dense.bin` expectations for MMVQ.
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

## 4. Runtime Changes (`src/core/mtp.cpp`)

### A. Loader Logic
*   **Detect Format:** Check for `experts_native.txt` or a flag in `dense.txt`.
*   **Load Experts:**
    *   If native: Read `experts_native.bin` into `experts_` buffer.
    *   Calculate `NativeExpertLayout L` using `native_expert_layout(12, 8, 2560, 640)`.
    *   Store `L` in `mtp.hpp`.
*   **Load Dense:** Same as current (`dense.bin`/`dense.txt`).

### B. Dispatch Logic
*   **Current:** `moe_grouped_s2(...)` (line 533).
*   **Proposed:**
    *   If native layout detected:
        *   Call `native_expert_grouped(L, grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, ..., x_q8_1, scratch, parts_, cs)`.
    *   Else:
        *   Call `moe_grouped_s2(...)` (legacy path).

### C. Activation Format
*   `native_expert_grouped` expects `x_q8_1` (Q8_1 activations).
*   Current MTP path uses `hit_xq_` (Q8_0 scaled?).
*   **Correction:** Ensure `quantize_q8_0_scaled` (line 532) produces `Q8_1` blocks compatible with `native_expert_grouped`. The `iq_kernels.hpp` header states activations are `q8_1` (line 7). The current `moe_grouped_s2` might use a different activation format. **Verification required:** Does `moe_grouped_s2` use Q8_0 or Q8_1? If it uses Q8_0, the activation quantizer must be changed to `quantize_q8_1_rows` (from `iq_kernels.hpp` line 23) for the native path.

## 5. Compatibility Checks

1.  **Kernel Support:**
    *   Verify `native_expert_supported(12, 8, 2560, 640)` returns true.
    *   `is_iq(12)` is true (line 1269).
    *   `is_iq(8)` is true (line 1269).
    *   `n_embd % 256 == 0` (2560 % 256 = 0).
    *   `n_ff % 32 == 0` (640 % 32 = 0).
2.  **Tensor Names:**
    *   Ensure `dense.txt` has `fc_embedding.weight`, `fc_hidden.weight`, `self_attn.q_proj.weight`, etc.
3.  **Norm Offsets:**
    *   Verify `nextn.enorm` values are ~1.0 (offset included) vs ~0.0 (offset not included).

## 6. Byte/Memory Estimates

*   **Experts (Native Q4_K/Q8_0):**
    *   Gate/Up (Q4_K): 512 * 2 * (2560*640/256) * 144 bytes = 512 * 2 * 10 * 144 = 1,474,560 bytes/expert? No.
    *   Per expert gate: 2560*640 = 1,638,400 elements. Blocks: 6,400. Bytes: 6,400 * 144 = 921,600.
    *   Gate+Up: 1,843,200 bytes.
    *   Down (Q8_0): 640*2560 = 1,638,400 elements. Blocks: 51,200. Bytes: 51,200 * 34 = 1,740,800.
    *   Total per expert: ~3.58 MB.
    *   Total 512 experts: **~1.83 GB**.
*   **Dense (Q8_0/BF16):** ~0.2 GB.
*   **Total MTP VRAM:** **~2.0 GB**.
*   **Comparison:** Current Q2 MTP is ~0.9 GB. This is a **2.2x increase**.
*   **Impact:** On RTX 4060 (8 GB), this may cause OOM if the main model uses >5.5 GB. On RTX 5070 Ti (16 GB), it is safe.

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
