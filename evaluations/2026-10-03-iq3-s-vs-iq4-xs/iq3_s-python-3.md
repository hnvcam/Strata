# Proposal: Adapt Strata MTP Runtime for External Q4_K_M Sidecar

## 1. Executive Summary
The external sidecar (`mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`) is **structurally incompatible** with the current Strata MTP pipeline. The current pipeline (`tools/mtp_rt.py`, `src/core/mtp.cpp`) expects fused `Q2_0` experts, separate `Q8_0`/`BF16` dense weights, and specific tensor names (`mtp.layers.0.*`). The sidecar provides separate `Q4_K` experts, fused `Q4_K` embedding projections, and `blk.48.*` naming.

**Recommended Approach:** Implement a **pre-conversion tool** (`tools/mtp_rt_sidecar.py`) that transforms the sidecar into the existing legacy format (`experts.bin` Q2_0, `dense.bin` Q8_0/BF16). This avoids risky C++ kernel changes and leverages existing, tested kernels (`moe_grouped_s2`, `iq_mmvq`). Direct native Q4_K support in the MTP runtime is deferred due to kernel layout mismatches (`moe_grouped_s2` vs `native_expert_grouped`).

## 2. Affected Files
*   **New:** `tools/mtp_rt_sidecar.py` (Conversion script).
*   **Unchanged:** `src/core/mtp.cpp`, `include/strata/kernels/cpu/expert.hpp`, `src/kernels/cuda/iq_kernels.cu` (No C++ changes required for the recommended path).
*   **Configuration:** `setup.py` or `serve/server.py` may need a flag to point to the converted `rt` directory instead of the raw sidecar.

## 3. Conversion Design (`tools/mtp_rt_sidecar.py`)

This tool reads the sidecar GGUF and writes `experts.bin`, `dense.bin`, and `dense.txt` compatible with `src/core/mtp.cpp`.

### A. Expert Conversion (Q4_K → Q2_0 Blob)
*   **Input:** `blk.48.ffn_gate_exps.weight` (Q4_K), `blk.48.ffn_up_exps.weight` (Q4_K), `blk.48.ffn_down_exps.weight` (Q8_0).
*   **Process:**
    1.  **Dequantize** all three tensors to FP32.
    2.  **Reshape/Transpose:** GGUF shapes are `[H, FF, NE]` (innermost first). Convert to `[NE, FF, H]` for gate/up and `[NE, H, FF]` for down to match internal expectations.
    3.  **Fuse:** Concatenate Gate and Up into a single `[NE, 2*FF, H]` array.
    4.  **Requantize:** Use `q2_0()` logic from `tools/mtp_pack.py` (lines 58-82) to convert FP32 to `Q2_0`.
    5.  **Blob Layout:** Use `blob_of()` logic from `tools/mtp_rt.py` (lines 48-58) to pack into the interleaved `experts.bin` format.
*   **Rationale:** The current MTP runtime uses `moe_grouped_s2` (`src/core/mtp.cpp` line 533), which strictly requires the `Q2_0` blob layout. Native `Q4_K` support would require switching to `native_expert_grouped` and changing the data layout, which is a significant C++ refactor.

### B. Dense Weight Conversion (Mixed Quant → Q8_0/BF16)
*   **Input:** `blk.48.*` tensors.
*   **Process:**
    1.  **Embedding/Hidden Split:**
        *   Read `blk.48.nextn.eh_proj.weight` (Q4_K, shape `[5120, 2560]`).
        *   **Split** into two tensors:
            *   `fc_embedding.weight`: First 2560 rows (Embedding part).
            *   `fc_hidden.weight`: Last 2560 rows (Hidden part).
        *   **Quantize** both to `Q8_0` using `q8_0()` from `tools/mtp_rt.py` (lines 34-45).
    2.  **Attention/Shared Expert:**
        *   Rename `blk.48.self_attn.*` → `self_attn.*`.
        *   Rename `blk.48.mlp.shared_expert.*` → `mlp.shared_expert.*`.
        *   **Quantize** to `Q8_0` (if currently Q4_K/Q6_K) to match `dense.bin` expectations for MMVQ kernels.
    3.  **Hyper-Connections (HC):**
        *   Rename `blk.48.hc_*` → `hc_*`.
        *   **Dequantize** Q4_K/Q6_K to **BF16**. The current `dense.bin` loader (`tools/mtp_rt.py` line 96) expects BF16 for non-Q8 tensors.
    4.  **Norms:**
        *   Read `blk.48.nextn.enorm`, `hnorm`, etc.
        *   **Check Metadata:** The sidecar states norms *already include* the Gemma +1 offset.
        *   **Action:** Do **NOT** apply `+1.0` (unlike `tools/mtp_rt.py` line 86). Store as F32.

### C. Output Format
*   `experts.bin`: 512 blobs of size `BLOB` (1,382,400 bytes).
*   `dense.bin`: Concatenated Q8_0/BF16/F32 weights.
*   `dense.txt`: Index file with names `fc_embedding.weight`, `self_attn.q_proj.weight`, etc.

## 4. Runtime Compatibility Checks

The existing `src/core/mtp.cpp` loader will validate:
1.  **Tensor Names:** Must match `required[]` list (lines 201-204). The conversion tool must ensure `fc_embedding.weight` and `fc_hidden.weight` exist.
2.  **Quantization Types:** `dense.txt` must specify `q8_0` for attention/shared experts.
3.  **Expert Size:** `experts.bin` must be exactly `512 * BLOB` bytes.

## 5. Memory Implications

*   **Current Q2 MTP:** ~0.7 GB experts + ~0.2 GB dense = ~0.9 GB VRAM.
*   **Converted Q4_K Sidecar (to Q2_0):**
    *   Experts: Same size as current Q2_0 (~0.7 GB).
    *   Dense: Q8_0/BF16 mix. Similar to current (~0.2 GB).
    *   **Total:** ~0.9 GB. **No VRAM increase** compared to current Q2 MTP.
*   **Native Q4_K (Hypothetical):**
    *   Experts: ~1.8 GB (Q4_K is ~2.5x larger than Q2_0).
    *   Dense: ~0.3 GB.
    *   **Total:** ~2.1 GB.
    *   **Risk:** On RTX 4060 (8 GB), this might cause OOM if the main model uses >5.5 GB. The recommended conversion path avoids this risk.

## 6. Validation Plan

1.  **Static Check:**
    *   Run `tools/mtp_rt_sidecar.py` on the sidecar.
    *   Verify `experts.bin` size is `512 * 1382400`.
    *   Verify `dense.txt` contains `fc_embedding.weight` and `fc_hidden.weight`.
2.  **Numerical Check:**
    *   Compare dequantized FP32 experts from the sidecar vs. the converted Q2_0 blob. Expect RMS error consistent with Q2_0 quantization noise.
    *   Verify `fc_embedding` and `fc_hidden` split correctly reconstructs `eh_proj`.
3.  **Runtime Smoke Test:**
    *   Load the converted `rt` directory in Strata.
    *   Run a short generation (e.g., 10 tokens).
    *   Check for NaNs in logits (indicates norm offset errors).
    *   Check draft acceptance rate. Expect it to be **lower** than native Q4_K (due to Q2_0 requantization) but **higher** than a broken load.

## 7. Remaining Uncertainties

1.  **Acceptance Rate Impact:** Requantizing Q4_K experts to Q2_0 loses precision. The draft acceptance rate may drop compared to a hypothetical native Q4_K implementation. This is a trade-off for stability.
2.  **Norm Offset Metadata:** The sidecar description claims norms include +1. If this is incorrect, the model will produce garbage. The conversion tool should ideally verify this by checking if norm values are near 0 (no offset) or near 1 (offset included).
3.  **Kernel Efficiency:** The `moe_grouped_s2` kernel is optimized for Q2_0. It will run at the same speed as the current Q2 MTP, regardless of the source precision.

## 8. Conclusion

Implement `tools/mtp_rt_sidecar.py` to convert the sidecar to the legacy Q2_0/Q8_0 format. This is the lowest-risk path to enable the sidecar without modifying C++ kernels. It ensures compatibility with `src/core/mtp.cpp` and avoids VRAM spikes on smaller GPUs. Native Q4_K support should be considered a separate, higher-priority engineering task if precision loss from Q2_0 requantization is unacceptable.
