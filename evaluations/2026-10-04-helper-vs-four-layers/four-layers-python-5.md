# Final Proposal: Q4_K_M MTP Sidecar Adaptation

## 1. Executive Summary
This proposal outlines the integration of the external `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` sidecar into Strata. The sidecar uses **Q4_K** for routed gate/up experts and **Q8_0** for down experts. This format is **incompatible** with the current legacy Q2_0 runtime (`src/core/mtp.cpp`), which expects a custom Q2_0 blob layout and `q8_0` activations.

To support this while retaining legacy Q2 support, we propose a **dual-path runtime** that detects the new format and switches to the native GGUF expert kernel (`native_expert_grouped`) and `q8_1` activation quantization. This requires a new converter tool and significant changes to the MTP runtime initialization and execution loop.

## 2. Current Incompatibility Analysis

### A. Expert Format and Kernel Mismatch
*   **Current State:** `tools/mtp_rt.py` converts experts to a custom Q2_0 blob layout (`include/strata/kernels/cpu/expert.hpp` lines 35–45). `src/core/mtp.cpp` (line 533) calls `moe_grouped_s2`, which expects this custom layout and `q8_0` activations.
*   **Sidecar State:** Experts are in native GGUF formats: `Q4_K` (gate/up) and `Q8_0` (down).
*   **Conflict:** The runtime cannot load native GGUF blocks into the Q2 blob buffer. The kernel `moe_grouped_s2` cannot process Q4_K/Q8_0 blocks.

### B. Activation Quantization Mismatch
*   **Current State:** `src/core/mtp.cpp` line 532 quantizes activations to `q8_0` (32 values, fp16 scale).
*   **Sidecar Requirement:** The native expert kernel `native_expert_grouped` (`include/strata/kernels/iq_kernels.hpp` line 59) requires `q8_1` activations (32 values, fp16 scale + fp16 sum).
*   **Conflict:** Using `q8_0` activations with `native_expert_grouped` will produce incorrect results.

### C. Dense Tensor Naming and Quantization
*   **Current State:** `tools/mtp_rt.py` strips prefixes (e.g., `mtp.layers.0.`) and forces specific tensors to `Q8_0` or `BF16`. `src/core/mtp.cpp` (lines 201–204) checks for these short names and `q8_0` kind.
*   **Sidecar State:** Uses full names (e.g., `blk.48.self_attn.q_proj.weight`) and mixed quantizations (`Q4_K`, `Q6_K`, `Q5_0`).
*   **Conflict:** The runtime will fail to find tensors by name and will misinterpret quantized weights as `Q8_0`/`BF16`.

### D. Input Projection Structure
*   **Current State:** Runtime expects separate `fc_embedding.weight` and `fc_hidden.weight`.
*   **Sidecar State:** Provides a single fused `nextn.eh_proj.weight` [5120, 2560].
*   **Conflict:** The runtime cannot directly consume the fused tensor without splitting or kernel changes.

## 3. Proposed Design

### A. New Converter: `tools/mtp_rt_v2.py`
Create a new tool to convert the sidecar to a "Native MTP" format.

1.  **Experts:**
    *   Extract raw `Q4_K` and `Q8_0` blocks for `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`.
    *   Reorder into `NativeExpertLayout` (`[gate | up | down]` per expert).
    *   Write to `experts_native.bin`.
2.  **Dense Tensors:**
    *   **Naming:** Map sidecar names to Strata short names (e.g., `blk.48.self_attn.q_proj.weight` -> `self_attn.q_proj.weight`).
    *   **Input Projection:** Split `nextn.eh_proj.weight` into `fc_embedding.weight` and `fc_hidden.weight`. Since the split boundary (2560) aligns with Q4_K block size (256), raw block slicing is safe.
    *   **Router:** Convert `ffn_gate_inp` (F32) to `BF16` to match runtime expectations.
    *   **Hyper-Connections:** Dequantize `hc_*` weights (Q4_K/Q5_0/Q6_K) to `BF16` to simplify runtime logic.
    *   **Norms:** Copy F32 norms directly. **Do not** add +1.0 (sidecar is pre-offset).
    *   **Attention/Shared:** Keep native quantization (`Q4_K`, `Q6_K`, `Q8_0`).
    *   **Indexer:** Skip unused indexer tensors.
    *   Write to `dense_native.bin` and `dense_native.txt` (with GGML type IDs).

### B. Runtime Changes: `src/core/mtp.cpp`
Modify `MTP::init` and the execution loop to support the native path.

1.  **Format Detection:**
    *   Check for `dense_native.txt`. If present, set `native_mode = true`.
2.  **Allocation (Native Mode):**
    *   **Experts:** Allocate `experts_` based on `NativeExpertLayout::bytes` (not `cpu::BLOB`).
    *   **Scratch:** Allocate `native_expert_scratch_` using `native_expert_scratch_bytes`.
    *   **Activations:** Resize `hit_xq_` buffer to accommodate `q8_1` blocks (36 bytes/block vs 34 for `q8_0`).
3.  **Execution Loop (Lines 522–550):**
    *   **Activation Quantization:** Replace `quantize_q8_0_scaled` with `quantize_q8_1_rows` for `mixed_`.
    *   **Dense MMVs:**
        *   For `Q8_0`/`BF16`/`F32` tensors: Use existing kernels.
        *   For `Q4_K`/`Q6_K`/`Q5_0` tensors: Use `iq_mmvq` with `q8_1` activations.
    *   **Experts:** Replace `moe_grouped_s2` with `native_expert_grouped`. Pass `x_q8_1` (from `hit_xq_`), `experts_` (native layout), and `native_expert_scratch_`.
    *   **Hyper-Connections:** Use BF16 weights (dequantized by converter) with existing `gr_write` logic.

## 4. Compatibility and Memory Considerations

### A. Hardware Fit
*   **Current Q2 MTP:** ~840 MB VRAM.
*   **Proposed Q4_K_M MTP:** ~2.6 GB VRAM (Experts ~2.3 GB + Dense ~300 MB).
*   **Impact:**
    *   **RTX 5070 Ti (16 GB):** Likely fits alongside the main model.
    *   **RTX 4060 (8 GB):** **High Risk.** If the MTP is placed on the 4060 (last stage), it must share 8 GB with the last layer of the main model and KV cache. 2.6 GB for MTP leaves <5.4 GB for the rest, which may cause OOM or force a tiny KV cache.
    *   **Recommendation:** Test on 5070 Ti first. If 4060 OOMs, this sidecar is incompatible with the current split strategy on this hardware profile.

### B. Kernel Support
*   Verify `native_expert_supported(12, 8, 2560, 640)` returns `true`. The sidecar uses Q4_K (12) for gate/up and Q8_0 (8) for down.

## 5. Validation Plan

1.  **Unit Test:** Verify `native_expert_grouped` produces correct output for Q4_K/Q8_0 experts at dimensions 2560/640.
2.  **Converter Test:** Run `mtp_rt_v2.py` on the sidecar. Verify `dense_native.txt` has correct types and `experts_native.bin` size matches expectations.
3.  **Integration Test:** Load the converted files in Strata. Check for OOM during initialization.
4.  **Accuracy Test:** Compare draft acceptance rate against the Q2 baseline. Q4_K_M should have higher acceptance.
5.  **Norm Check:** Verify that RMSNorm outputs are correct (no double +1.0 offset).

## 6. Remaining Uncertainties

1.  **VRAM Fit on 4060:** Whether the 2.6 GB MTP fits on the 4060 in a split configuration.
2.  **Performance:** Whether the overhead of `q8_1` quantization for dense tensors outweighs bandwidth savings.
3.  **Accuracy:** Exact impact of Q4_K quantization on draft acceptance rate.
4.  **Kernel Compilation:** Whether `native_expert_grouped` is compiled with support for Q4_K/Q8_0 at these specific dimensions.

## 7. Conclusion

This adaptation is **feasible** but requires a **significant runtime refactor** to support the native expert path. It is not a simple file conversion. The primary risk is **VRAM exhaustion** on the secondary GPU. If the 4060 cannot hold the MTP, this sidecar may require a single-GPU deployment on the 5070 Ti or a smaller quantization format.
