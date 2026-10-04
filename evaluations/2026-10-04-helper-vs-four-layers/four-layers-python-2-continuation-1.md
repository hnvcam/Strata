### 8. Concrete Corrections and Missing Requirements

To make the proposal viable, the following specific corrections must be applied to the conversion and runtime strategy.

#### A. Correction: Expert Data Flow and Activation Quantization
The previous proposal incorrectly assumed `native_expert_grouped` could be called with the existing activation buffers.
*   **Fact:** `mtp.cpp` line 532 calls `quantize_q8_0_scaled`. This produces `q8_0` blocks (32 values, fp16 scale).
*   **Fact:** `native_expert_grouped` (via `iq_kernels.hpp` line 59) expects `x_q8_1` (32 values, fp16 scale **and** fp16 sum).
*   **Correction:** The runtime must replace `quantize_q8_0_scaled` with `quantize_q8_1_rows` (from `iq_kernels.hpp` line 23) when the native path is active. The buffer `hit_xq_` in `mtp.cpp` (line 272) is currently sized for `q8_0` (`N/32 * 34`). It must be resized for `q8_1` (`N/32 * 36` or similar, depending on alignment) and the stride updated.
*   **Missing Requirement:** The scratch buffer allocation in `mtp.cpp` (lines 258–279) does not currently allocate space for `native_expert_scratch_bytes`. This must be added to the `Bump` allocator logic when the native format is detected.

#### B. Correction: Dense Tensor Accessors and Kernels
The proposal suggested using `iq_mmvq` for dense tensors but ignored the current accessor pattern.
*   **Fact:** `mtp.cpp` uses `q8(name)` and `bf16(name)` which return pointers to specific offsets in `dense_`. These assume a fixed layout or type.
*   **Correction:** The `Tensor` struct in `mtp.hpp` (line 142) stores `kind`. The loader must parse the new `dense_native.txt` which will contain types like `q4_k`, `q6_k`, `q5_0`.
*   **Runtime Change:** A new helper `get_tensor_ptr(name, &type)` is required.
    *   If type is `Q8_0`/`BF16`/`F32`: Use existing kernels (`bf16_gemv_fp32_mmvf`, etc.).
    *   If type is `Q4_K`/`Q6_K`/`Q5_0`: Use `iq_mmvq`.
    *   **Crucial:** `iq_mmvq` requires activations in `q8_1` format. The current dense path in `mtp.cpp` (lines 526–545) uses `bf16_gemv_fp32_mmvf` which takes BF16 activations.
    *   **Implication:** If *any* dense tensor is quantized (Q4_K), the entire dense activation stream for that layer must be quantized to `q8_1` before the MMV calls. This is a significant change to the data flow in `mtp.cpp` lines 522–550. It is not enough to just change the weight pointer; the input `mixed_` must be quantized to `q8_1` instead of being used as BF16/F32.

#### C. Correction: Input Projection Splitting
*   **Fact:** Sidecar has `nextn.eh_proj.weight` [5120, 2560].
*   **Fact:** Runtime expects `fc_embedding.weight` [2560, 2560] and `fc_hidden.weight` [2560, 2560].
*   **Correction:** The converter `mtp_rt_v2.py` must split `nextn.eh_proj.weight` into two tensors.
    *   **Constraint:** The split is along the output dimension (rows). `eh_proj` output is 5120. Rows 0–2559 correspond to embedding input? Or columns?
    *   **Spec Check:** "each output row concatenating the embedding input columns followed by hidden input columns." This implies the *input* is concatenated. The weight matrix shape is `[5120, 2560]` (Output x Input). Wait, standard GGUF is `[Out, In]` or `[In, Out]`?
    *   **GGUF Fact:** `gguf_reader.py` line 101 reads shape. `mtp_pack.py` line 169 says "ggml order is innermost-first".
    *   **Ambiguity:** If the sidecar stores `[5120, 2560]` and the runtime expects two `[2560, 2560]` matrices, the converter must slice the tensor. If the quantization blocks (Q4_K, 256 values) align with the slice boundary (2560 is a multiple of 256), this is safe. If not, it requires dequantization and requantization. **Assumption:** 2560 is divisible by 256, so slicing raw blocks is safe.

#### D. Missing Requirement: Router Weight Type
*   **Fact:** Sidecar `ffn_gate_inp` is F32.
*   **Fact:** Runtime `mtp.cpp` line 526 uses `bf16("mlp.gate.weight")`.
*   **Correction:** The converter must convert `ffn_gate_inp` (F32) to BF16 to match the runtime expectation, OR the runtime must be updated to handle F32 router weights. Given the router is small, converting to BF16 in the converter is the path of least resistance.

### 9. Revised Implementation Plan (Proposal Only)

1.  **Converter (`tools/mtp_rt_v2.py`):**
    *   Read sidecar GGUF.
    *   **Experts:** Extract raw Q4_K/Q8_0 blocks for `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`. Write to `experts_native.bin` in `[gate|up|down]` layout per expert.
    *   **Dense:**
        *   Split `nextn.eh_proj` into `fc_embedding` and `fc_hidden`.
        *   Convert `ffn_gate_inp` to BF16.
        *   Dequantize HC weights (`hc_*`) to BF16.
        *   Keep Attention/Shared Expert weights in native Q4_K/Q6_K/Q8_0.
        *   Copy Norms as F32 (no +1.0).
        *   Write `dense_native.bin` and `dense_native.txt` (with type IDs).
    *   **Skip:** Indexer tensors.

2.  **Runtime (`src/core/mtp.cpp`):**
    *   **Detection:** Check for `dense_native.txt`.
    *   **Allocation:**
        *   If native: Allocate `experts_` based on `NativeExpertLayout::bytes`. Allocate `native_expert_scratch_`.
        *   Resize `hit_xq_` for `q8_1` blocks.
    *   **Execution Loop (Lines 522–550):**
        *   **Quantize Activations:** Replace `quantize_q8_0_scaled` with `quantize_q8_1_rows` for `mixed_`.
        *   **Dense MMVs:**
            *   For `fc_embedding`/`fc_hidden`: If Q4_K, use `iq_mmvq` with `q8_1` input. If BF16, use existing.
            *   For Attention/Shared: Use `iq_mmvq` for quantized weights.
        *   **Experts:** Replace `moe_grouped_s2` with `native_expert_grouped`. Pass `x_q8_1` (from `hit_xq_`), `experts_` (native layout), and `native_expert_scratch_`.
        *   **HC:** Use BF16 weights (dequantized by converter) with existing `gr_write` logic.

3.  **Validation:**
    *   **Unit Test:** Verify `native_expert_grouped` output matches reference for Q4_K/Q8_0.
    *   **Integration:** Run `mtp_rt_v2.py` on sidecar. Load in Strata. Check for OOM.
    *   **Accuracy:** Compare draft acceptance rate vs Q2 baseline.

### 10. Conclusion on Feasibility

The proposal is **feasible** but **high-risk** due to the required runtime changes. The "dual-path" approach is necessary because the Q2 path is deeply integrated with the custom blob format. The primary blocker is not the file format, but the **activation quantization mismatch** (`q8_0` vs `q8_1`) and the **scratch buffer requirements** for the native expert kernel.

**Recommendation:** Proceed with the converter first. Then, implement the runtime changes behind a feature flag `STRATA_MTP_NATIVE_EXPERTS=1`. Test on the 5070 Ti first (more
