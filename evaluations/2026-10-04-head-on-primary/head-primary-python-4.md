### Revised Proposal: Native Q4_K/Q8_0 MTP Integration

**Decision:** The previous "lossy conversion" proposal is **rejected** based on the clarified requirement that experts must remain Q4_K (gate/up) and Q8_0 (down). The only viable path is to **extend the MTP runtime** to support native IQ formats for experts and dense layers, leveraging existing kernels but requiring significant C++ changes.

#### 1. Architectural Reality Check
*   **Current State:** `src/core/mtp.cpp` is hard-coded to Q2_0 experts (`moe_grouped_s2`) and Q8_0/BF16 dense layers. It does **not** use `native_expert_grouped` or `iq_mmvq`.
*   **Requirement:** Use Q4_K/Q8_0 experts.
*   **Implication:** The MTP runtime must be refactored to use the **Native Expert** path (`native_expert_grouped`) and **IQ MMVQ** path (`iq_mmvq`) for dense layers. This is not a loader change; it is a kernel dispatch change.

#### 2. Affected Files
*   **`tools/mtp_rt.py` (or new `tools/mtp_rt_native.py`):** Must be rewritten to output native GGUF blocks instead of custom Q2_0 blobs.
*   **`src/core/mtp.cpp`:** Must be modified to:
    1.  Parse `dense.txt` for `q4_k`, `q8_0`, `bf16`, `f32` kinds.
    2.  Allocate `experts.bin` based on `native_expert_layout` sizes.
    3.  Replace `moe_grouped_s2` with `native_expert_grouped`.
    4.  Replace `bf16_gemv_fp32_mmvf` with `iq_mmvq` for dense projections.
    5.  Replace `shared_expert` with a version supporting native types.
*   **`include/strata/core/mtp.hpp`:** Update `Tensor` struct to handle new kinds.

#### 3. Conversion Design (`tools/mtp_rt_native.py`)

**A. Expert Tensor Processing (Native Layout)**
1.  **Read:** Load `blk.48.ffn_gate_exps.weight` (Q4_K), `blk.48.ffn_up_exps.weight` (Q4_K), `blk.48.ffn_down_exps.weight` (Q8_0).
2.  **Layout:** The `NativeExpertLayout` (see `include/strata/kernels/iq_kernels.hpp` lines 40–48) expects a single blob per expert: `[gate rows | up rows | down rows]`.
    *   For each expert $e$ (0–511):
        *   Extract the $e$-th slice of gate and up weights.
        *   Concatenate: `gate_data` + `up_data` + `down_data`.
        *   **No dequantization.** Keep raw Q4_K/Q8_0 bytes.
3.  **Output:** Write `experts.bin` as a concatenation of 512 native blobs.
    *   **Size Calculation:**
        *   Gate/Up (Q4_K): $2560 \times 640 \times 512 \times (144/256)$ bytes $\approx 0.47$ GB.
        *   Down (Q8_0): $640 \times 2560 \times 512 \times (34/32)$ bytes $\approx 0.10$ GB.
        *   Total Experts: $\approx 0.57$ GB.

**B. Dense Tensor Processing**
1.  **Name Mapping:** Map `blk.48.*` to `self_attn.*`, `mlp.*`, etc.
2.  **Quantization:**
    *   **Q4_K/Q6_K/Q5_0/Q8_0:** Keep raw bytes. Output `kind` as `q4_k`, `q6_k`, etc.
    *   **BF16/F32:** Keep raw bytes. Output `kind` as `bf16`/`f32`.
3.  **Norms:**
    *   Sidecar norms have `+1` offset.
    *   **Action:** Write norms as **F32** with the `+1` offset **already included**. Do **not** add `+1.0` in the converter. The runtime must be updated to expect pre-offset norms (or the converter must subtract 1.0 if the runtime adds it). *Correction:* `tools/mtp_rt.py` adds `+1.0`. If we bypass `mtp_rt.py` and write directly, we must ensure the runtime logic matches. The safest path is to write **raw weights** (subtract 1.0) and let the runtime add 1.0, OR write **pre-offset** and remove the `+1.0` from the runtime. Given the runtime is being modified, **write pre-offset** and remove the `+1.0` addition in `mtp.cpp`.

**C. Metadata (`dense.txt`)**
*   Format: `name kind rows cols offset bytes`.
*   `kind` values: `q4_k`, `q6_k`, `q5_0`, `q8_0`, `bf16`, `f32`.

#### 4. Runtime Changes (`src/core/mtp.cpp`)

**A. Loader Updates**
1.  **Parse `dense.txt`:** Accept new `kind` strings.
2.  **Expert Allocation:**
    *   Replace `strata::kernels::cpu::BLOB` with `native_expert_layout(GGML_Q4_K, GGML_Q8_0, 2560, 640).bytes`.
    *   Allocate `experts_` buffer of size $512 \times \text{blob\_size}$.
3.  **Dense Allocation:** Allocate `dense_` buffer based on total bytes from `dense.txt`.

**B. Kernel Dispatch**
1.  **Experts:**
    *   Replace `moe_grouped_s2` (line 533) with `native_expert_grouped`.
    *   Construct `NativeExpertLayout` with `gu_type=GGML_Q4_K`, `d_type=GGML_Q8_0`.
    *   Pass `experts_` pointer and group indices.
2.  **Dense Projections:**
    *   Replace `bf16_gemv_fp32_mmvf` with `iq_mmvq`.
    *   Pass the correct `ggml_type` (e.g., `GGML_Q4_K`) and data pointer from `dense_`.
3.  **Shared Expert:**
    *   Replace `shared_expert` (line 543) with a call that supports native types. If no such function exists, use `iq_mmvq` for gate/up/down and combine manually.
4.  **Hyper-Connections:**
    *   Update HC kernels to accept Q4_K/Q5_0/BF16. If no native HC kernels exist, **dequantize HC tensors to BF16** in the converter (as a fallback) to minimize runtime changes.

**C. Activation Formats**
*   `native_expert_grouped` expects **Q8_1** activations (see `iq_kernels.hpp` line 7).
*   Current MTP uses `quantize_q8_0_scaled` (line 532).
*   **Change:** Replace `quantize_q8_0_scaled` with `quantize_q8_1_rows` (see `iq_kernels.hpp` line 23).

#### 5. Compatibility Checks
1.  **Dimensions:**
    *   `n_embd=2560`, `n_ff=640`.
    *   Q4_K block size 256. $2560 \% 256 == 0$. OK.
    *   Q8_0 block size 32. $640 \% 32 == 0$. OK.
    *   `native_expert_supported` (line 1417) will return true.
2.  **Alignment:**
    *   Ensure `dense.bin` offsets are aligned to 256 bytes (or `general.alignment`).
3.  **Norm Offset:**
    *   Verify that the runtime no longer adds `+1.0` if the converter writes pre-offset norms.

#### 6. Memory Implications
*   **Experts:** ~0.57 GB (Q4_K/Q8_0).
*   **Dense:** ~0.2–0.3 GB (mixed Q4_K/Q8_0/BF16).
*   **Arena/State:** ~100–200 MB (unchanged).
*   **Total VRAM:** ~0.9–1.1 GB.
*   **Comparison:** Slightly larger than Q2_0 (0.71 GB) but smaller than Q8_0 (2.67 GB). Fits on RTX 4060 (8 GB).

#### 7. Validation Plan
1.  **Unit Test:** Verify `native_expert_layout` returns correct byte counts for Q4_K/Q8_0.
2.  **Integration Test:** Load the new `rt/` directory. Check for loader errors.
3.  **Numerical Check:** Compare output logits of the new MTP against a reference implementation (e.g., llama.cpp) for a fixed prompt.
4.  **Acceptance Rate:** Measure draft acceptance. Expect higher than Q2_0 due to Q4_K precision.
5.  **Performance:** Profile `native_expert_grouped` vs `moe_grouped_s2`. Q4_K may be slower due to larger memory bandwidth requirements.

#### 8. Remaining Uncertainties
1.  **HC Kernel Support:** Does Strata have native Q4_K/Q5_0 kernels for Hyper-Connections? If not, HC tensors must be dequantized to BF16 in the converter.
2.  **Shared Expert Kernel:** Does `shared_expert` support native types? If not, it must be replaced with `iq_mmvq` calls.
3.  **Performance:** The Q4_K path may be slower than Q2_0 due to increased memory traffic. The acceptance rate gain must outweigh this.

#### 9. Conclusion
This proposal requires **significant C++ changes** to `src/core/mtp.cpp` to switch from the specialized Q2_0 path to the generic Native IQ path. It is the only way to satisfy the requirement of keeping experts in Q4_K/Q8_0. The maintainer must decide if the engineering effort and potential performance regression are worth the precision gain.
