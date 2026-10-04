### Final Proposal: Dual-Path MTP Runtime for Q4_K_M Sidecar

**Objective:** Enable Strata to import and execute the external `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` sidecar while preserving its native Q4_K (gate/up) and Q8_0 (down) expert quantization. This proposal introduces a **dual-path runtime** that retains full support for the legacy Q2_0 MTP format.

#### 1. Architectural Strategy
The current MTP runtime (`src/core/mtp.cpp`) is hard-coded to the Q2_0 format. To support Q4_K_M without breaking legacy support, we must:
1.  **Detect Format:** Determine at load time whether the `rt/` directory contains Q2_0 blobs or Native IQ blobs.
2.  **Dispatch Kernels:** Route execution to either the specialized Q2_0 kernels (`moe_grouped_s2`) or the generic Native IQ kernels (`native_expert_grouped`, `iq_mmvq`).
3.  **Converter:** Create a new converter that outputs Native IQ blobs for the sidecar, leaving the existing Q2_0 converter untouched.

#### 2. Affected Files

**Python (Converter):**
*   **New:** `tools/mtp_rt_native.py`
    *   Reads the Q4_K_M sidecar.
    *   Outputs `experts.bin` in **Native Expert Layout** (concatenated gate/up/down raw blocks).
    *   Outputs `dense.bin` with mixed quantizations (Q4_K, Q8_0, BF16, F32).
    *   Outputs `dense.txt` with new `kind` identifiers (`q4_k`, `q8_0`, etc.).
*   **Unchanged:** `tools/mtp_rt.py` (continues to produce Q2_0 format).

**C++ (Runtime):**
*   **`src/core/mtp.cpp`**:
    *   **Loader:** Modify `load()` to parse `dense.txt` for new kinds. Detect expert format via a new metadata field in `dense.txt` or by checking `experts.bin` size against `native_expert_layout` vs `cpu::BLOB`.
    *   **Allocation:** Replace fixed `cpu::BLOB` allocation with dynamic sizing based on detected format.
    *   **Dispatch:** Wrap kernel calls in `if (is_native_format)` blocks.
        *   *Experts:* `moe_grouped_s2` (Q2) vs `native_expert_grouped` (Q4_K).
        *   *Dense:* `bf16_gemv_fp32_mmvf` (Q8/BF16) vs `iq_mmvq` (Q4_K).
        *   *Activations:* `quantize_q8_0_scaled` (Q2) vs `quantize_q8_1_rows` (Native).
*   **`include/strata/core/mtp.hpp`**:
    *   Add `bool is_native_` flag to `MTP` class.
    *   Add `NativeExpertLayout native_layout_` member.

#### 3. Conversion Design (`tools/mtp_rt_native.py`)

**A. Expert Processing (Native Layout)**
1.  **Read:** Load `blk.48.ffn_gate_exps.weight` (Q4_K), `blk.48.ffn_up_exps.weight` (Q4_K), `blk.48.ffn_down_exps.weight` (Q8_0).
2.  **Layout:** Construct one blob per expert (512 total) following `NativeExpertLayout` (see `include/strata/kernels/iq_kernels.hpp` lines 40–48):
    *   `[Gate Rows (Q4_K) | Up Rows (Q4_K) | Down Rows (Q8_0)]`
    *   **No dequantization.** Preserve raw GGUF block bytes.
3.  **Output:** Write `experts.bin`.
    *   **Size:** $\approx 0.57$ GB (vs 0.71 GB for Q2_0).

**B. Dense Tensor Processing**
1.  **Name Mapping:** Map `blk.48.*` to runtime names (e.g., `self_attn.q_proj.weight`).
2.  **Quantization Preservation:**
    *   **Q4_K/Q6_K/Q5_0/Q8_0:** Write raw bytes. Set `kind` to `q4_k`, `q6_k`, etc.
    *   **BF16/F32:** Write raw bytes. Set `kind` to `bf16`/`f32`.
3.  **Hyper-Connections:**
    *   Sidecar uses Q4_K/Q5_0.
    *   **Action:** Write raw bytes. The runtime must support these types for HC kernels. *Fallback:* If HC kernels do not support native types, dequantize to BF16 in the converter.
4.  **Norms:**
    *   Sidecar norms include `+1` offset.
    *   **Action:** Write as **F32** with `+1` offset **included**.
    *   **Runtime Change:** Remove the `+1.0` addition in `src/core/mtp.cpp` (currently in `tools/mtp_rt.py` line 86, but applied at load time in C++ if needed). Ensure the runtime expects pre-offset norms for native path.

**C. Metadata (`dense.txt`)**
*   Add a header line or metadata key: `format=native_iq`.
*   List tensors with new kinds: `q4_k`, `q8_0`, `bf16`, `f32`.

#### 4. Runtime Design (`src/core/mtp.cpp`)

**A. Format Detection**
*   In `load()`, read `dense.txt`. If `format=native_iq` is present, set `is_native_ = true`.
*   Alternatively, check if `experts.bin` size matches `512 * native_expert_layout(...).bytes`.

**B. Kernel Dispatch**
*   **Experts:**
    *   If `is_native_`: Call `native_expert_grouped` (see `include/strata/kernels/iq_kernels.hpp` line 59).
    *   Else: Call `moe_grouped_s2` (current path).
*   **Dense Projections:**
    *   If `is_native_`: Call `iq_mmvq` (see `include/strata/kernels/iq_kernels.hpp` line 26) for Q4_K/Q8_0 tensors.
    *   Else: Call `bf16_gemv_fp32_mmvf` (current path).
*   **Activations:**
    *   If `is_native_`: Use `quantize_q8_1_rows` (see `include/strata/kernels/iq_kernels.hpp` line 23).
    *   Else: Use `quantize_q8_0_scaled` (current path).

**C. Shared Expert**
*   If `is_native_`: Replace `shared_expert` call with `iq_mmvq` for gate/up/down and manual combine.
*   Else: Keep current `shared_expert` call.

#### 5. Compatibility & Memory

**Compatibility Checks:**
1.  **Dimensions:** `n_embd=2560`, `n_ff=640`.
    *   Q4_K block 256: $2560 \% 256 == 0$. OK.
    *   Q8_0 block 32: $640 \% 32 == 0$. OK.
    *   `native_expert_supported` (see `src/kernels/cuda/iq_kernels.cu` line 1417) will return true.
2.  **Kernel Availability:** Verify `iq_mmvq` supports Q4_K for dense layers. (Source `iq_kernels.hpp` line 2 confirms Q4_K support).
3.  **Norm Offset:** Ensure runtime does not double-add `+1.0`.

**Memory Implications:**
*   **Experts:** ~0.57 GB (Q4_K/Q8_0).
*   **Dense:** ~0.2–0.3 GB (mixed).
*   **Arena/State:** ~100–200 MB (unchanged).
*   **Total VRAM:** ~0.9–1.1 GB.
*   **Impact:** Fits on RTX 4060 (8 GB). Slightly larger than Q2_0 (0.71 GB) but smaller than Q8_0 (2.67 GB).

#### 6. Validation Plan
1.  **Loader Test:** Load `rt/` directory. Verify `is_native_` is set correctly.
2.  **Numerical Check:** Compare logits from Q4_K_M MTP against a reference (llama.cpp) for a fixed prompt.
3.  **Acceptance Rate:** Measure draft acceptance on HumanEval. Expect higher than Q2_0.
4.  **Performance:** Profile `native_expert_grouped` vs `moe_grouped_s2`. Q4_K may be slower due to bandwidth.
5.  **Legacy Regression:** Ensure Q2_0 MTP still loads and runs correctly.

#### 7. Uncertainties
1.  **HC Kernel Support:** Does Strata have native Q4_K/Q5_0 kernels for Hyper-Connections? If not, HC tensors must be dequantized to BF16 in the converter.
2.  **Shared Expert Kernel:** Does `shared_expert` support native types? If not, it must be replaced with `iq_mmvq` calls.
3.  **Performance Trade-off:** The Q4_K path may be slower than Q2_0. The acceptance rate gain must justify the latency.

#### 8. Conclusion
This proposal enables Q4_K_M MTP support by extending the runtime to use existing Native IQ kernels. It preserves legacy Q2_0 support via format detection. The primary risk is the engineering effort to refactor `src/core/mtp.cpp` for dual-path dispatch.
