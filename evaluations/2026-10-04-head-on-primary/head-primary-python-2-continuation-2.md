### 9. Final Deliverable: Corrected Proposal & Review Summary

**Status:** The previous proposal is **rejected** due to architectural incompatibility. The following is the corrected, source-verified proposal.

#### 1. Core Incompatibility (Verified)
*   **Runtime Coupling:** `src/core/mtp.cpp` is hard-coded to the **Q2_0** expert format. It uses `moe_grouped_s2` (line 533) and allocates memory based on `strata::kernels::cpu::BLOB` (line 182), which is defined in `include/strata/kernels/cpu/expert.hpp` (line 45) specifically for Q2_0 (18 bytes/64 values).
*   **Kernel Mismatch:** The MTP runtime does **not** call `native_expert_grouped` (from `iq_kernels.hpp`). It uses specialized Q2_0 kernels. Therefore, simply pointing the loader at Q4_K data will cause memory corruption or kernel crashes.
*   **Dense Tensor Rigidity:** The loader (`src/core/mtp.cpp` lines 201–204) explicitly requires `q8_0` for key dense tensors. The forward pass uses `bf16_gemv_fp32_mmvf` and `shared_expert` with hardcoded `GGML_Q8_0` types (line 536). It does not support Q4_K/Q6_K/Q5_0 dense tensors.

#### 2. Corrected Proposal: Lossy Conversion Path
Since modifying the C++ runtime is out of scope for a "proposal and review" and the runtime is not modular, the only viable path is to **convert the sidecar to the native Q2_0 format** during the Python preprocessing step.

**Step 1: New Converter Tool (`tools/mtp_rt_from_sidecar.py`)**
*   **Input:** `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`.
*   **Expert Processing:**
    1.  Read `blk.48.ffn_gate_exps.weight` (Q4_K), `blk.48.ffn_up_exps.weight` (Q4_K), and `blk.48.ffn_down_exps.weight` (Q8_0).
    2.  **Dequantize** these tensors to FP32 using `gguf-py` or `ggml` reference dequantizers.
    3.  **Re-quantize** the FP32 data to **Q2_0** using the existing `q2_0` function from `tools/mtp_pack.py` (lines 58–82). This ensures the output matches the `experts.bin` format expected by `src/core/mtp.cpp`.
    4.  **Relayout:** Interleave gate/up rows and separate codes/scales into the `experts.bin` blob format defined in `include/strata/kernels/cpu/expert.hpp` (lines 48–51).
*   **Dense Tensor Processing:**
    1.  Map sidecar names (`blk.48.attn_q_norm`) to runtime names (`self_attn.q_norm`).
    2.  **Dequantize** Q4_K/Q6_K/Q5_0 dense tensors to FP32.
    3.  **Re-quantize** to **Q8_0** (for tensors in the `Q8` list in `mtp_rt.py`) or convert to **BF16** (for others), matching the `dense.bin` format.
    4.  **Hyper-Connections:** Dequantize Q4_K/Q5_0 HC tensors to FP32 and convert to **BF16**.
*   **Norm Handling:**
    1.  Sidecar norms have `+1` offset.
    2.  `tools/mtp_rt.py` adds `+1.0` (line 86).
    3.  **Action:** The converter must **subtract 1.0** from the sidecar norms before writing them to `dense.bin` (as F32), so that when `mtp_rt.py` (or the new converter's equivalent logic) adds `+1.0`, the result is correct. Alternatively, if bypassing `mtp_rt.py`, write raw weights and ensure the runtime expects raw. Given the existing pipeline, subtracting 1.0 is safer.

**Step 2: Runtime Changes**
*   **None.** The existing `src/core/mtp.cpp` remains unchanged.

**Step 3: Validation**
*   **Acceptance Rate:** Compare the draft acceptance rate of the Q2_0-converted sidecar vs. the original Q2_0 MTP.
*   **VRAM:** Verify that the total VRAM usage (weights + `arena_` + `state_arena_`) fits on the RTX 4060 (8 GB) when the drafter is on the last stage.

#### 3. What Cannot Be Concluded Without Testing
1.  **Precision Benefit:** Whether the Q4_K sidecar, when converted to Q2_0, yields better acceptance than the original Q2_0 MTP. The conversion is lossy, so the benefit may be negligible or negative.
2.  **Performance:** The overhead of dequantizing Q4_K to FP32 and re-quantizing to Q2_0 in Python may be significant.
3.  **Kernel Efficiency:** The Q2_0 kernels are highly optimized. The Q4_K sidecar's precision advantage is lost in conversion.

#### 4. Alternative: Full Runtime Rewrite (High Effort)
If the Q4_K precision is critical, a full rewrite of `src/core/mtp.cpp` is required to:
1.  Replace `moe_grouped_s2` with `native_expert_grouped`.
2.  Replace `bf16_gemv_fp32_mmvf` with `iq_mmvq`.
3.  Update the loader to support Q4_K/Q6_K/Q5_0 types.
4.  Update `dense.txt` parsing to handle new kinds.
This is a major engineering effort and not recommended for a simple sidecar import.

#### 5. Conclusion
The external Q4_K_M sidecar is **not directly compatible** with Strata's current MTP runtime. The only low-risk proposal is to **convert the sidecar to Q2_0** during preprocessing, accepting the precision loss. Direct integration requires a major rewrite of the MTP runtime.
