### 8. Final Verdict and Corrected Proposal

The preceding proposal is **rejected** in its current form. It relies on the false premise that the MTP runtime (`src/core/mtp.cpp`) is a generic engine capable of swapping expert formats via configuration or minor loader changes. The source code proves the MTP runtime is a **specialized, hard-coded implementation** for the Q2_0 expert format and Q8_0/BF16 dense layers.

#### Concrete Mistakes Identified:
1.  **Architectural Incompatibility:** The proposal suggests using `native_expert_grouped` for MTP experts. The MTP runtime does not link to or call this function. It uses `moe_grouped_s2`, which is specific to the Q2_0 blob layout defined in `include/strata/kernels/cpu/expert.hpp`. Switching to `native_expert_grouped` requires rewriting the entire MTP forward pass, not just the loader.
2.  **Dense Tensor Support:** The proposal assumes `iq_mmvq` can be used for MTP dense layers. The MTP runtime uses `bf16_gemv_fp32_mmvf` and `shared_expert` with hardcoded Q8_0/BF16 expectations. It does not support Q4_K/Q6_K/Q5_0 dense tensors.
3.  **Memory Underestimation:** The proposal ignores the `arena_` and `state_arena_` allocations in `src/core/mtp.cpp` (lines 206–284), which are substantial and independent of weight quantization. The total VRAM footprint will be significantly higher than the estimated 0.9 GB.
4.  **Converter Limitations:** The current `tools/mtp_rt.py` cannot output Q4_K/Q5_0 for HC tensors. It only outputs Q8_0 or BF16. The proposal fails to address how to handle the sidecar's mixed-quantization HC tensors without dequantizing them to BF16 (losing precision) or rewriting the HC kernels.

#### Corrected Proposal:

Given the constraints (no code modification allowed in this exercise, but a proposal is required), the only viable path that respects the existing architecture is **Option 2: Lossy Conversion**.

**Strategy:**
Convert the external Q4_K_M sidecar **back** to the Strata-native Q2_0 format during the `mtp_rt.py` step. This retains the existing runtime compatibility but sacrifices the precision benefit of the sidecar.

**Steps:**
1.  **New Converter Tool (`tools/mtp_rt_from_sidecar.py`):**
    *   Read the Q4_K_M sidecar GGUF.
    *   **Experts:** Dequantize Q4_K gate/up and Q8_0 down tensors to FP32. Re-quantize them to Q2_0 using the existing `q2_0` function in `tools/mtp_pack.py` (lines 58–82). This ensures the output matches the `experts.bin` format expected by `src/core/mtp.cpp`.
    *   **Dense Tensors:** Dequantize Q4_K/Q6_K/Q5_0 dense tensors to FP32. Re-quantize to Q8_0 (for those in the `Q8` list) or convert to BF16 (for others), matching the `dense.bin` format.
    *   **HC Tensors:** Dequantize Q4_K/Q5_0 HC tensors to FP32 and convert to BF16.
    *   **Norms:** Verify if the sidecar norms have the +1 offset. If yes, subtract 1.0 before writing, because `tools/mtp_rt.py` (line 86) adds 1.0. *Correction:* If the sidecar has +1, and the converter adds +1, we get +2. The converter must **not** add +1 if the source already has it. However, since we are dequantizing to FP32 first, we can control this. The safest path is to dequantize, subtract 1.0 (to get raw weights), and let `mtp_rt.py` add 1.0. Or, if bypassing `mtp_rt.py`, write raw weights and ensure the runtime expects raw. Given `mtp_rt.py` adds +1, the converter should output raw weights (no +1).
    *   **Naming:** Map `blk.48.*` names to `mtp.layers.0.*` names.

2.  **Runtime Changes:** **None.** The existing `src/core/mtp.cpp` remains unchanged.

3.  **Validation:**
    *   Compare the acceptance rate of the Q2_0-converted sidecar vs. the original Q2_0 MTP.
    *   If the acceptance rate is significantly lower, the Q4_K sidecar is not worth the conversion overhead, and the original Q2_0 MTP should be retained.

**Why this is the only valid proposal:**
*   It respects the hard-coded Q2_0 dependency in `src/core/mtp.cpp`.
*   It uses existing helpers (`q2_0` quantizer in `mtp_pack.py`).
*   It requires no changes to the C++ runtime, kernels, or loader.
*   It addresses the naming and norm offset issues.

**What cannot be concluded:**
*   Whether the Q4_K sidecar, when converted to Q2_0, yields better acceptance than the original Q2_0 MTP. This requires empirical testing.
*   Whether the performance penalty of dequantizing Q4_K to FP32 and re-quantizing to Q2_0 is acceptable for the user's workflow.

**Conclusion:**
The external Q4_K_M sidecar is **not directly compatible** with Strata's current MTP runtime. Direct integration requires a major rewrite of the MTP runtime to use generic IQ kernels. The only low-risk proposal is to convert the sidecar to Q2_0, accepting the precision loss.
