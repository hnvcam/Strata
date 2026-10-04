VRAM headroom) to isolate kernel correctness from memory pressure. If the 5070 Ti holds the MTP and the main model fits, proceed to test the 4060 split configuration. If the 4060 OOMs, document that this sidecar is incompatible with the current dual-GPU split strategy on this hardware profile, requiring either a smaller quantization (e.g., Q3_K) or a single-GPU deployment on the 5070 Ti.

### 11. Final Review Checklist for Maintainer

Before accepting any implementation, verify these specific points against the source:

1.  **Activation Stride:** Confirm `hit_xq_` stride in `mtp.cpp` matches `q8_1` block size (36 bytes) when native mode is active, not `q8_0` (34 bytes).
2.  **Scratch Alignment:** Ensure `native_expert_scratch_bytes` allocation is 256-byte aligned as required by `iq_kernels.cu` internal logic.
3.  **Norm Double-Offset:** Add a unit test that loads a known norm value from the sidecar and verifies it is *not* incremented by 1.0 in the runtime.
4.  **Router Type:** Verify `mlp.gate.weight` is loaded as BF16. If the converter fails to convert `ffn_gate_inp` (F32) to BF16, the `bf16()` accessor in `mtp.cpp` will read garbage.
5.  **Expert Grouping:** Verify that `moe_group_resident` (or its native equivalent) correctly maps expert IDs to the new `NativeExpertLayout` pointers. The current `moe_group_resident` assumes the Q2 blob stride. A new grouping function or a modified stride calculation is required.

### 12. What Cannot Be Concluded Without Testing

*   **Kernel Compatibility:** Whether `native_expert_grouped` is compiled with support for `Q4_K` (type 12) and `Q8_0` (type 8) for the specific dimensions `n_embd=2560, n_ff=640`. The `is_iq` function in `iq_kernels.cu` line 1269 includes type 12 and 8, but `native_expert_supported` (line 1417) must return true. This is a compile-time/runtime check that must be verified.
*   **Performance Regression:** Whether the overhead of `q8_1` quantization for dense tensors (if required) outweighs the bandwidth savings of Q4_K weights.
*   **Acceptance Rate:** Whether Q4_K_M experts provide a meaningful improvement in draft acceptance over Q2_0 to justify the 3x VRAM cost.

### Summary

The initial proposal was **architecturally unsound** because it ignored the tight coupling between the Q2 expert format and the MTP runtime's activation pipeline. The corrected proposal requires a **significant runtime refactor** to support the native expert path, including changes to activation quantization, buffer allocation, and kernel dispatch. It is not a simple "import and execute" task. The sidecar is **not compatible** with the current Strata MTP runtime without these changes. The "conversion only" approach is insufficient.
