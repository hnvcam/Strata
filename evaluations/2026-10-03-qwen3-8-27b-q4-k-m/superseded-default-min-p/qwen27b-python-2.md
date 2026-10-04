# Review of Proposal: Importing Q4_K_M MTP Sidecar

## 1. Critical Mistake: Misidentification of the Expert Kernel Path

**Mistake:** The proposal recommends using `native_expert_grouped` for the MTP draft layer experts.

**Correction:** `native_expert_grouped` is **not** the correct kernel for the MTP draft layer. The MTP draft layer uses a **different MoE execution path** than the main model.

**Source Evidence:**
- `src/core/mtp.cpp` (lines 524–534) shows the MTP MoE path:
  ```cpp
  moe_group_resident(ids_, ..., experts_, (int64_t) strata::kernels::cpu::BLOB, ...);
  quantize_q8_0_scaled(mixed_, hit_xq_, hit_xs_, ...);
  moe_grouped_s2(grp_ptr_, ..., hit_xq_, hit_xs_, hit_scratch_, parts_, cs);
  ```
- `moe_grouped_s2` is a **specialized kernel** for the MTP draft layer that operates on the **Q2_0 blob layout** defined in `include/strata/kernels/cpu/expert.hpp`. It is **not** the same as `native_expert_grouped`.
- `native_expert_grouped` (in `src/kernels/cuda/iq_kernels.cu`) is used for the **main model's** IQ4_XS/IQ2_XS experts, which have a different layout and execution model (grouped by expert ID, not resident in VRAM).

**Implication:** The MTP draft layer **cannot** reuse `native_expert_grouped` without significant refactoring. The Q2_0 blob layout is hard-coded into `moe_grouped_s2` and the MTP runtime's memory management.

**Correct Approach:** A **new kernel** or a **new blob layout** must be created for Q4_K/Q8_0 experts in the MTP draft layer. This is a **major** change, not a simple dispatch.

---

## 2. Critical Mistake: Misunderstanding of the Dense Tensor Requirements

**Mistake:** The proposal suggests converting Q4_K/Q6_K dense tensors to Q8_0 and Q4_K/Q6_K hyper-connection weights to BF16.

**Correction:** The MTP runtime **does not support** Q4_K or Q6_K dense tensors. It only supports Q8_0, BF16, and F32. However, the **conversion** is not as simple as dequantizing to F32 and re-quantizing.

**Source Evidence:**
- `tools/mtp_rt.py` (lines 29–31) defines the set of tensors that are quantized to Q8_0:
  ```python
  Q8 = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", ...}
  ```
- `src/core/mtp.cpp` (lines 201–204) checks for these Q8_0 tensors:
  ```cpp
  const char* required[] = {"fc_embedding.weight", "fc_hidden.weight", "self_attn.q_proj.weight", ...};
  for (const char* n : required) if (!q8(n)) { err = ...; return false; }
  ```
- The MTP runtime uses `bf16_gemv_fp32_mmvf` for BF16 tensors and `q8` for Q8_0 tensors. It **does not** have a path for Q4_K or Q6_K.

**Implication:** The Python tooling must **dequantize** the Q4_K/Q6_K dense tensors to F32 and then **re-quantize** them to Q8_0 or BF16. This is a **lossy** conversion and may degrade the draft acceptance rate.

**Correct Approach:** The Python tooling must:
1.  Dequantize Q4_K/Q6_K dense tensors to F32.
2.  Re-quantize them to Q8_0 (for the large projections) or BF16 (for the hyper-connections).
3.  Verify that the re-quantized tensors produce the same results as the original Q4_K/Q6_K tensors (within a tolerance).

---

## 3. Critical Mistake: Misunderstanding of the Hyper-Connection Weights

**Mistake:** The proposal suggests converting Q4_K/Q6_K hyper-connection weights to BF16.

**Correction:** The MTP runtime **expects** the hyper-connection weights to be in BF16, but the **conversion** is not straightforward. The hyper-connection weights are used in a **specific** way in the MTP draft layer, and the conversion must preserve the **exact** values.

**Source Evidence:**
- `src/core/mtp.cpp` (lines 522–523) shows the hyper-connection read:
  ```cpp
  fused_gr_read_multi(fa, T, xn_, cs);
  ```
- The hyper-connection weights are stored in `dense.bin` as BF16 (see `tools/mtp_rt.py`, lines 88–96).

**Implication:** The Python tooling must **dequantize** the Q4_K/Q6_K hyper-connection weights to F32 and then **re-quantize** them to BF16. This is a **lossy** conversion and may degrade the draft acceptance rate.

**Correct Approach:** The Python tooling must:
1.  Dequantize Q4_K/Q6_K hyper-connection weights to F32.
2.  Re-quantize them to BF16.
3.  Verify that the re-quantized weights produce the same results as the original Q4_K/Q6_K weights (within a tolerance).

---

## 4. Missing Requirement: Tensor Name Mapping

**Mistake:** The proposal does not specify how the tensor names will be mapped.

**Correction:** The sidecar uses `blk.48.*` names, while the MTP runtime expects `mtp.layers.0.*` names. The Python tooling must **map** the names.

**Source Evidence:**
- `tools/mtp_rt.py` (lines 70–71) expects `mtp.layers.0.mlp.experts.gate_up_proj` and `mtp.layers.0.mlp.experts.down_proj`.
- The sidecar uses `blk.48.ffn_gate_exps.weight`, `blk.48.ffn_up_exps.weight`, `blk.48.ffn_down_exps.weight`.

**Implication:** The Python tooling must **map** the sidecar's tensor names to the MTP runtime's expected names.

**Correct Approach:** The Python tooling must:
1.  Define a mapping from `blk.48.*` names to `mtp.layers.0.*` names.
2.  Apply the mapping when reading the sidecar's tensors.

---

## 5. Missing Requirement: Expert Blob Layout

**Mistake:** The proposal does not specify the expert blob layout for Q4_K/Q8_0 experts.

**Correction:** The MTP runtime expects a **specific** blob layout for Q2_0 experts. A **new** blob layout must be defined for Q4_K/Q8_0 experts.

**Source Evidence:**
- `include/strata/kernels/cpu/expert.hpp` (lines 47–51) defines the Q2_0 blob layout:
  ```cpp
  inline constexpr size_t O_GU_CODES = 0;
  inline constexpr size_t O_D_CODES = 2ull * FF * ROW_GU;
  inline constexpr size_t O_GU_SCALES = O_D_CODES + 1ull * H * ROW_D;
  inline constexpr size_t O_D_SCALES = O_GU_SCALES + 2ull * FF * SC_GU * 2;
  ```

**Implication:** A **new** blob layout must be defined for Q4_K/Q8_0 experts, and a **new** kernel must be written to process it.

**Correct Approach:** The Python tooling must:
1.  Define a new blob layout for Q4_K/Q8_0 experts.
2.  Pack the experts into this layout.
3.  The C++ runtime must be updated to read this layout and dispatch to a new kernel.

---

## 6. Missing Requirement: Validation of the Conversion

**Mistake:** The proposal does not specify how the conversion will be validated.

**Correction:** The conversion from Q4_K/Q6_K to Q8_0/BF16 is **lossy**. The validation must ensure that the converted tensors produce the same results as the original tensors.

**Source Evidence:**
- `tools/mtp_pack.py` (lines 163–167) shows how the conversion is validated:
  ```python
  for e in range(min(a.check_experts, shape[0])):
      ref = x[e].reshape(-1)
      got = dequant(a.experts, parts[e], ref.size)
      errs.append(float(np.sqrt(((got - ref) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-12)))
  ```

**Implication:** The Python tooling must **validate** the conversion by comparing the dequantized values of the converted tensors to the original values.

**Correct Approach:** The Python tooling must:
1.  Dequantize the converted tensors to F32.
2.  Compare the dequantized values to the original values.
3.  Report the relative RMS error.

---

## 7. Verified Source Facts

1.  The MTP runtime uses `moe_grouped_s2` for the Q2_0 experts.
2.  The MTP runtime does not support Q4_K or Q6_K dense tensors.
3.  The MTP runtime expects the hyper-connection weights to be in BF16.
4.  The sidecar uses `blk.48.*` tensor names.
5.  The sidecar uses Q4_K for gate/up experts and Q8_0 for down experts.

## 8. Untested Expectations

1.  The draft acceptance rate with Q4_K/Q8_0 experts will be the same as with Q2_0 experts.
2.  The performance of the new kernel will be the same as `moe_grouped_s2`.
3.  The memory bandwidth will not be a bottleneck with the larger expert memory.
4.  The conversion from Q4_K/Q6_K to Q8_0/BF16 will not degrade the draft acceptance rate.

## 9. Conclusion

The proposal contains **critical mistakes** in the identification of the expert kernel path and the understanding of the dense tensor requirements. The correct approach is to:
1.  Define a **new** blob layout for Q4_K/Q8_0 experts.
2.  Write a **new** kernel to process this layout.
3.  Convert the Q4_K/Q6_K dense tensors to Q8_0/BF16.
4.  Map the sidecar's tensor names to the MTP runtime's expected names.
5.  Validate the conversion by comparing the dequantized values to the original values.

This is a **major** change, not a simple dispatch. The proposal must be revised to reflect this.
