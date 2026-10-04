# Proposal: Importing the Q4_K_M MTP Sidecar into Strata

## 1. Executive Summary

The supplied `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` is **not directly compatible** with the existing Strata MTP runtime. The current runtime (`tools/mtp_rt.py` + `src/core/mtp.cpp`) is hard-coded for a **Q2_0 expert blob layout** and a specific set of dense tensor types (Q8_0, BF16, F32). The sidecar uses **Q4_K** for gate/up experts and **Q8_0** for down experts, with a different tensor naming convention (`blk.48.*` vs `mtp.layers.0.*`) and a different architecture string (`qwen4exp` vs `qwen4exp-mtp`).

To support this sidecar **while retaining Q2 MTP support**, Strata must:
1.  Extend the Python tooling to parse the new GGUF naming and quantization types.
2.  Introduce a **new expert blob layout** for Q4_K/Q8_0 experts (or reuse the existing `native_expert_grouped` path if the engine already supports it for MTP).
3.  Update the C++ MTP loader to detect the expert format and dispatch to the correct kernel.
4.  Ensure the dense tensors (attention, shared expert, hyper-connections) are converted to formats the MTP runtime supports (Q8_0 or BF16).

**Critical Constraint:** The existing `moe_grouped_s2` kernel (used for Q2_0 experts) **cannot** process Q4_K experts. The `native_expert_grouped` kernel (used for IQ4_XS/IQ2_XS main model experts) **can** process Q4_K/Q8_0, but the MTP runtime currently does not use it for the draft layer.

---

## 2. Current Incompatibility (Concrete Source References)

### 2.1 Expert Format Mismatch
- **Current Runtime:** `tools/mtp_rt.py` (lines 48–58) constructs `experts.bin` using `blob_of()`, which expects Q2_0 blocks (18 bytes per 64 weights). The C++ loader (`src/core/mtp.cpp`, line 182) allocates `g.n_expert * strata::kernels::cpu::BLOB` bytes, where `BLOB` is defined in `include/strata/kernels/cpu/expert.hpp` (line 45) as `3 * (H * FF * BB / QK)` = 1,382,400 bytes per expert. This layout is specific to Q2_0.
- **Sidecar:** The sidecar uses Q4_K (144 bytes per 256 weights) for `ffn_gate_exps` and `ffn_up_exps`, and Q8_0 (34 bytes per 32 weights) for `ffn_down_exps`. The byte count per expert is vastly different:
  - Q4_K gate/up: `2560 * 640 / 256 * 144 * 2` (gate + up) = 2,304,000 bytes.
  - Q8_0 down: `640 * 2560 / 32 * 34` = 1,382,400 bytes.
  - Total per expert: ~3.68 MB vs. 1.38 MB for Q2_0.
- **Kernel Mismatch:** `src/core/mtp.cpp` (line 530) calls `moe_grouped_s2`, which is hard-coded for the Q2_0 blob layout. It does not support Q4_K.

### 2.2 Tensor Naming and Structure
- **Current Runtime:** `tools/mtp_rt.py` (lines 70–71) expects tensors named `mtp.layers.0.mlp.experts.gate_up_proj` and `mtp.layers.0.mlp.experts.down_proj`. The dense tensors are named `mtp.layers.0.self_attn.*`, etc.
- **Sidecar:** Tensors are named `blk.48.ffn_gate_exps.weight`, `blk.48.ffn_up_exps.weight`, `blk.48.ffn_down_exps.weight`, `blk.48.self_attn.*`, etc. The architecture is `qwen4exp`, not `qwen4exp-mtp`.
- **Implication:** The Python tooling must map the new names to the internal Strata names or the C++ loader must be updated to accept the new names.

### 2.3 Dense Tensor Quantization
- **Current Runtime:** `tools/mtp_rt.py` (lines 29–31) quantizes specific dense tensors to Q8_0 and keeps others as BF16. The C++ loader (`src/core/mtp.cpp`, lines 201–204) checks for Q8_0 tensors like `fc_embedding.weight`, `self_attn.q_proj.weight`, etc.
- **Sidecar:** The sidecar already has dense tensors in Q4_K, Q6_K, or Q8_0. The MTP runtime expects Q8_0 for the large projections (to use `mmvq` kernels) and BF16 for hyper-connections/router. If the sidecar's dense tensors are Q4_K, they **cannot** be used directly by the current MTP runtime, which only supports Q8_0, BF16, and F32 for dense tensors.

### 2.4 Hyper-Connection and Router
- **Current Runtime:** The MTP runtime expects hyper-connection weights in BF16 and router weights in F32.
- **Sidecar:** The sidecar has hyper-connection weights in Q4_K/Q6_K and router weights in F32. The Q4_K/Q6_K hyper-connection weights **cannot** be used directly by the current MTP runtime, which expects BF16.

---

## 3. Proposed Changes

### 3.1 Python Tooling: `tools/mtp_rt.py` Extension

**Goal:** Parse the Q4_K_M sidecar and produce `experts.bin` and `dense.bin` in a format the C++ loader can understand.

**Changes:**
1.  **Tensor Name Mapping:** Add a mapping from `blk.48.*` names to the internal Strata names (e.g., `blk.48.ffn_gate_exps.weight` -> `mtp.layers.0.mlp.experts.gate_up_proj`).
2.  **Expert Blob Layout:**
    -   **Option A (Recommended):** Use the existing `native_expert_grouped` kernel path. This requires the experts to be in the "native" layout: `[gate rows | up rows | down rows]` with raw GGUF blocks. The `NativeExpertLayout` struct in `include/strata/kernels/iq_kernels.hpp` (lines 41–47) already supports this. The Python tooling must pack the experts into this layout.
    -   **Option B:** Create a new blob layout for Q4_K/Q8_0 experts and a new kernel. This is more work and less likely to be correct.
3.  **Dense Tensor Conversion:**
    -   Convert Q4_K/Q6_K dense tensors to Q8_0 (using `q8_0` function in `tools/mtp_rt.py`).
    -   Convert Q4_K/Q6_K hyper-connection weights to BF16 (dequantize to F32, then quantize to BF16).
    -   Keep F32 norms as F32.
4.  **Metadata:** Write a `dense.txt` file with the tensor names, kinds, rows, cols, offsets, and bytes. The C++ loader reads this file.

**Example Pseudocode for Expert Packing (Option A):**
```python
# For each expert e:
#   gate = dequantize_q4_k(tens["blk.48.ffn_gate_exps.weight"][e])  # (2560, 640)
#   up   = dequantize_q4_k(tens["blk.48.ffn_up_exps.weight"][e])    # (2560, 640)
#   down = dequantize_q8_0(tens["blk.48.ffn_down_exps.weight"][e])  # (640, 2560)
#   # Pack into native layout:
#   #   gate rows: 2560 rows of 640 Q4_K blocks
#   #   up rows:   2560 rows of 640 Q4_K blocks
#   #   down rows: 640 rows of 2560 Q8_0 blocks
#   # Write to experts.bin
```

### 3.2 C++ Runtime: `src/core/mtp.cpp` Extension

**Goal:** Detect the expert format and dispatch to the correct kernel.

**Changes:**
1.  **Format Detection:** Read a metadata field from the GGUF (e.g., `strata.mtp.expert_format`) or infer it from the tensor types. If the format is `q4_k_m`, use the `native_expert_grouped` path.
2.  **Kernel Dispatch:**
    -   If `expert_format == "q2_0"`, use `moe_grouped_s2` (current path).
    -   If `expert_format == "q4_k_m"`, use `native_expert_grouped` (new path).
3.  **Memory Allocation:** Allocate `experts_` based on the new blob size (3.68 MB per expert for Q4_K/Q8_0).
4.  **Dense Tensor Loading:** The existing `dense.bin` loading code should work if the Python tooling produces the correct `dense.txt` and `dense.bin`.

**Example Pseudocode for Kernel Dispatch:**
```cpp
if (expert_format == "q2_0") {
    moe_grouped_s2(...);
} else if (expert_format == "q4_k_m") {
    NativeExpertLayout L = native_expert_layout(GGML_Q4_K, GGML_Q8_0, H, FF);
    native_expert_grouped(L, grp_ptr_, grp_start_, ..., xq_, scratch_, parts_, cs);
}
```

### 3.3 Reusable Helpers and Kernels

-   **`native_expert_grouped`:** Already supports Q4_K/Q8_0 experts (see `src/kernels/cuda/iq_kernels.cu`, lines 1417–1455). This kernel is used for the main model's IQ4_XS/IQ2_XS experts. It can be reused for the MTP draft layer.
-   **`iq_mmvq`:** Supports Q4_K/Q8_0 dense tensors. Can be used for the MTP's dense projections if they are converted to Q8_0.
-   **`q8_0` quantizer:** In `tools/mtp_rt.py`, can be used to convert Q4_K/Q6_K dense tensors to Q8_0.
-   **`dequant` functions:** In `tools/mtp_pack.py`, can be used to dequantize Q4_K/Q8_0 tensors to F32 for conversion to BF16.

### 3.4 Compatibility Checks

1.  **Tensor Names:** Verify that all required tensors are present in the sidecar.
2.  **Tensor Shapes:** Verify that the shapes match the expected dimensions (H=2560, FF=640, NE=512).
3.  **Quantization Types:** Verify that the expert tensors are Q4_K/Q8_0 and the dense tensors are Q4_K/Q6_K/Q8_0/BF16/F32.
4.  **Architecture:** Verify that the architecture is `qwen4exp`.

### 3.5 Memory Implications

-   **Expert Memory:** 512 experts * 3.68 MB = 1.88 GB. This is larger than the Q2_0 experts (0.71 GB) but fits in the RTX 5070 Ti 16 GB.
-   **Dense Memory:** The dense tensors are ~0.18 GB (BF16) or ~0.36 GB (Q8_0). This fits in the RTX 5070 Ti 16 GB.
-   **Total MTP Memory:** ~2.06 GB. This fits in the RTX 5070 Ti 16 GB, leaving ~14 GB for the main model and KV cache.

### 3.6 Validation Needed Before Activation

1.  **Unit Tests:**
    -   Test the Python tooling with the sidecar GGUF.
    -   Test the C++ loader with the new `experts.bin` and `dense.bin`.
    -   Test the `native_expert_grouped` kernel with Q4_K/Q8_0 experts.
2.  **Integration Tests:**
    -   Run the MTP draft layer with the new experts and compare the output to the Q2_0 draft layer.
    -   Verify that the draft acceptance rate is not significantly degraded.
3.  **Performance Tests:**
    -   Measure the latency of the MTP draft layer with the new experts.
    -   Compare the throughput to the Q2_0 draft layer.

---

## 4. What Cannot Be Concluded Without Testing

1.  **Draft Acceptance Rate:** It is unknown whether the Q4_K_M experts will produce drafts that are accepted by the target model at the same rate as the Q2_0 experts. Q4_K is a higher precision than Q2_0, so it is expected to be better, but this must be measured.
2.  **Performance:** It is unknown whether the `native_expert_grouped` kernel will be faster or slower than `moe_grouped_s2` for the MTP draft layer. The Q4_K dequantization is more expensive than Q2_0, but the higher precision may lead to better acceptance rates, which could offset the cost.
3.  **Memory Bandwidth:** It is unknown whether the larger expert memory (1.88 GB vs. 0.71 GB) will cause memory bandwidth bottlenecks on the RTX 5070 Ti.
4.  **Compatibility with Hyper-Connections:** It is unknown whether the Q4_K/Q6_K hyper-connection weights, when converted to BF16, will produce the same results as the original BF16 weights. This must be verified by comparing the output of the MTP draft layer with and without the hyper-connections.

---

## 5. Conclusion

The Q4_K_M MTP sidecar can be imported into Strata by extending the Python tooling to parse the new GGUF format and pack the experts into the `native_expert_grouped` layout, and by updating the C++ MTP loader to dispatch to the `native_expert_grouped` kernel. The existing Q2 MTP support can be retained by keeping the `moe_grouped_s2` path for Q2_0 experts. The memory implications are acceptable for the RTX 5070 Ti 16 GB. Validation is needed to confirm that the draft acceptance rate and performance are not degraded.
