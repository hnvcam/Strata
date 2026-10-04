### Proposal: Q4_K_M MTP Sidecar Adaptation via Lossy Conversion

**Decision:** Implement a **lossy conversion pipeline** that transforms the external Q4_K_M sidecar into Strata’s native Q2_0 MTP format. Direct runtime integration of Q4_K experts is rejected due to architectural coupling in `src/core/mtp.cpp`.

#### 1. Affected Files
*   **New:** `tools/mtp_rt_from_sidecar.py` (or extend `tools/mtp_rt.py` with a `--source-format q4k_mtp` flag).
*   **Unchanged:** `src/core/mtp.cpp`, `include/strata/kernels/cpu/expert.hpp`, `src/kernels/cuda/iq_kernels.cu`. The C++ runtime remains untouched to preserve stability and Q2_0 optimizations.

#### 2. Conversion Design (`tools/mtp_rt_from_sidecar.py`)
This tool bridges the external GGUF to the internal `rt/` directory structure.

**A. Expert Tensor Processing (Q4_K/Q8_0 → Q2_0)**
1.  **Read:** Load `blk.48.ffn_gate_exps.weight` (Q4_K), `blk.48.ffn_up_exps.weight` (Q4_K), and `blk.48.ffn_down_exps.weight` (Q8_0) from the sidecar.
2.  **Dequantize:** Convert these tensors to FP32 using `gguf-py` or reference dequantizers.
3.  **Re-quantize:** Apply the existing `q2_0` quantizer from `tools/mtp_pack.py` (lines 58–82) to the FP32 data. This ensures bitwise compatibility with the Q2_0 kernel expectations.
4.  **Relayout:** Use the logic in `tools/mtp_rt.py` (lines 48–58) to interleave gate/up rows and separate codes/scales into the `experts.bin` blob format defined in `include/strata/kernels/cpu/expert.hpp` (lines 48–51).
5.  **Output:** Write `experts.bin` with 512 blobs of size `BLOB` (1,382,400 bytes).

**B. Dense Tensor Processing (Mixed → Q8_0/BF16)**
1.  **Name Mapping:** Map sidecar names (e.g., `blk.48.attn_q_norm`) to runtime names (e.g., `self_attn.q_norm`).
2.  **Dequantize:** Convert Q4_K, Q6_K, Q5_0, and Q8_0 dense tensors to FP32.
3.  **Re-quantize/Convert:**
    *   If the tensor is in the `Q8` list (see `tools/mtp_rt.py` lines 29–31), quantize to **Q8_0** using `q8_0()` (lines 34–45).
    *   Otherwise, convert to **BF16**.
4.  **Hyper-Connections:** Dequantize Q4_K/Q5_0 HC tensors to FP32 and convert to **BF16**.
5.  **Norms:**
    *   Sidecar norms include the Gemma `+1` offset.
    *   `tools/mtp_rt.py` adds `+1.0` (line 86).
    *   **Action:** Subtract `1.0` from the sidecar norms before writing them as F32 to `dense.bin`. This ensures the runtime receives the correct `1 + w` value after its internal addition.

**C. Metadata Generation**
*   Generate `dense.txt` with `kind` fields set to `q8_0`, `bf16`, or `f32` as appropriate.
*   Ensure offsets and byte counts match the new `dense.bin`.

#### 3. Runtime Compatibility
*   **Loader:** `src/core/mtp.cpp` (lines 153–200) reads `dense.txt` and `experts.bin`. Since the output format is identical to the current Q2_0 pipeline, no C++ changes are required.
*   **Kernels:** The existing Q2_0 kernels (`moe_grouped_s2`, `bf16_gemv_fp32_mmvf`) will process the converted data without modification.
*   **Checks:** The hard-coded checks in `src/core/mtp.cpp` (lines 201–204) for `q8_0` dense tensors will pass because the converter outputs Q8_0 for those specific tensors.

#### 4. Memory Implications
*   **Weights:** The converted Q2_0 experts will occupy ~0.71 GB (same as current Q2_0). Dense tensors will occupy ~0.18 GB (same as current).
*   **Overhead:** The `arena_` and `state_arena_` allocations in `src/core/mtp.cpp` (lines 206–284) are independent of quantization and remain unchanged (~100–200 MB depending on context).
*   **Total VRAM:** ~0.9–1.1 GB for the drafter. This fits comfortably on the RTX 4060 (8 GB) when the drafter is placed on the last stage (see `src/program/generate.cpp` line 2299).

#### 5. Validation Plan
1.  **Format Verification:** Use `tools/gguf_reader.py` to verify the sidecar header and tensor types. Verify the generated `experts.bin` size is exactly `512 * BLOB`.
2.  **Numerical Sanity:** Compare the dequantized Q2_0 output against the original Q4_K sidecar for a few random expert slices. Calculate relative RMS error. Expect higher error than Q4_K, but within Q2_0 norms.
3.  **Acceptance Rate:** Run a standard benchmark (e.g., HumanEval) with the converted MTP. Compare acceptance rate against the baseline Q2_0 MTP.
    *   *Hypothesis:* Acceptance may be slightly higher due to better source precision, but the Q2_0 quantization noise may negate this.
4.  **Stability:** Run long-context generation to ensure no VRAM leaks or kernel crashes.

#### 6. Remaining Uncertainties
1.  **Precision Gain:** It is unknown if the Q4_K source provides a meaningful acceptance rate improvement after being crushed back to Q2_0. If the gain is <1%, the conversion overhead is not justified.
2.  **Norm Offset:** The assumption that sidecar norms have `+1` and runtime adds `+1` must be verified by inspecting a few norm values. If the runtime expects raw weights, the converter must not subtract 1.0.
3.  **Kernel Efficiency:** The Q2_0 kernels are highly optimized. The conversion process adds CPU overhead during setup but no runtime overhead.

#### 7. Conclusion
This proposal prioritizes **correctness and stability** by leveraging the existing, well-tested Q2_0 runtime. It avoids the high risk and effort of rewriting the MTP C++ engine to support native Q4_K. The maintainer should implement the Python converter and validate the acceptance rate. If the acceptance rate does not improve significantly, the external sidecar should be discarded in favor of the native Q2_0 MTP.
