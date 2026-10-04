Based on the provided source code and product facts, here is the analysis and proposal for integrating the external Q4_K_M MTP sidecar into Strata.

### 1. Current Incompatibility Analysis

The current Strata MTP pipeline is hard-coded to a specific legacy format (`Q2_0` experts, `Q8_0` dense weights) and a specific tensor naming convention. The external sidecar (`mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`) is incompatible for three primary reasons:

#### A. Expert Format and Layout Mismatch
*   **Current State:** `tools/mtp_rt.py` (lines 6–8, 48–58) expects experts in `Q2_0` format (type 42). It manually repacks these into a custom "blob" layout (`experts.bin`) defined in `include/strata/kernels/cpu/expert.hpp` (lines 35–51). This layout interleaves gate/up rows and separates codes from scales.
*   **Sidecar State:** The sidecar uses `Q4_K` (type 12) for `ffn_gate_exps` and `ffn_up_exps`, and `Q8_0` (type 8) for `ffn_down_exps`.
*   **Conflict:** The runtime loader in `src/core/mtp.cpp` (lines 180–200) blindly reads `experts.bin` assuming a fixed size `strata::kernels::cpu::BLOB` (1,382,400 bytes per expert). It does not inspect the GGUF tensor types. The `Q4_K` block size (144 bytes per 256 values) and `Q8_0` block size (34 bytes per 32 values) do not match the `Q2_0` geometry (18 bytes per 64 values) expected by the CPU blob generator or the GPU `native_expert_grouped` kernel if it were fed raw GGUF blocks without conversion.

#### B. Dense Weight Quantization and Naming
*   **Current State:** `tools/mtp_rt.py` (lines 9–11, 29–31) converts dense tensors to `Q8_0` (for MMVQ kernels) or keeps them as `BF16`/`F32`. It renames tensors from `mtp.layers.0.*` to short names like `self_attn.q_proj.weight`.
*   **Sidecar State:** The sidecar uses `Q4_K`, `Q6_K`, `Q8_0`, and `Q5_0` for various dense projections (attention, hyper-connections). The naming convention is `blk.48.*` (e.g., `blk.48.self_attn.q_proj.weight`).
*   **Conflict:** `src/core/mtp.cpp` (lines 201–204) explicitly checks for `q8_0` tensors with specific short names (`fc_embedding.weight`, `self_attn.q_proj.weight`, etc.). The sidecar's `Q4_K` attention weights will fail this check or be misinterpreted if loaded as `Q8_0`. The `Q5_0` hyper-connection weights are not currently supported by the `dense.bin` loader logic which only handles `q8_0`, `bf16`, and `f32`.

#### C. Tensor Naming and Structure
*   **Current State:** The pipeline assumes a single MTP layer named `mtp.layers.0.*`.
*   **Sidecar State:** The sidecar places the MTP layer at `blk.48.*` (the last layer of the 49-layer architecture). It also includes `nextn.*` tensors (e.g., `nextn.eh_proj.weight`) which are not present in the current `mtp.*` schema.
*   **Conflict:** `tools/mtp_rt.py` (lines 70–71) hard-codes tensor names: `mtp.layers.0.mlp.experts.gate_up_proj`. It will fail to find `blk.48.ffn_gate_exps.weight`.

### 2. Proposed Conversion and Runtime Changes

To support the sidecar without breaking existing Q2 support, a **dual-path loader** or a **pre-conversion step** is required. Given the constraint "Do not implement code changes," the proposal focuses on the necessary architectural adjustments.

#### Option A: Pre-Conversion to Legacy Format (Recommended for Stability)
Convert the sidecar into the existing `experts.bin`/`dense.bin` format using a new tool (e.g., `tools/mtp_rt_q4k.py`). This avoids modifying the C++ runtime.

1.  **Expert Conversion:**
    *   Read `blk.48.ffn_gate_exps.weight` (Q4_K) and `blk.48.ffn_up_exps.weight` (Q4_K).
    *   Read `blk.48.ffn_down_exps.weight` (Q8_0).
    *   **Dequantize** these to FP32.
    *   **Requantize** to `Q2_0` using the logic in `tools/mtp_pack.py` (lines 58–82) or `Q4_0` if higher precision is desired (but `Q2_0` matches the existing kernel expectations).
    *   **Repack** into the `experts.bin` blob layout defined in `include/strata/kernels/cpu/expert.hpp`.
    *   *Note:* This loses the precision advantage of Q4_K. If Q4_K precision is required, Option B is necessary.

2.  **Dense Conversion:**
    *   Map `blk.48.self_attn.*` to `self_attn.*`.
    *   Map `blk.48.mlp.shared_expert.*` to `mlp.shared_expert.*`.
    *   Map `blk.48.nextn.eh_proj.weight` to `fc_embedding.weight` (or a new name if the runtime supports it).
    *   **Quantize** all dense weights to `Q8_0` using `q8_0()` from `tools/mtp_rt.py` (lines 34–45) to match the `dense.bin` format.
    *   **Handle Norms:** Ensure `nextn.enorm`, `nextn.hnorm`, etc., are loaded as F32 with the `+1` offset applied (as done in `tools/mtp_rt.py` lines 85–87).

#### Option B: Native Q4_K Support (Higher Complexity)
Modify the runtime to accept native GGUF blocks for experts and dense weights.

1.  **Runtime Loader Changes (`src/core/mtp.cpp`):**
    *   Replace the fixed-size `experts.bin` read with a GGUF tensor reader.
    *   Detect tensor types (`Q4_K`, `Q8_0`) from the GGUF header.
    *   Use `strata::kernels::native_expert_grouped` (from `include/strata/kernels/iq_kernels.hpp`) which already supports `Q4_K` (type 12) and `Q8_0` (type 8) via `native_expert_supported` (lines 1417–1421).
    *   *Constraint:* The current `mtp.cpp` uses `moe_grouped_s2` (line 533) which expects the custom blob layout. It would need to switch to `native_expert_grouped` or a new kernel path for Q4_K experts.

2.  **Dense Weight Loader Changes:**
    *   Extend `dense.txt`/`dense.bin` to support `q4_k`, `q6_k`, `q5_0` kinds.
    *   Update `src/core/mtp.cpp` to load these types and pass them to `iq_mmvq` (from `iq_kernels.hpp`) which supports `Q4_K` (line 26).

### 3. Reusable Helpers and Kernels

*   **`tools/gguf_reader.py`:** Already supports `Q4_K` (type 12) and `Q8_0` (type 8) in `BLOCK_GEOMETRY` (lines 39–48). This can be used to validate the sidecar's structure.
*   **`include/strata/kernels/iq_kernels.hpp`:**
    *   `native_expert_supported` (line 51) confirms that `Q4_K` (gu_type) and `Q8_0` (d_type) are supported for grouped experts.
    *   `iq_mmvq` (line 26) supports `Q4_K` for dense matrix-vector products.
*   **`tools/mtp_pack.py`:** The `q8_0` quantizer (lines 100–108) can be reused to convert sidecar dense weights to the current `Q8_0` format.

### 4. Compatibility Checks

Before activation, the following checks must pass:

1.  **Tensor Name Mapping:**
    *   Verify `blk.48.ffn_gate_exps.weight` maps to expert gate.
    *   Verify `blk.48.nextn.eh_proj.weight` maps to the embedding projection.
    *   Verify `blk.48.self_attn.q_proj.weight` maps to attention Q.
2.  **Shape Validation:**
    *   `ffn_gate_exps`: `[2560, 640, 512]` (GGUF innermost first: 512 experts, 640 FF, 2560 H).
    *   `ffn_down_exps`: `[640, 2560, 512]`.
    *   `nextn.eh_proj`: `[5120, 2560]` (Concatenated Embedding + Hidden).
3.  **Quantization Type Check:**
    *   Ensure `ffn_gate_exps` and `ffn_up_exps` are `Q4_K` (12).
    *   Ensure `ffn_down_exps` is `Q8_0` (8).
    *   Ensure dense weights are compatible with `iq_mmvq` (Q4_K, Q6_K, Q8_0).

### 5. Memory Implications

*   **Current Q2 MTP:**
    *   Experts: 512 * 1.38 MB = ~707 MB.
    *   Dense: ~180 MB (Q8_0/BF16).
    *   Total VRAM: ~900 MB.
*   **Sidecar Q4_K MTP (Native):**
    *   Experts:
        *   Gate/Up: 512 * 2 * (2560 * 640 / 256) * 144 bytes = 512 * 2 * 10 * 144 = 1.47 GB.
        *   Down: 512 * (640 * 2560 / 32) * 34 bytes = 512 * 50 * 34 = 0.87 GB.
        *   Total Experts: ~2.34 GB.
    *   Dense:
        *   Attention/Hyper-connections in Q4_K/Q6_K/Q8_0. Roughly 1.5–2x larger than current Q8_0/BF16 mix.
    *   Total VRAM: ~4–5 GB.
*   **Impact on RTX 5070 Ti (16 GB):**
    *   The main model (IQ4_XS) likely uses 10–12 GB.
    *   Adding a 4–5 GB MTP drafter may exceed VRAM, forcing a layer split or causing OOM.
    *   **Mitigation:** Use `--split-skip-if-fits` (line 2139) to check if the drafter fits on the second GPU (RTX 4060 8 GB). The 4060 has 8 GB, which might be tight for a 4–5 GB drafter plus KV cache.

### 6. Validation Needed

1.  **Acceptance Rate:** Measure draft acceptance rate with Q4_K experts vs. Q2_0. Q4_K should yield higher acceptance but may be slower due to larger weight size.
2.  **Correctness:** Verify that `nextn.eh_proj` correctly concatenates embedding and hidden states. The current `fc_embedding` and `fc_hidden` are separate in the Q2 path. The sidecar combines them. The runtime must handle this combined projection.
3.  **Norm Offsets:** Confirm that `nextn.enorm` and `nextn.hnorm` include the Gemma `+1` offset. If not, the loader must add it (as done in `tools/mtp_rt.py` line 86).

### 7. What Cannot Be Concluded Without Testing

*   **Performance:** Whether Q4_K experts provide a net speedup over Q2_0 due to higher acceptance rates vs. memory bandwidth costs.
*   **VRAM Fit:** Whether the Q4_K MTP drafter fits on the RTX 4060 alongside the KV cache for the split configuration.
*   **Numerical Stability:** Whether the `Q4_K` experts introduce instability in the draft layer's logits, affecting the final output quality.
*   **Kernel Efficiency:** Whether `native_expert_grouped` with `Q4_K` is optimized for the specific `n_embd=2560, n_ff=640` dimensions of this model.

### Conclusion

The external sidecar is **not compatible** with the current Strata MTP runtime. The primary blocker is the hard-coded `Q2_0` expert format and `Q8_0` dense format in `tools/mtp_rt.py` and `src/core/mtp.cpp`.

**Recommended Path:**
1.  Write a new Python tool `tools/mtp_rt_q4k.py` that reads the sidecar GGUF.
2.  Dequantize Q4_K experts to FP32.
3.  Requantize to `Q2_0` (or `Q4_0` if runtime supports it) and repack into `experts.bin`.
4.  Convert dense weights to `Q8_0` and rename tensors to match `dense.txt` expectations.
5.  This avoids C++ changes and leverages existing kernels, but sacrifices the precision benefit of Q4_K.

If Q4_K precision is critical, C++ changes are required to switch the MTP loader to use `native_expert_grouped` with native GGUF blocks, which is a significant refactor.
