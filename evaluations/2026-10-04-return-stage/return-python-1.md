Based on the provided source code and product facts, here is the proposal for integrating the external Q4_K_M MTP sidecar into Strata.

### 1. Current Incompatibility Analysis

The current Strata MTP runtime (`tools/mtp_rt.py`, `src/core/mtp.cpp`) is hard-coded to ingest a specific intermediate format derived from a Q2_0 quantized GGUF. The external sidecar (`mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`) is incompatible for three primary reasons:

**A. Expert Format and Layout Mismatch**
*   **Current State:** `tools/mtp_rt.py` expects `mtp.layers.0.mlp.experts.gate_up_proj` and `down_proj` to be **Q2_0** (Type 42). It reads these tensors and repacks them into a custom binary blob (`experts.bin`) defined in `include/strata/kernels/cpu/expert.hpp`.
    *   `expert.hpp` (lines 35-45) defines `BLOB` size based on Q2_0 geometry: `3 * (H * FF * 18 // 64)`.
    *   `mtp_rt.py` (lines 48-58) performs a lossless relayout of Q2_0 blocks, interleaving gate/up rows and separating codes from scales.
*   **Sidecar State:** The sidecar uses **Q4_K** (Type 12) for gate/up and **Q8_0** (Type 8) for down.
    *   Q4_K has 256 values per 144-byte block. Q8_0 has 32 values per 34-byte block.
    *   The byte counts and internal structure (super-blocks, sub-blocks, scales) are fundamentally different from Q2_0. The current `blob_of` function in `mtp_rt.py` will fail assertions or produce garbage data because it assumes 18-byte blocks and specific scale/code interleaving.

**B. Dense Tensor Quantization Mismatch**
*   **Current State:** `tools/mtp_rt.py` (lines 9-11, 29-31) expects dense tensors (attention, shared expert, etc.) to be **BF16** or **F32** in the source GGUF. It then quantizes specific large projections to **Q8_0** for the engine's MMVQ kernels, while keeping others as BF16.
    *   `mtp.cpp` (lines 201-204) explicitly checks for `q8_0` kind for required tensors like `self_attn.q_proj.weight`.
*   **Sidecar State:** The sidecar already contains **Q4_K**, **Q6_K**, **Q8_0**, and **Q5_0** tensors.
    *   The sidecar's `ffn_gate_exps` are Q4_K, not BF16.
    *   The sidecar's `eh_proj` is Q4_K.
    *   The current loader `mtp_rt.py` does not know how to read Q4_K/Q6_K tensors from the GGUF and convert them into the `dense.bin` format expected by `mtp.cpp`. It assumes it can just cast BF16 to F32 or quantize BF16 to Q8_0.

**C. Tensor Naming and Structure**
*   **Current State:** `mtp_rt.py` expects tensor names like `mtp.layers.0.mlp.experts.gate_up_proj`.
*   **Sidecar State:** The sidecar uses names like `blk.48.ffn_gate_exps.weight`. The prefix `blk.48` and suffix `.weight` differ from the `mtp.layers.0...` convention. The sidecar also includes `nextn` prefixes for specific MTP layers (e.g., `blk.48.nextn.eh_proj.weight`).

### 2. Proposed Conversion and Runtime Changes

To support the sidecar without breaking existing Q2 support, a new conversion path is required. We cannot simply "add a 4-bit option" to the packer; the entire ingestion pipeline must branch.

#### Phase 1: New Converter Tool (`tools/mtp_rt_q4.py`)

A new tool is needed to read the external GGUF and produce the `experts.bin` and `dense.bin` files.

1.  **Expert Blob Generation:**
    *   **Reuse:** `include/strata/kernels/iq_kernels.hpp` and `src/kernels/cuda/iq_kernels.cu` already contain logic for `native_expert_layout` and `native_expert_grouped` which support Q4_K/Q8_0 pairs (see `iq_kernels.cu` lines 1417-1421).
    *   **Change:** The new converter must generate `experts.bin` in the **Native Expert Layout** defined in `iq_kernels.hpp` (lines 41-47), not the legacy Q2_0 blob layout.
    *   **Logic:**
        *   Read `blk.48.ffn_gate_exps.weight` (Q4_K) and `blk.48.ffn_down_exps.weight` (Q8_0).
        *   The sidecar stores gate and up separately? The spec says `ffn_gate_exps` and `ffn_up_exps` are separate tensors. The current Q2 pipeline fused them. The new converter must interleave them into the `NativeExpertLayout` structure: `[gate rows | up rows | down rows]`.
        *   Calculate offsets using `native_expert_layout` logic: `up_off = n_ff * gu_row`, `down_off = 2 * up_off`.
        *   Write raw Q4_K and Q8_0 blocks directly to `experts.bin`.

2.  **Dense Tensor Processing:**
    *   **Reuse:** `tools/gguf_reader.py` can parse the header.
    *   **Change:** Instead of quantizing BF16 to Q8_0, the converter must **pass through** the existing quantized tensors if they match the engine's expected types, or dequantize/requantize if necessary.
    *   **Mapping:**
        *   Map `blk.48.attn_q_norm` -> `self_attn.q_norm` (F32).
        *   Map `blk.48.attn_q_proj.weight` (Q4_K) -> `self_attn.q_proj.weight`.
        *   **Critical Decision:** Does `mtp.cpp` support Q4_K for dense tensors?
            *   `mtp.cpp` (line 204) checks for `q8_0`.
            *   `iq_kernels.hpp` (line 26) `iq_mmvq` supports Q4_K.
            *   **Proposal:** Modify `mtp.cpp` to accept `q4_k` kind in `dense.txt` and route to `iq_mmvq` instead of the Q8_0-specific MMVQ. Alternatively, the converter could dequantize Q4_K to F32 and requantize to Q8_0 to maintain compatibility with the existing `mtp.cpp` loader, but this loses precision and increases memory. **Recommendation:** Extend `mtp.cpp` to support Q4_K dense tensors natively, as the kernels already exist.

3.  **Metadata and Norms:**
    *   The sidecar norms (`enorm`, `hnorm`) are F32 and already include the Gemma +1 offset (per spec).
    *   `mtp_rt.py` (line 86) adds +1.0 to F32 norms. The new converter must **skip** this step for the sidecar, as the offset is pre-applied.

#### Phase 2: Runtime Loader Changes (`src/core/mtp.cpp`)

1.  **Format Detection:**
    *   Add a flag or metadata check in `dense.txt` or a new `mtp.meta` file to indicate the expert format (Q2_0 Legacy vs. Native Q4_K/Q8_0).
2.  **Expert Loading:**
    *   If Native format: Use `strata::kernels::native_expert_layout` to calculate blob sizes.
    *   Allocate `experts_` based on `NativeExpertLayout::bytes`.
    *   Load `experts.bin` directly into VRAM.
    *   Update `moe_group_resident` calls to use the native grouped expert kernel (`native_expert_grouped`) instead of the legacy `moe_grouped_s2` (which assumes Q2_0).
3.  **Dense Tensor Loading:**
    *   Update the `Tensor` struct parsing in `mtp.cpp` (line 142) to recognize `q4_k`, `q6_k`, `q5_0` kinds.
    *   Update the forward pass (lines 522-553) to dispatch to `iq_mmvq` for Q4_K tensors instead of `bf16_gemv_fp32_mmvf` or Q8_0-specific kernels.

### 3. Reusable Helpers and Kernels

*   **`include/strata/kernels/iq_kernels.hpp`**:
    *   `native_expert_layout`: Essential for calculating the new `experts.bin` structure.
    *   `native_expert_grouped`: The CUDA kernel to execute the Q4_K/Q8_0 experts.
    *   `iq_mmvq`: The CUDA kernel for Q4_K dense matrix multiplications.
*   **`tools/gguf_reader.py`**:
    *   Can be used to parse the sidecar header and verify tensor types/shapes before conversion.
*   **`src/kernels/cuda/iq_kernels.cu`**:
    *   Contains the dequantization and dot-product logic for Q4_K/Q8_0. No new CUDA kernels are needed, only wiring.

### 4. Compatibility Checks

1.  **Architecture Check:** Verify `general.architecture` is `qwen4exp` and `block_count` is 49.
2.  **Tensor Existence:** Ensure `blk.48.ffn_gate_exps.weight`, `blk.48.ffn_up_exps.weight`, and `blk.48.ffn_down_exps.weight` exist.
3.  **Type Verification:**
    *   Gate/Up must be Q4_K (Type 12).
    *   Down must be Q8_0 (Type 8).
    *   Dense tensors must be one of the supported types (Q4_K, Q6_K, Q8_0, F32).
4.  **Shape Verification:**
    *   Gate/Up: `[2560, 640, 512]` (Innermost first: 512 experts, 640 FF, 2560 H).
    *   Down: `[640, 2560, 512]`.
    *   `eh_proj`: `[5120, 2560]`.
5.  **Norm Offset Check:** Verify that norm tensors are F32 and assume pre-offset. (Cannot be checked programmatically without reference data, so rely on file naming/metadata).

### 5. Memory Implications

*   **Current Q2_0 MTP:**
    *   Experts: ~0.71 GB (Q2_0).
    *   Dense: ~0.18 GB (Q8_0/BF16).
    *   Total VRAM: ~0.9 GB.
*   **Proposed Q4_K_M MTP:**
    *   Experts:
        *   Gate/Up (Q4_K): $512 \times 2 \times 640 \times 2560 \times (4.5/8) \approx 1.47$ GB. (Q4_K is ~4.5 bits/weight).
        *   Down (Q8_0): $512 \times 2560 \times 640 \times (8/8) \approx 0.84$ GB.
        *   Total Experts: ~2.31 GB.
    *   Dense:
        *   If kept as Q4_K/Q6_K: ~0.15 GB.
        *   If requantized to Q8_0: ~0.25 GB.
    *   Total VRAM: ~2.5 - 2.6 GB.
*   **Impact on RTX 5070 Ti (16 GB) + RTX 4060 (8 GB):**
    *   The increase is ~1.6 GB.
    *   `generate.cpp` (line 2299) reserves `kDrafterMib = 1000` (1 GB). This will need to be increased to ~2.6 GB.
    *   On the 8 GB card (if used for drafter), this is a significant portion. If the drafter is on the 16 GB card, it is negligible.
    *   **Risk:** If the drafter is placed on the 8 GB card, the available VRAM for KV cache and main model layers will shrink by 1.6 GB, potentially reducing context length or forcing more offloading.

### 6. Validation Needed Before Activation

1.  **Bitwise/Statistical Parity:**
    *   Run the Q2_0 MTP and the new Q4_K_M MTP on the same prompt.
    *   Compare draft acceptance rates. Q4_K should have higher acceptance than Q2_0.
    *   Compare output logits of the draft layer against a reference implementation (e.g., llama.cpp) to ensure the Q4_K dequantization and dot products are correct.
2.  **Memory Leak Check:**
    *   Verify that `native_expert_grouped` scratch memory is properly allocated and freed.
3.  **Kernel Dispatch:**
    *   Ensure that `mtp.cpp` correctly identifies the format and does not attempt to run Q4_K data through Q2_0 kernels.
4.  **Norm Handling:**
    *   Verify that the Gemma +1 offset is not double-applied.

### 7. What Cannot Be Concluded Without Testing

1.  **Performance Impact:**
    *   Q4_K dequantization is more complex than Q2_0. The `native_expert_grouped` kernel may be slower than the highly optimized Q2_0 VNNI/CUDA kernels. The net speedup from higher acceptance rates vs. slower draft generation is unknown.
2.  **Exact VRAM Footprint:**
    *   The calculation above is theoretical. Actual VRAM usage depends on CUDA context overhead, alignment padding in `NativeExpertLayout`, and temporary buffers for `iq_mmvq`.
3.  **Numerical Stability:**
    *   Q4_K uses super-blocks with sub-block scales. The interaction with the MTP's specific activation patterns (e.g., `eh_proj` mixing embedding and hidden states) might reveal precision issues not seen in standard layers.
4.  **Sidecar Completeness:**
    *   The spec mentions "unused indexer tensors." It is unclear if the current `mtp.cpp` expects these to be present (even if ignored) or if their absence causes a crash during tensor lookup.

### Summary of Proposal

1.  **Do not modify** `tools/mtp_pack.py` or `tools/mtp_rt.py` for the Q2 path.
2.  **Create** `tools/mtp_rt_q4.py` to convert the external GGUF to `experts.bin` (Native Layout) and `dense.bin` (Pass-through Q4_K/Q6_K).
3.  **Modify** `src/core/mtp.cpp` to:
    *   Detect format (Q2 vs Native).
    *   Allocate memory using `native_expert_layout`.
    *   Dispatch to `native_expert_grouped` and `iq_mmvq`.
4.  **Update** `generate.cpp` to increase `kDrafterMib` reservation when Q4_K_M is detected.
5.  **Validate** acceptance rates and VRAM usage on the target hardware.
