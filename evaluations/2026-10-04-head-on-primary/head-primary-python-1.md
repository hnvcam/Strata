Based on the provided source code and product facts, here is the proposal for integrating the external Q4_K_M MTP sidecar into Strata.

### 1. Current Incompatibility Analysis

The current Strata MTP runtime (`tools/mtp_rt.py`, `src/core/mtp.cpp`) is strictly coupled to a **Q2_0** expert format and a specific **Q8_0/BF16** dense layout. The external sidecar uses **Q4_K** for experts and mixed quantizations for dense layers, creating three primary incompatibilities:

1.  **Expert Blob Layout & Format Mismatch**:
    *   **Current**: `tools/mtp_rt.py` (lines 48–58) expects `gate_up_proj` and `down_proj` in **Q2_0** (18 bytes per 64 values). It manually interleaves gate/up rows and separates codes/scales into a custom `experts.bin` blob defined in `include/strata/kernels/cpu/expert.hpp` (lines 35–51).
    *   **Sidecar**: The sidecar provides `ffn_gate_exps.weight` and `ffn_up_exps.weight` as separate tensors in **Q4_K** (144 bytes per 256 values). `ffn_down_exps.weight` is **Q8_0**.
    *   **Conflict**: The current loader (`src/core/mtp.cpp` lines 180–200) allocates a fixed-size buffer based on `strata::kernels::cpu::BLOB` (Q2_0 size) and expects the custom interleaved layout. It cannot read Q4_K blocks or the separate gate/up/down structure.

2.  **Dense Tensor Quantization & Naming**:
    *   **Current**: `tools/mtp_rt.py` (lines 29–31, 92–96) converts specific dense tensors to **Q8_0** and others to **BF16**. It expects specific names like `self_attn.q_proj.weight`.
    *   **Sidecar**: Uses names like `blk.48.attn_q_norm` and `blk.48.ffn_gate_shexp.weight`. Quantizations vary: Q4_K, Q6_K, Q8_0, Q5_0, and F32.
    *   **Conflict**: The runtime (`src/core/mtp.cpp` lines 201–204) hard-checks for `q8_0` types for specific tensors. It does not support Q4_K/Q6_K/Q5_0 for dense projections in the current MTP path.

3.  **Architecture & Tensor Naming**:
    *   **Current**: Expects `mtp.layers.0.*` names.
    *   **Sidecar**: Uses `blk.48.*` names (e.g., `blk.48.ffn_gate_exps.weight`).
    *   **Conflict**: The loader in `tools/mtp_rt.py` (lines 82–83) strips `mtp.layers.0.` prefixes. It will fail to find tensors named `blk.48.*`.

### 2. Proposed Conversion Strategy

A new conversion tool (or a heavily modified `mtp_rt.py`) is required to bridge the sidecar to the runtime. This tool must **not** attempt to force Q4_K into the Q2_0 blob format. Instead, it should leverage the existing **Native Expert** support found in `include/strata/kernels/iq_kernels.hpp`.

#### Step A: Expert Conversion (Native Format)
The sidecar's Q4_K experts are compatible with Strata's **Native Expert** kernels (`native_expert_grouped`), which support Q4_K/Q5_K/Q8_0 (see `iq_kernels.hpp` lines 2–3, 40–48).

1.  **Extract Experts**: Read `blk.48.ffn_gate_exps.weight` (Q4_K), `blk.48.ffn_up_exps.weight` (Q4_K), and `blk.48.ffn_down_exps.weight` (Q8_0) from the GGUF.
2.  **Layout Transformation**:
    *   The sidecar stores gate and up as separate tensors. The `NativeExpertLayout` (see `iq_kernels.cu` lines 1423–1435) expects a single blob per expert with `[gate rows | up rows | down rows]`.
    *   For each expert $e$ (0–511):
        *   Extract the $e$-th slice of gate and up weights.
        *   Concatenate: `gate_data` + `up_data` + `down_data`.
        *   Ensure row alignment matches `iq_row_bytes` for Q4_K and Q8_0.
3.  **Output**: Write these concatenated blobs to `experts.bin`. **Note**: The size will be significantly larger than the Q2_0 blob.

#### Step B: Dense Tensor Conversion
The runtime currently expects Q8_0/BF16. To support the sidecar's mixed quantization without rewriting the entire MTP forward pass in C++, we must **dequantize** or **re-quantize** dense tensors to the formats the current MTP loader expects, OR extend the loader to support native types.

*Recommendation: Extend the Loader (Preferred for Performance)*
Instead of converting dense tensors to Q8_0 (which loses precision and adds conversion overhead), modify `src/core/mtp.cpp` to support native types for dense layers, similar to how the main model handles them.

1.  **Mapping**: Create a mapping from sidecar names (`blk.48.attn_q_norm`) to runtime names (`self_attn.q_norm`).
2.  **Quantization Handling**:
    *   **Q4_K/Q6_K/Q5_0/Q8_0**: Pass these directly to the engine's MMVQ kernels. The `iq_mmvq` function (see `iq_kernels.hpp` line 26) supports these types.
    *   **F32**: Keep as F32.
    *   **Norms**: The sidecar norms are F32 with Gemma +1 offset already applied (per product facts). The current `mtp_rt.py` (line 86) adds +1.0. The new converter must **skip** adding +1.0 if the source already has it, or verify the source format. *Product facts state: "source GGUF norm scales already include their Gemma +1 offset."* Therefore, the converter must **not** add 1.0.

#### Step C: Metadata & Index Generation
Generate `dense.txt` with updated `kind` fields:
*   `q4_k`, `q6_k`, `q5_0`, `q8_0`, `f32`, `bf16`.
*   Update `mtp.cpp` to parse these new kinds.

### 3. Runtime Changes Required

1.  **Loader (`src/core/mtp.cpp`)**:
    *   **Expert Loading**: Replace the fixed `BLOB` size calculation with dynamic sizing based on `native_expert_layout` (see `iq_kernels.cu` lines 1423–1435). The loader must detect the expert type (Q4_K/Q8_0) from the GGUF metadata or a new header in `experts.bin`.
    *   **Dense Loading**: Modify the `Tensor` struct and loader to handle `q4_k`, `q6_k`, etc., instead of just `q8_0`/`bf16`/`f32`.
    *   **Kernel Dispatch**: Ensure `moe_grouped_s2` (line 533) and `shared_expert` (line 543) can accept native expert pointers and types. The current code hardcodes `GGML_Q8_0` for shared experts (line 536). This must be dynamic.

2.  **Kernel Support**:
    *   Verify `native_expert_supported` (see `iq_kernels.cu` lines 1417–1421) returns true for `gu_type=Q4_K` and `d_type=Q8_0` with `n_embd=2560`, `n_ff=640`.
    *   Verify `iq_mmvq` supports Q4_K for dense projections.

3.  **Hyper-Connections**:
    *   The sidecar uses Q4_K/Q6_K for HC down/inject and Q5_0 for up. The current runtime likely expects BF16 or Q8_0. The loader must support these types for HC tensors.

### 4. Reusable Helpers & Kernels

*   **`native_expert_grouped`** (`iq_kernels.hpp` line 59): Already supports Q4_K/Q5_K/Q8_0 experts. This is the key to avoiding a full rewrite of the expert path.
*   **`iq_mmvq`** (`iq_kernels.hpp` line 26): Supports Q4_K/Q5_K/Q8_0 for dense matrix-vector products.
*   **`GGUFFile`** (`tools/gguf_reader.py`): Can be used to parse the sidecar header and tensor offsets.
*   **`iq_dequant_gu_f16`** (`iq_kernels.hpp` line 37): If the prompt path requires FP16 experts, this can dequantize Q4_K gate/up to FP16. However, the decode path should use native Q4_K for speed.

### 5. Compatibility Checks

1.  **Dimension Check**:
    *   Sidecar: `n_embd=2560`, `n_ff=640`, `n_experts=512`.
    *   Runtime: `H=2560`, `FF=640`, `NE=512` (see `expert.hpp` lines 35–36).
    *   **Result**: Compatible.

2.  **Block Alignment**:
    *   Q4_K block size: 256 elements.
    *   `n_embd=2560` is divisible by 256.
    *   `n_ff=640` is divisible by 256? **No**. 640 / 256 = 2.5.
    *   **Critical Issue**: Q4_K requires the inner dimension to be a multiple of 256. The sidecar's `ffn_down_exps.weight` has shape `[640, 2560, 512]`. The inner dimension for the down projection is 640.
    *   **Check**: `native_expert_supported` (line 1419) checks `n_ff % qd == 0`. For Q8_0 (down), `qd=32`. 640 % 32 == 0. OK.
    *   For Q4_K (gate/up), `qg=256`. `n_embd=2560`. 2560 % 256 == 0. OK.
    *   **Conclusion**: The dimensions are compatible with the native kernels.

3.  **Norm Offset**:
    *   Sidecar norms have +1 offset.
    *   Current `mtp_rt.py` adds +1.0.
    *   **Action**: The new converter must **not** add +1.0.

### 6. Memory Implications

*   **Current Q2_0**: ~0.71 GB for experts (see `mtp_pack.py` line 15).
*   **Sidecar Q4_K/Q8_0**:
    *   Gate/Up (Q4_K): 2560 * 640 * 512 * (4.5/8) bytes ≈ 0.47 GB.
    *   Down (Q8_0): 640 * 2560 * 512 * (1.0/8) bytes ≈ 0.10 GB.
    *   Total Experts: ~0.57 GB.
    *   Dense Tensors: Mixed Q4_K/Q6_K/Q8_0. Likely ~0.2–0.3 GB.
    *   **Total VRAM**: ~0.8–0.9 GB.
*   **Comparison**: The Q4_K sidecar is **larger** than Q2_0 (0.71 GB) but smaller than Q8_0 (2.67 GB).
*   **PC Constraint**: RTX 5070 Ti (16 GB) + RTX 4060 (8 GB).
    *   The drafter is loaded on the last stage (see `generate.cpp` line 2293).
    *   If the main model is split, the drafter resides on the second GPU (4060, 8 GB).
    *   0.9 GB for the drafter is well within the 8 GB limit, leaving room for KV cache and the main model's last layer.

### 7. Validation Needed

1.  **Bitwise Correctness**:
    *   Compare the output of the new Q4_K MTP path against a reference implementation (e.g., llama.cpp with the same sidecar) for a fixed prompt.
    *   Verify that the `+1` offset handling in norms matches the reference.

2.  **Acceptance Rate**:
    *   Measure the draft acceptance rate on a standard benchmark (e.g., HumanEval, GSM8K).
    *   Compare Q4_K vs. Q2_0. Q4_K should have higher acceptance but potentially lower speed if the kernel is not optimized.

3.  **Kernel Performance**:
    *   Profile `native_expert_grouped` with Q4_K/Q8_0.
    *   Ensure the Q4_K MMVQ kernels are utilized for dense layers.

4.  **Memory Stability**:
    *   Run a long-context generation to ensure no VRAM leaks in the new loader.

### 8. What Cannot Be Concluded Without Testing

1.  **Performance Gain**: While Q4_K has higher precision, the `native_expert_grouped` kernel for Q4_K may be slower than the specialized Q2_0 VNNI kernel. The net speedup (acceptance rate vs. draft latency) is unknown.
2.  **Numerical Stability**: The interaction between Q4_K experts and Q8_0 down projections in the MTP layer may introduce different error profiles than Q2_0. This could affect the final output quality.
3.  **Kernel Coverage**: It is unverified if `native_expert_grouped` is fully optimized for the specific `n_embd=2560`, `n_ff=640` dimensions with Q4_K. It may fall back to a slower generic path.
4.  **Sidecar Integrity**: The product facts state the sidecar is "observed format." Without hashing and verifying the sidecar against a known good source, we cannot rule out corruption or non-standard quantization parameters (e.g., non-standard scale rounding).

### Summary of Proposal

1.  **Do not** use `tools/mtp_rt.py` as-is.
2.  **Create** a new converter `tools/mtp_rt_native.py` that:
    *   Reads the Q4_K/Q8_0 sidecar.
    *   Maps `blk.48.*` names to runtime names.
    *   Concatenates gate/up/down into native expert blobs.
    *   Preserves Q4_K/Q6_K/Q5_0/Q8_0/F32 types for dense tensors.
    *   **Does not** add +1.0 to norms.
3.  **Modify** `src/core/mtp.cpp` to:
    *   Support dynamic expert blob sizes and types.
    *   Support native quantization types for dense tensors.
    *   Use `native_expert_grouped` for experts.
4.  **Validate** acceptance rate and VRAM usage on the target hardware.
