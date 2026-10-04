Based on the provided source code and product facts, here is the proposal for integrating the external Q4_K_M MTP sidecar into Strata while retaining legacy Q2 support.

### 1. Current Incompatibility Analysis

The current Strata MTP pipeline is hard-coded for a specific internal format (`Q2_0` experts, `Q8_0`/`BF16` dense) and a specific tensor naming convention. The external sidecar (`mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf`) is incompatible for three primary reasons:

**A. Expert Format and Layout Mismatch**
*   **Current State:** `tools/mtp_rt.py` (lines 6–8, 48–58) expects experts in `Q2_0` format (64 values/block, 18 bytes). It converts them into a custom "blob" layout (`experts.bin`) defined in `include/strata/kernels/cpu/expert.hpp` (lines 35–45). This layout interleaves gate/up rows and separates codes from scales for the CPU VNNI kernel.
*   **Sidecar State:** The sidecar uses `Q4_K` for gate/up (`ffn_gate_exps`, `ffn_up_exps`) and `Q8_0` for down (`ffn_down_exps`). `Q4_K` uses 256-value blocks (144 bytes), and `Q8_0` uses 32-value blocks (34 bytes).
*   **Conflict:** The runtime loader `src/core/mtp.cpp` (lines 180–200) allocates memory based on `strata::kernels::cpu::BLOB` (fixed size for Q2_0). It cannot load Q4_K/Q8_0 blocks. Furthermore, the GPU kernel `native_expert_grouped` (in `iq_kernels.hpp`) supports native GGUF layouts, but the current MTP path uses the custom CPU-blob path (`moe_grouped_s2` in `mtp.cpp` line 533), not the native GPU path.

**B. Dense Tensor Quantization and Naming**
*   **Current State:** `tools/mtp_rt.py` (lines 9–11, 29–31) converts specific dense tensors to `Q8_0` and others to `BF16`. It strips prefixes like `mtp.layers.0.` to create short names (e.g., `self_attn.q_proj.weight`). The runtime `src/core/mtp.cpp` (lines 201–204) explicitly checks for these short names and `q8_0` kind.
*   **Sidecar State:** The sidecar uses `Q4_K`, `Q6_K`, `Q8_0`, and `Q5_0` for dense tensors. It uses full names like `blk.48.self_attn.q_proj.weight`. It also includes `Q4_K` for `nextn.eh_proj.weight`, which is not in the current `Q8` list.
*   **Conflict:** The runtime expects `Q8_0` for attention/shared experts. Loading `Q4_K` weights into the current `q8()` accessor will fail or produce garbage. The naming convention (`blk.48.` vs `self_attn.`) requires a mapping layer.

**C. Hyper-Connection and Norm Handling**
*   **Current State:** `tools/mtp_rt.py` (lines 85–87) adds `+1.0` to F32 norm weights to match GemmaRMSNorm behavior. It keeps hyper-connection weights as `BF16`.
*   **Sidecar State:** The sidecar already includes the `+1` offset in F32 norms (per product facts). It uses `Q4_K`/`Q6_K` for hyper-connection down/inject and `Q5_0` for up matrices.
*   **Conflict:** Applying `+1.0` again would double the offset. The runtime `mtp.cpp` expects `BF16` for hyper-connections (via `bf16()` accessor), but the sidecar provides quantized formats (`Q4_K`, `Q5_0`).

### 2. Proposed Conversion and Runtime Changes

To support the sidecar without breaking Q2 support, we propose a **dual-path loader** strategy.

#### A. New Conversion Tool: `tools/mtp_rt_v2.py`
Instead of modifying `mtp_rt.py`, create a new tool that handles the sidecar format.

1.  **Input:** Accepts the sidecar GGUF.
2.  **Expert Handling:**
    *   Do **not** convert experts to the custom Q2 blob.
    *   Instead, extract the raw `Q4_K` and `Q8_0` blocks for `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps`.
    *   Reorder them into the `NativeExpertLayout` expected by `native_expert_grouped` (see `iq_kernels.hpp` lines 40–47). This layout is `[gate rows | up rows | down rows]` with raw GGUF blocks.
    *   Write these to a new file `experts_native.bin`.
3.  **Dense Tensor Handling:**
    *   Map sidecar names to Strata short names (e.g., `blk.48.self_attn.q_proj.weight` -> `self_attn.q_proj.weight`).
    *   **Norms:** Copy F32 norms directly (do **not** add +1.0, as they are pre-offset).
    *   **Dense Projections:** Keep `Q4_K`, `Q6_K`, `Q8_0`, `Q5_0` as-is. Do not force Q8_0.
    *   **Hyper-Connections:** Keep `Q4_K`/`Q5_0`/`Q6_K` as-is.
    *   Write to `dense_native.bin` and `dense_native.txt`. The `.txt` index must record the actual GGML type ID (e.g., 12 for Q4_K) instead of just "q8_0" or "bf16".

#### B. Runtime Loader Changes (`src/core/mtp.cpp`)
Modify `MTP::init` to detect the format.

1.  **Format Detection:**
    *   Check for `dense_native.txt`. If present, use the new path. If not, fall back to `dense.txt` (legacy Q2 path).
2.  **Dense Loading:**
    *   Parse `dense_native.txt`.
    *   Allocate `dense_` buffer.
    *   For each tensor, copy raw bytes from `dense_native.bin`.
    *   **Accessor Update:** The current `q8()` and `bf16()` accessors are insufficient. Introduce a generic `get_tensor(name)` that returns a pointer and type. Update `mtp.cpp` lines 526–545 to use this generic accessor.
    *   **Kernel Dispatch:**
        *   For `Q8_0`/`BF16`/`F32`: Use existing kernels (`bf16_gemv_fp32_mmvf`, etc.).
        *   For `Q4_K`/`Q6_K`/`Q5_0`: Use `iq_mmvq` (from `iq_kernels.hpp` line 26) for dense projections. This requires quantizing activations to `q8_1` (which `mtp.cpp` already does for experts, see line 532).
3.  **Expert Loading:**
    *   If `experts_native.bin` exists:
        *   Load raw GGUF blocks.
        *   Call `native_expert_grouped` (from `iq_kernels.hpp` line 59) instead of `moe_grouped_s2`.
        *   This requires setting up `NativeExpertLayout` with `gu_type=12` (Q4_K) and `d_type=8` (Q8_0).
        *   Verify `native_expert_supported(12, 8, 2560, 640)` returns true.
    *   If not, use legacy `experts.bin` and `moe_grouped_s2`.

#### C. Hyper-Connection Handling
*   The sidecar uses quantized hyper-connections. The current `mtp.cpp` uses `bf16()` for these.
*   **Change:** Modify the hyper-connection application logic in `mtp.cpp` (around line 550) to support quantized weights.
*   **Option 1 (Simpler):** Dequantize hyper-connection weights to `BF16` during loading (in `mtp_rt_v2.py`). This increases memory but simplifies runtime. Given the small size of HC weights, this is recommended.
*   **Option 2 (Complex):** Implement `iq_mmvq` calls for HC matrices in the runtime. This is more efficient but requires significant changes to the mixer logic.

### 3. Reusable Helpers and Kernels

*   **`iq_mmvq` (`iq_kernels.hpp` line 26):** Can be reused for all dense projections (attention, shared expert, router) if activations are quantized to `q8_1`.
*   **`native_expert_grouped` (`iq_kernels.hpp` line 59):** Can be reused for the Q4_K/Q8_0 experts. This is the key kernel that avoids the custom Q2 blob.
*   **`iq_dequant_f32` (`iq_kernels.hpp` line 30):** Can be used to dequantize hyper-connection weights to F32/BF16 at load time if Option 1 is chosen.
*   **`GGUFFile` (`tools/gguf_reader.py`):** Can be reused in `mtp_rt_v2.py` to parse the sidecar header and tensor offsets.

### 4. Compatibility Checks

1.  **Architecture Check:** Verify `general.architecture` is `qwen4exp` and `block_count` is 49.
2.  **Tensor Existence:** Ensure all required tensors (`self_attn.*`, `mlp.shared_expert.*`, `nextn.*`) exist in the sidecar.
3.  **Type Support:** Verify `native_expert_supported(12, 8, 2560, 640)` returns true. If not, the sidecar cannot be used with the current GPU kernels.
4.  **Norm Offset:** Verify that F32 norms in the sidecar are pre-offset. This can be checked by comparing a known norm value against the checkpoint (if available) or by assuming the product fact is correct.
5.  **Vocabulary:** Ensure the draft vocabulary (248,320 IDs) matches the target model. The sidecar is shared, so it relies on the target model's embedding/head.

### 5. Memory Implications

*   **Current Q2 MTP:**
    *   Experts: 512 * 1.38 MB = ~707 MB.
    *   Dense: ~180 MB (Q8_0/BF16).
    *   Total: ~887 MB.
*   **Proposed Q4_K_M MTP:**
    *   Experts:
        *   Gate/Up (Q4_K): 512 * 2 * (2560 * 640 / 256) * 144 bytes = 512 * 2 * 10 * 144 = ~1.47 GB.
        *   Down (Q8_0): 512 * (2560 * 640 / 32) * 34 bytes = 512 * 50 * 34 = ~870 MB.
        *   Total Experts: ~2.34 GB.
    *   Dense:
        *   Attention/Shared (Q4_K/Q6_K/Q8_0): Slightly larger than Q8_0 due to block overhead, but similar magnitude. ~200 MB.
        *   Hyper-Connections (Q4_K/Q5_0): Small.
    *   Total: ~2.5–2.6 GB.
*   **Impact:** The Q4_K_M sidecar requires **~1.6 GB more VRAM** than the Q2 version.
*   **PC Constraints:**
    *   RTX 5070 Ti (16 GB) + RTX 4060 (8 GB).
    *   The main model (IQ4_XS) likely uses ~10–12 GB on the 5070 Ti.
    *   Adding 2.6 GB for MTP might push the 5070 Ti to ~14–15 GB, leaving little room for KV cache.
    *   **Recommendation:** The MTP layer should be placed on the **RTX 4060 (8 GB)** if possible, or the main model's KV cache must be reduced. The `--layer-split` feature (see `serve/server.py` line 628) can help, but MTP is currently loaded on the last stage. If the last stage is the 4060, it must fit the MTP (2.6 GB) + head + KV. This is tight but feasible if the main model's last layers are on the 5070 Ti.

### 6. Validation Needed

1.  **Unit Test:** Create a small test GGUF with Q4_K experts and verify `native_expert_grouped` produces the same output as a reference dequantize-and-matmul.
2.  **Integration Test:** Load the sidecar and run a single forward pass. Compare logits against the Q2 MTP (if accuracy allows) or against a reference implementation (e.g., llama.cpp with MTP support).
3.  **Memory Profile:** Run `nvidia-smi` during initialization to confirm VRAM usage matches estimates.
4.  **Acceptance Rate:** Measure draft acceptance rate. Q4_K_M should have higher acceptance than Q2_0, but lower than BF16.

### 7. What Cannot Be Concluded Without Testing

1.  **Kernel Performance:** Whether `native_expert_grouped` with Q4_K/Q8_0 is faster or slower than the custom Q2 CPU-blob path on the RTX 5070 Ti/4060.
2.  **Accuracy Impact:** The exact impact of Q4_K quantization on draft acceptance rate compared to Q2_0.
3.  **VRAM Fit:** Whether the 2.6 GB MTP layer fits alongside the IQ4_XS main model and KV cache on the specific GPU configuration without OOM.
4.  **Norm Offset Correctness:** Whether the sidecar's F32 norms are truly pre-offset. If not, outputs will be incorrect.
5.  **Hyper-Connection Quantization Error:** Whether quantizing HC weights to Q4_K/Q5_0 introduces significant error in the mixer.

### Summary of Proposal

1.  **Do not modify** `mtp_rt.py` or the Q2 runtime path.
2.  **Create** `tools/mtp_rt_v2.py` to convert the sidecar to `dense_native.bin`/`experts_native.bin` with raw GGUF blocks and correct naming.
3.  **Modify** `src/core/mtp.cpp` to detect the new format and use `iq_mmvq` for dense tensors and `native_expert_grouped` for experts.
4.  **Dequantize** hyper-connection weights to BF16 at load time to simplify runtime.
5.  **Validate** VRAM usage and accuracy before activation.
