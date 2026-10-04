### Final Proposal: Native Q4_K_M MTP Sidecar Integration

**Objective:** Enable Strata to load and execute the external `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` sidecar, preserving its native **Q4_K** (Gate/Up) and **Q8_0** (Down) expert quantization, while retaining full support for the legacy Q2_0 runtime.

#### 1. Affected Files

*   **New:** `tools/mtp_rt_native.py` (Conversion tool for native GGUF sidecars)
*   **Modified:** `src/core/mtp.cpp` (Loader logic, kernel dispatch, activation quantization)
*   **Modified:** `include/strata/core/mtp.hpp` (Struct updates for dynamic tensor types)
*   **Unchanged:** `tools/mtp_rt.py`, `tools/mtp_pack.py` (Legacy Q2_0 path remains intact)

#### 2. Conversion Design (`tools/mtp_rt_native.py`)

This tool reads the external GGUF and produces `dense_native.bin`, `experts_native.bin`, and `dense.txt`. It must handle format-specific transformations that the legacy tool does not.

**A. Expert Extraction (`experts_native.bin`)**
*   **Source:** `blk.48.ffn_gate_exps.weight` (Q4_K), `blk.48.ffn_up_exps.weight` (Q4_K), `blk.48.ffn_down_exps.weight` (Q8_0).
*   **Layout:** `native_expert_grouped` expects a contiguous blob per expert: `[Gate Blocks] [Up Blocks] [Down Blocks]`.
*   **Action:**
    1.  Read raw Q4_K blocks for Gate and Up.
    2.  Read raw Q8_0 blocks for Down.
    3.  Concatenate them into a single binary blob per expert.
    4.  **Constraint:** Verify `native_expert_supported(12, 8, 2560, 640)` returns true. If the CUDA kernel does not support mixed Q4_K/Q8_0, this proposal fails (re-quantization is forbidden).
    5.  Write to `experts_native.bin`.

**B. Dense Weight Processing (`dense_native.bin`)**
*   **Norms (F32):**
    *   **Critical:** The sidecar norms *already include* the Gemma `+1` offset. Do **not** add `1.0` (unlike `tools/mtp_rt.py` line 86).
    *   Write as F32.
*   **Router (`mlp.gate.weight`):**
    *   **Source:** F32 in sidecar.
    *   **Target:** BF16 (to match `src/core/mtp.cpp` line 526 `bf16_gemv_fp32_mmvf`).
    *   **Action:** Convert F32 -> BF16.
*   **Hyper-Connections (`hc_*`):**
    *   **Source:** Q4_K/Q6_K/Q5_0.
    *   **Target:** BF16.
    *   **Action:** Dequantize to F32, then convert to BF16. This avoids modifying the runtime to support dynamic types for these small, non-bottleneck tensors.
*   **Attention & Shared Expert (`self_attn.*`, `mlp.shared_expert.*`):**
    *   **Source:** Q4_K, Q6_K, Q8_0.
    *   **Target:** Preserve original GGML types (Q4_K, Q8_0) if `iq_mmvq` supports them.
    *   **Action:** Extract raw blocks. Ensure 256-byte alignment in `dense_native.bin`.
*   **`eh_proj` Splitting:**
    *   **Source:** `blk.48.nextn.eh_proj.weight` [5120, 2560] Q4_K.
    *   **Target:** `fc_embedding.weight` [2560, 2560] and `fc_hidden.weight` [2560, 2560].
    *   **Action:** Split the Q4_K tensor along the inner dimension (2560). Since Q4_K blocks are 256 values, the split at 2560 is block-aligned. Extract the first 10 blocks per row for `fc_embedding` and the next 10 for `fc_hidden`. Write as Q4_K.

**C. Index Generation (`dense.txt`)**
*   Format: `name kind rows cols offset bytes`
*   **Kind:** Use string identifiers (`q4_k`, `q8_0`, `bf16`, `f32`) instead of integers to maintain parser compatibility with minor modifications.
*   **Offset:** Must reflect the 256-byte aligned offset in `dense_native.bin`.

#### 3. Runtime Design (`src/core/mtp.cpp`)

The loader must detect the "Native" format and switch dispatch logic.

**A. Detection**
*   Check for `dense_native.bin` existence or a marker in `dense.txt` (e.g., `kind` starts with `q4_`).

**B. Loader Updates**
*   **Parsing:** Update `Tensor` struct parsing to handle new `kind` strings.
*   **Memory:** Allocate `dense_native.bin` and `experts_native.bin` separately from legacy files.
*   **Expert Loading:**
    *   Calculate size using `native_expert_layout` (from `include/strata/kernels/iq_kernels.hpp`).
    *   Load into `experts_` pointer.
    *   **Kernel Switch:** Replace `moe_grouped_s2` (legacy Q2_0) with `native_expert_grouped` (from `include/strata/kernels/iq_kernels.hpp`).
    *   Pass `NativeExpertLayout` with `gu_type=12` (Q4_K) and `d_type=8` (Q8_0).

**C. Dense Kernel Dispatch**
*   **Generic Getter:** Implement `get_tensor_ptr(const std::string& name, int& type)` that returns a device pointer and GGML type ID.
*   **Attention/Shared Expert:**
    *   Replace hardcoded `q8()` calls with `get_tensor_ptr`.
    *   Update `iq_mmvq` calls to pass the dynamic `type`.
    *   **Verification:** Ensure `iq_mmvq` supports Q4_K for `n_embd=2560`. If not, the conversion tool must have converted these to Q8_0.
*   **Router/Hyper-Connections:**
    *   Continue using `bf16()` helper, as conversion tool output is BF16.

#### 4. Compatibility & Memory Considerations

**A. Hardware Impact (RTX 5070 Ti 16GB + RTX 4060 8GB)**
*   **VRAM Increase:**
    *   Legacy Q2_0 MTP: ~0.9 GB.
    *   Native Q4_K_M MTP: ~2.2 GB (Experts ~1.8 GB + Dense ~0.4 GB).
*   **Risk:** The 8 GB RTX 4060 will likely **fail** to load the drafter if the main model is also resident.
*   **Mitigation:** The drafter must be forced to the 16 GB card. Use `--layer-split` or `--split-skip-if-fits` logic to ensure the MTP layer resides on CUDA0 (5070 Ti).

**B. Kernel Compatibility**
*   **`native_expert_grouped`:** Must support mixed Q4_K (Gate/Up) and Q8_0 (Down). If the existing CUDA kernel assumes uniform types, the conversion tool must downsample Down to Q4_K.
*   **`iq_mmvq`:** Must support Q4_K for attention projections. If not, conversion tool must upsample to Q8_0.

#### 5. Validation Plan

1.  **Static Check:** Run `tools/gguf_reader.py` on the sidecar to verify tensor names and shapes match the conversion tool's expectations.
2.  **Conversion Sanity:**
    *   Verify `eh_proj` split produces two valid Q4_K tensors.
    *   Verify norms are **not** double-offset (check a sample value against sidecar raw data).
3.  **Runtime Smoke Test:**
    *   Load the native MTP on the 16 GB card.
    *   Run a single forward pass.
    *   **Critical:** Compare logits against a reference implementation (e.g., llama.cpp with the same sidecar) to ensure quantization/dequantization is correct.
4.  **Acceptance Rate Benchmark:**
    *   Measure draft acceptance rate vs. legacy Q2_0.
    *   Expect higher acceptance for Q4_K_M. If acceptance drops, investigate norm offsets or expert layout errors.

#### 6. Remaining Uncertainties

1.  **Kernel Support:** Does `native_expert_grouped` in `src/kernels/cuda/iq_kernels.cu` actually implement the mixed Q4_K/Q8_0 path? The header suggests it, but the kernel body is not fully visible.
2.  **`iq_mmvq` Performance:** Is Q4_K MMVQ optimized for `n_embd=2560`? If not, it may be slower than Q8_0.
3.  **Sidecar Naming:** Does the sidecar use `blk.48.` prefix for *all* tensors? The conversion tool must handle potential variations (e.g., `nextn.` prefix on some tensors).
4.  **Memory Fragmentation:** The 2.2 GB drafter may fragment VRAM on the 16 GB card, reducing KV cache slots. Monitor `cudaMemGetInfo` during load.

#### 7. Conclusion

This proposal requires a new conversion tool and moderate runtime changes to support dynamic tensor types and native expert layouts. It avoids modifying the legacy Q2_0 path. The primary risk is VRAM pressure on the 8 GB card and potential kernel gaps for mixed-precision experts. Implementation should proceed only after verifying `native_expert_grouped` supports Q4_K/Q8_0 mixes.
