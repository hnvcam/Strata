This is a review of the previous proposal from the perspective of a skeptical Strata maintainer. The proposal contains several critical architectural misunderstandings regarding how Strata’s engine handles quantization, memory layout, and tensor naming. It proposes a "dual-path" loader that ignores the fundamental design constraint of the current MTP runtime: **it is not a generic GGUF runner; it is a specialized, pre-compiled kernel host for a specific binary format.**

Here are the concrete mistakes, missing requirements, and corrections.

### 1. Critical Mistake: Misunderstanding `dense.bin` and `iq_mmvq` Integration

**The Proposal Claims:**
> "Update `dense.txt` to include the actual GGML type ID... Replace calls to `q8()`... with a generic `get_tensor(type, offset)`... Update kernel calls (e.g., `iq_mmvq`) to pass the correct `ggml_type` for each tensor."

**The Reality (Source Evidence):**
*   **`src/core/mtp.cpp` lines 201–204:** The loader explicitly checks for the existence of specific tensors using a helper `q8(name)`. This helper likely assumes the tensor is stored in Q8_0 format within `dense.bin`.
*   **`src/core/mtp.cpp` lines 536–538:** The shared expert weights are hardcoded:
    ```cpp
    nsw.gate_type = GGML_Q8_0; nsw.gate_data = q8("mlp.shared_expert.gate_proj.weight");
    nsw.up_type = GGML_Q8_0;   nsw.up_data = q8("mlp.shared_expert.up_proj.weight");
    nsw.down_type = GGML_Q8_0; nsw.down_data = q8("mlp.shared_expert.down_proj.weight");
    ```
    The code **hardcodes** `GGML_Q8_0`. It does not read the type from `dense.txt` for these critical paths.
*   **`include/strata/kernels/iq_kernels.hpp` line 26:** `iq_mmvq` takes `int ggml_type` as an argument. However, the *call sites* in `mtp.cpp` (not fully shown but implied by the `q8()` helper usage and hardcoded types) are likely optimized for Q8_0 or BF16.
*   **`tools/mtp_rt.py` lines 92–96:** The current packer *converts* BF16 dense weights to Q8_0. It does not preserve Q4_K.

**Correction:**
You cannot simply "pass the correct type." You must **rewrite the dense weight loading and execution logic** in `src/core/mtp.cpp` to:
1.  Parse the `kind` field in `dense.txt` as an integer GGML type ID (not just a string).
2.  Replace the hardcoded `GGML_Q8_0` assignments with dynamic types read from the index.
3.  Ensure that every kernel call site (attention, shared expert, router) correctly dispatches to `iq_mmvq` with the specific type ID.
4.  **Crucially:** Verify that `iq_mmvq` supports **Q4_K** for *all* dense tensors in the MTP layer. The header comment in `iq_kernels.hpp` says it supports Q4_K for "Unsloth's UD-Q4_K_XL", but we must verify if the *specific* shapes (e.g., `self_attn.q_proj` with head_dim 256) are supported by the existing CUDA kernels. If `iq_mmvq` lacks a Q4_K path for specific dimensions, the proposal fails.

### 2. Critical Mistake: Expert Layout and `native_expert_grouped` Compatibility

**The Proposal Claims:**
> "Write them to a new file `experts_native.bin` in a layout compatible with `strata::kernels::native_expert_grouped`... The layout for `native_expert_grouped` expects `[gate rows | up rows | down rows]` per expert, with raw GGUF blocks."

**The Reality (Source Evidence):**
*   **`include/strata/kernels/iq_kernels.hpp` lines 40–47:** `NativeExpertLayout` defines:
    ```cpp
    struct NativeExpertLayout {
        int gu_type = -1, d_type = -1;
        int64_t n_embd = 0, n_ff = 0;
        size_t gu_row = 0, d_row = 0;       // bytes per row
        size_t up_off = 0, down_off = 0;    // byte offsets inside the blob
        size_t bytes = 0;                   // the whole blob
    };
    ```
    And `native_expert_layout` (lines 1423–1435 in `iq_kernels.cu`) calculates:
    ```cpp
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    ```
    This implies the layout is: `[Gate Rows (n_ff * gu_row)] [Up Rows (n_ff * gu_row)] [Down Rows (n_embd * d_row)]`.
*   **Sidecar Specimen:** The sidecar has `ffn_gate_exps.weight` and `ffn_up_exps.weight` as **separate tensors**.
*   **Current `mtp_rt.py`:** It *interleaves* gate and up rows (line 52: `inter[0::2] = gub[:FF]`).
*   **The Conflict:** `native_expert_grouped` expects a **contiguous blob** where Gate and Up are separate blocks (Gate block, then Up block, then Down block). The sidecar provides them as separate GGUF tensors. The proposal says "extract... and write them to a new file." This is correct in principle, but the proposal glosses over the **memory alignment and stride requirements**.
*   **Missing Requirement:** `iq_row_bytes` must return the correct padded size for Q4_K. Q4_K blocks are 144 bytes for 256 values. If `n_embd` (2560) is not a multiple of 256, padding is required. 2560 / 256 = 10. So no padding needed for rows. But we must ensure the GGUF reader extracts the *exact* block bytes without extra GGUF padding between tensors.

**Correction:**
The conversion tool must:
1.  Read `ffn_gate_exps.weight` (Q4_K) and `ffn_up_exps.weight` (Q4_K) separately.
2.  Concatenate them into a single blob per expert: `[Gate Data] [Up Data] [Down Data]`.
3.  Ensure that `iq_row_bytes(Q4_K, 2560)` returns exactly `10 * 144 = 1440` bytes.
4.  Verify that `native_expert_supported` (line 1417 in `iq_kernels.cu`) returns `true` for `gu_type=12` (Q4_K), `d_type=8` (Q8_0), `n_embd=2560`, `n_ff=640`.
    *   `n_embd % 256 == 0` (2560 % 256 = 0) -> OK.
    *   `n_ff % qd == 0` (640 % 32 = 0) -> OK.
    *   **But:** Does `gu_qk(12)` return 256? Yes, Q4_K block size is 256.
    *   **Risk:** The `native_expert_grouped` kernel might assume `gu_type` and `d_type` are the same, or have specific optimizations for Q4_K/Q4_K. The sidecar uses **Q4_K for Gate/Up** and **Q8_0 for Down**. This mixed-type expert support must be explicitly verified in the CUDA kernel implementation (`iq_kernels.cu`), which is only partially shown. The proposal assumes it works because the struct *allows* different types, but the *kernels* might not.

### 3. Critical Mistake: Tensor Naming and `eh_proj` Handling

**The Proposal Claims:**
> "Map `blk.48.*` names to Strata's internal expected names... Handle the `eh_proj` split: If the sidecar provides a fused `[5120, 2560]` projection, split it into `fc_embedding` and `fc_hidden` equivalents..."

**The Reality (Source Evidence):**
*   **`src/core/mtp.cpp` line 201:** The loader requires `fc_embedding.weight` and `fc_hidden.weight`.
*   **Sidecar Specimen:** `blk.48.nextn.eh_proj.weight` is `[5120, 2560]`. The description says: "each output row concatenating the embedding input columns followed by hidden input columns."
*   **The Conflict:** The sidecar has **one** tensor `eh_proj`. Strata expects **two** tensors `fc_embedding` and `fc_hidden`.
*   **Missing Requirement:** The conversion tool must **split** the `eh_proj` tensor.
    *   `fc_embedding` should be the first 2560 columns (or rows, depending on layout).
    *   `fc_hidden` should be the last 2560 columns.
    *   **Crucial:** The proposal says "split it... or update the runtime." **Do not update the runtime.** The runtime is complex and optimized. The conversion tool must do the work.
    *   **Quantization Issue:** `eh_proj` is Q4_K. Splitting a Q4_K tensor is **not trivial**. Q4_K blocks span 256 values. If you split the tensor along the inner dimension (2560), you must ensure the split point aligns with block boundaries. 2560 / 256 = 10. So splitting at 2560 is safe. But you must extract the raw Q4_K blocks for the first half and the second half separately. You cannot just slice the byte array if the GGUF layout is row-major and the split is along the column axis. You need to dequantize, split, and requantize? **No.** The proposal says "retain existing Q2 MTP support." It implies we want to use the sidecar's Q4_K weights. If we split a Q4_K tensor, we must preserve the Q4_K format. This requires careful byte-level manipulation of the GGUF blocks.

**Correction:**
The conversion tool must:
1.  Read `blk.48.nextn.eh_proj.weight` (Q4_K, shape [5120, 2560]).
2.  Split it into two Q4_K tensors: `fc_embedding` [2560, 2560] and `fc_hidden` [2560, 2560].
3.  This requires parsing the Q4_K blocks. Since the split is along the inner dimension (2560), and Q4_K blocks are 256 values, the split is at block boundary 10. We can simply take the first 10 blocks of each row for `fc_embedding` and the next 10 for `fc_hidden`.
4.  Write these as separate entries in `dense_native.bin` with type Q4_K.

### 4. Missing Requirement: Norms and Gemma +1 Offset

**The Proposal Claims:**
> "Source GGUF norm scales already include their Gemma +1 offset."

**The Reality (Source Evidence):**
*   **`tools/mtp_rt.py` lines 85–87:**
    ```python
    if int(t.tensor_type) == 0:        # F32 norm weights: raw GemmaRMSNorm w -> 1 + w
        arr = (data.astype(np.float32) + 1.0)
    ```
    The current tool **adds 1.0** to F32 norm weights.
*   **Sidecar Specimen:** "source GGUF norm scales already include their Gemma +1 offset."
*   **The Conflict:** If the sidecar already has the +1, and we load it as F32, we must **NOT** add 1.0 again.
*   **Missing Requirement:** The new conversion tool must **detect** that the sidecar norms are pre-offset and **skip** the `+1.0` step. If we blindly apply the current logic, we will double the offset, breaking the model.

**Correction:**
The conversion tool must have a flag or logic to skip the `+1.0` addition for norm weights when processing the external sidecar.

### 5. Missing Requirement: Router and Hyper-Connection Types

**The Proposal Claims:**
> "Dense attention/shared-expert projections use Q4_K, Q6_K or Q8_0... Hyper-connection down/inject use Q4_K or Q6_K; their up matrices use Q5_0."

**The Reality (Source Evidence):**
*   **`src/core/mtp.cpp` line 526:**
    ```cpp
    bf16_gemv_fp32_mmvf(mixed_ + t * N, bf16("mlp.gate.weight"), logits_ + t * g.n_expert, ...);
    ```
    The router (`mlp.gate.weight`) is loaded as **BF16** (`bf16()` helper).
*   **Sidecar Specimen:** "Router ffn_gate_inp and the 1-D ffn_gate_inp_shexp are F32."
*   **The Conflict:** The sidecar has F32 router weights. Strata expects BF16.
*   **Missing Requirement:** The conversion tool must **convert** F32 router weights to BF16. Or, the runtime must be updated to support F32 router weights. Given the "minimal change" goal, **conversion** is preferred.
*   **Hyper-Connections:** The sidecar has Q4_K/Q6_K/Q5_0 for hyper-connections. Strata's current `dense.txt` only supports `q8_0`, `bf16`, `f32`. The runtime likely uses `bf16()` or `q8()` helpers for these. We must verify if `iq_mmvq` is used for hyper-connections. If not, we may need to convert these to Q8_0 or BF16 in the conversion tool to fit the existing runtime structure, or implement full dynamic type support for *all* dense tensors.

**Correction:**
1.  Convert F32 router weights to BF16 in the conversion tool.
2.  For hyper-connection weights (Q4_K/Q6_K/Q5_0), either:
    *   **Option A (Hard):** Update runtime to support dynamic types for hyper-connections.
    *   **Option B (Easy):** Convert them to Q8_0 or BF16 in the conversion tool. Given the small size of hyper-connection matrices, **converting to BF16** is likely the safest and fastest path to integration without touching the CUDA kernels.

### 6. Memory Implications Re-evaluation

**The Proposal Claims:**
> "Total MTP VRAM: ~2.1–2.2 GB."

**The Reality:**
*   The proposal ignores the **scratch buffers** for `native_expert_grouped`.
*   **`include/strata/kernels/iq_kernels.hpp` line 54:** `native_expert_scratch_bytes` depends on `cap_entries` and `n_ff`.
*   **`src/core/mtp.cpp` line 273:** `hit_scratch_` is allocated.
*   The scratch buffer for Q4_K experts might be larger than for Q2_0 due to different intermediate formats.
*   **RTX 4060 (8 GB):** If the drafter is on the 4060, 2.2 GB is a huge chunk. The main model (IQ4_XS) + KV cache + drafter might exceed 8 GB. The proposal suggests `--split-skip-if-fits`, but this is a runtime flag, not a guarantee.
*   **Correction:** The proposal must explicitly state that **loading the Q4_K_M drafter on the 8 GB card is likely to fail or cause severe thrashing** unless the main model is extremely small or context is tiny. The drafter should probably be forced to the 16 GB card.

### 7. Validation Gaps

**The Proposal Claims:**
> "Compare the output logits of the MTP layer against a reference implementation (e.g., llama.cpp)."

**The Reality:**
*   We do not have a reference implementation of the *Strata* MTP layer in llama.cpp. Strata's MTP is a custom implementation.
*   **Correction:** Validation must be **internal**:
    1.  **Bitwise Check:** Convert the sidecar to Q8_0/BF16 using the *old* tool (if possible) and compare against the new native loader's output for the same inputs.
    2.  **Acceptance Rate:** The only true metric is draft acceptance rate. If it drops, the quantization or splitting is wrong.
    3.  **Crash Test:** Run with `CUDA_LAUNCH_BLOCKING=1` to catch kernel mismatches.

### Summary of Corrections for the Maintainer

1.  **Do not assume `iq_mmvq` works for all dense types.** Verify Q4_K support for specific MTP tensor shapes. If missing, convert dense weights to Q8_0/BF16 in the conversion tool.
2.  **Do not assume `native_expert_grouped` supports mixed Q4_K/Q8_0 experts.** Verify the CUDA kernel implementation. If not, convert all expert weights to a single type (e.g., Q4_K for all, or Q8_0 for all) in the conversion tool.
3.  **Split `eh_proj` carefully.** Use byte-level block extraction for Q4_K, not naive slicing.
4.  **Skip `+1.0` for norms.** The sidecar is pre-offset.
5.  **Convert Router/Hyper-connections to BF16.** Do not try to load
