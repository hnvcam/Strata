# Proposal: Native Q4_K/Q8_0 Expert Support for MTP Sidecar

## Executive Summary
The requirement to retain **Q4_K gate/up** and **Q8_0 down** experts necessitates abandoning the legacy Q2_0 runtime path. The current `src/core/mtp.cpp` is hard-coded to load Q2_0 blobs and invoke `moe_grouped_s2`.

This proposal outlines the changes required to enable the **Native Expert Path** (`native_expert_grouped`) for the MTP layer. This involves:
1.  **Runtime Loader:** Modifying `mtp.cpp` to detect expert types and load raw GGUF blocks into a `NativeExpertLayout` compatible buffer.
2.  **Kernel Dispatch:** Switching from `moe_grouped_s2` to `native_expert_grouped` when Q4_K/Q8_0 experts are detected.
3.  **Memory Budgeting:** Increasing `kDrafterMib` to accommodate the ~2x larger expert footprint.
4.  **Converter:** A new tool to map sidecar names to Strata names and prepare the `dense.bin` (Q8_0/BF16/F32) while passing through expert bytes.

---

## 1. Affected Files

### Modified Files
*   `src/core/mtp.cpp`:
    *   **Loader:** Replace fixed-size Q2_0 blob loading with dynamic layout detection.
    *   **Dispatch:** Add conditional logic to call `native_expert_grouped` instead of `moe_grouped_s2`.
    *   **Scratch:** Allocate scratch buffers for `native_expert_grouped`.
*   `src/program/generate.cpp`:
    *   **Constants:** Increase `kDrafterMib` from 1000 to ~2200 MiB.
*   `include/strata/core/mtp.hpp`:
    *   **State:** Add members to store `NativeExpertLayout` and scratch pointers.

### New Files
*   `tools/mtp_rt_native.py`: Converter that maps sidecar names, converts dense tensors to Q8_0/BF16/F32, and writes expert tensors in a format the new loader can read (or directly reads GGUF if loader supports it). *Recommendation: Keep GGUF format for experts to avoid intermediate binary formats, requiring loader to parse GGUF tensor offsets.*

---

## 2. Technical Design

### 2.1 Expert Layout and Dispatch

**Current State (Q2_0):**
*   `mtp.cpp` lines 180-200: Reads `experts.bin` as a flat array of `g.n_expert * BLOB` bytes.
*   `mtp.cpp` line 533: Calls `moe_grouped_s2`.

**Target State (Native Q4_K/Q8_0):**
*   **Layout:** The sidecar provides `blk.48.ffn_gate_exps.weight` (Q4_K) and `blk.48.ffn_down_exps.weight` (Q8_0).
*   **Strata Native Layout:** `include/strata/kernels/iq_kernels.hpp` line 41 defines `NativeExpertLayout`:
    ```cpp
    struct NativeExpertLayout {
        int gu_type = -1, d_type = -1;
        int64_t n_embd = 0, n_ff = 0;
        size_t gu_row = 0, d_row = 0;       // bytes per row
        size_t up_off = 0, down_off = 0;    // byte offsets inside the blob
        size_t bytes = 0;                   // the whole blob
    };
    ```
    *   **Gate/Up Interleaving:** `native_expert_grouped` expects gate and up rows to be contiguous or interleaved?
        *   `iq_kernels.hpp` line 35-38 (`iq_dequant_gu_f16`) suggests gate/up are separate.
        *   `native_expert_layout` (line 1423 in `iq_kernels.cu`) calculates `up_off = n_ff * gu_row` and `down_off = 2 * up_off`. This implies the blob layout is `[Gate Rows | Up Rows | Down Rows]`.
    *   **Sidecar Mapping:**
        *   Sidecar `ffn_gate_exps` (Q4_K) $\rightarrow$ Gate Rows.
        *   Sidecar `ffn_up_exps` (Q4_K) $\rightarrow$ Up Rows.
        *   Sidecar `ffn_down_exps` (Q8_0) $\rightarrow$ Down Rows.
    *   **Action:** The converter (or loader) must concatenate these three tensors per expert into a single blob: `[Gate_Expert_i | Up_Expert_i | Down_Expert_i]`.

**Dispatch Logic in `mtp.cpp`:**
1.  **Detection:** Read metadata (e.g., from `dense.txt` or a new `meta.txt`) to determine expert types.
    *   If `gu_type == 12` (Q4_K) and `d_type == 8` (Q8_0):
        *   Set `native_expert = true`.
        *   Calculate `NativeExpertLayout L = native_expert_layout(12, 8, H, FF)`.
        *   Verify `native_expert_supported(12, 8, H, FF)` returns true.
2.  **Loading:**
    *   Allocate `experts_` with size `g.n_expert * L.bytes`.
    *   Load sidecar expert tensors into this buffer, respecting the `[Gate|Up|Down]` layout.
3.  **Execution:**
    *   In the MoE loop (line 524-534):
        *   If `native_expert`:
            *   Call `native_expert_grouped(L, grp_ptr_, ..., x_q8_1, scratch_, parts_, cs)`.
            *   *Note:* `native_expert_grouped` expects `x_q8_1` (Q8_1 activations). The current code uses `hit_xq_` (Q8_0 scaled?).
            *   **Correction:** `native_expert_grouped` (line 59 in `iq_kernels.hpp`) takes `const void* x_q8_1`. The current MTP code uses `quantize_q8_0_scaled` (line 532). We must ensure activations are quantized to **Q8_1** (fp16 scale + fp16 sum) for native kernels, or verify if `native_expert_grouped` accepts Q8_0.
            *   *Source Check:* `iq_kernels.hpp` line 7 says "Activations are q8_1... the llama.cpp CUDA contract."
            *   *Action:* Replace `quantize_q8_0_scaled` with `quantize_q8_1_rows` (line 23 in `iq_kernels.hpp`) for the native path.

### 2.2 Dense Tensor Conversion

Dense tensors do not need native Q4_K support if we convert them to existing formats.

*   **Q8_0 Set:** `fc_embedding`, `fc_hidden`, `self_attn.*`, `mlp.shared_expert.*`.
    *   Convert sidecar Q4_K/Q6_K/Q8_0 to Q8_0 using `q8_0()` from `mtp_rt.py`.
*   **BF16 Set:** Hyper-connections, Router.
    *   Convert sidecar Q4_K/Q5_0/Q6_K to BF16.
*   **F32 Norms:**
    *   Keep as F32. **Do not add 1.0** (sidecar already has offset).

### 2.3 Tensor Name and Axis Corrections

*   **`eh_proj` Split:**
    *   Sidecar: `blk.48.nextn.eh_proj.weight` `[5120, 2560]`.
    *   Strata: `fc_embedding.weight` `[2560, 2560]` and `fc_hidden.weight` `[2560, 2560]`.
    *   **Axis Check:** GGUF is innermost-first. `[5120, 2560]` means 5120 rows, 2560 cols.
    *   **Split:** Rows 0-2559 $\rightarrow$ `fc_embedding`. Rows 2560-5119 $\rightarrow$ `fc_hidden`.
    *   **Uncertainty:** Verify if `eh_proj` is `[Embedding; Hidden]` or `[Hidden; Embedding]`.
*   **Expert Axes:**
    *   Sidecar `ffn_gate_exps` `[2560, 640, 512]`. Innermost is 512 (experts).
    *   Strata `NativeExpertLayout` expects per-expert blobs.
    *   **Action:** The converter/loader must transpose/reshape to extract per-expert matrices `[2560, 640]` (Gate) and `[640, 2560]` (Down).

---

## 3. Compatibility Checks

1.  **Kernel Support:**
    *   Verify `native_expert_supported(12, 8, 2560, 640)` returns `true`.
    *   Source: `iq_kernels.cu` line 1417.
    *   Check: `is_iq(12)` (Q4_K) and `is_iq(8)` (Q8_0) must be true. `iq_kernels.cu` line 1269 includes 12 and 8 in `is_iq`.
    *   Check: `n_embd % qg == 0`. Q4_K block size is 256. $2560 \% 256 == 0$. OK.
    *   Check: `n_ff % qd == 0`. Q8_0 block size is 32. $640 \% 32 == 0$. OK.
2.  **Activation Format:**
    *   Ensure `x_q8_1` buffer is allocated and populated with Q8_1 blocks (fp16 scale/sum).
    *   Current `mtp.cpp` uses `hit_xq_` (Q8_0). Must switch to Q8_1 for native path.
3.  **Scratch Memory:**
    *   `native_expert_scratch_bytes` (line 1437 in `iq_kernels.cu`) requires significant scratch.
    *   Formula: $3 \times (\text{cap\_entries} \times n\_ff \times 4) + \dots$
    *   For `cap_entries = T * K` (e.g., 32 * 10 = 320), $n\_ff=640$:
        *   $3 \times 320 \times 640 \times 4 \approx 2.4$ MB.
        *   Plus Q8_1 scratch.
    *   This is small compared to VRAM, but must be allocated in `mtp.cpp` arena.

---

## 4. Byte and Memory Estimates

**Expert VRAM:**
*   **Gate/Up (Q4_K):**
    *   Elements: $512 \times 2560 \times 640 \times 2$ (gate+up) = $1.67$ billion elements.
    *   Q4_K bytes/element: $144 / 256 = 0.5625$.
    *   Size: $1.67 \times 10^9 \times 0.5625 \approx 940$ MB.
*   **Down (Q8_0):**
    *   Elements: $512 \times 640 \times 2560 = 838$ million elements.
    *   Q8_0 bytes/element: $34 / 32 = 1.0625$.
    *   Size: $838 \times 10^6 \times 1.0625 \approx 890$ MB.
*   **Total Experts:** $\approx 1.83$ GB.
*   **Comparison:** Current Q2_0 experts are ~0.71 GB. **Increase: +1.12 GB.**

**Dense VRAM:**
*   Converted to Q8_0/BF16/F32. Similar to current (~0.18 GB).

**Total Drafter VRAM:**
*   $1.83$ GB (Experts) + $0.18$ GB (Dense) + Overhead $\approx 2.0$ GB.
*   **`kDrafterMib` Update:** Must increase from 1000 to **2200 MiB** (2.15 GB) to be safe.

**Impact on RTX 4060 (8 GB):**
*   Drafter takes 2.0 GB.
*   Remaining for KV/Cache: ~6 GB.
*   This is tight. `--split-skip-if-fits` will likely fail, forcing a layer split. This may reduce performance due to cross-GPU communication.

---

## 5. Validation Plan

1.  **Unit Test (Converter):**
    *   Verify `eh_proj` split produces two valid Q8_0 tensors.
    *   Verify expert blob layout matches `NativeExpertLayout` expectations (Gate|Up|Down).
2.  **Integration Test (Loader):**
    *   Load sidecar. Verify `native_expert_supported` returns true.
    *   Verify `experts_` allocation size matches `g.n_expert * L.bytes`.
3.  **Numerical Check:**
    *   Run a single forward pass. Compare logits with a reference llama.cpp run using the same sidecar.
    *   **Critical:** Check for "garbage" output, which would indicate incorrect axis mapping (Gate/Up swap) or activation format mismatch (Q8_0 vs Q8_1).
4.  **Performance Check:**
    *   Measure decode speed. Native Q4_K kernels may be slower than Q2_0 `moe_grouped_s2` due to higher bandwidth requirements.

---

## 6. Remaining Uncertainties

1.  **`eh_proj` Row Order:** Is it `[Embedding; Hidden]` or `[Hidden; Embedding]`?
2.  **Q8_1 Activation Overhead:** Switching from Q8_0 to Q8_1 activations for the native path may require changes to `quantize_q8_0_scaled` call sites.
3.  **Kernel Performance:** `native_expert_grouped` for Q4_K/Q8_0 at these dimensions is untested in Strata. It may be significantly slower than the specialized Q2_0 kernel.
4.  **RTX 4060 Viability:** The 2.0 GB drafter may make the 8 GB card unusable for long contexts without aggressive splitting.

## Conclusion
This proposal requires **C++ runtime changes** to support native Q4_K/Q8_0 experts. It is the only way to meet the requirement of retaining expert quantization. The primary risks are numerical correctness (axis mapping) and performance regression (kernel efficiency). The memory increase is substantial and must be accounted for in VRAM budgeting.
