# Proposal: Dual-Path MTP Runtime for Native Q4_K/Q8_0 Sidecars

## Executive Summary
To support the external `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` sidecar while retaining legacy Q2_0 support, Strata must implement a **dual-path MTP runtime**. The current runtime (`src/core/mtp.cpp`) is hard-coded to load Q2_0 blobs and invoke `moe_grouped_s2`. This proposal introduces a detection mechanism in the loader to switch to the **Native Expert Path** (`native_expert_grouped`) when Q4_K/Q8_0 experts are detected.

This approach preserves the supplied expert quantization (Q4_K gate/up, Q8_0 down) without re-quantization, at the cost of increased VRAM usage (~2.0 GB vs ~0.9 GB) and required C++ runtime modifications.

---

## 1. Affected Files

### Modified Files
1.  **`src/core/mtp.cpp`**:
    *   **Loader:** Replace fixed-size Q2_0 blob loading with dynamic layout detection.
    *   **Dispatch:** Add conditional logic to invoke `native_expert_grouped` instead of `moe_grouped_s2`.
    *   **Activations:** Switch activation quantization from Q8_0 to Q8_1 for the native path.
    *   **Scratch:** Allocate scratch buffers required by `native_expert_grouped`.
2.  **`include/strata/core/mtp.hpp`**:
    *   Add members to store `NativeExpertLayout` and scratch pointers.
3.  **`src/program/generate.cpp`**:
    *   Update `kDrafterMib` constant to reflect the larger native drafter footprint.

### New Files
1.  **`tools/mtp_rt_native.py`**:
    *   A converter that maps sidecar tensor names to Strata names, converts dense tensors to Q8_0/BF16/F32, and prepares expert data for the native loader.

---

## 2. Converter Design (`tools/mtp_rt_native.py`)

The converter transforms the sidecar GGUF into a format compatible with the new runtime.

### 2.1 Expert Handling (Native Q4_K/Q8_0)
**Source:** `blk.48.ffn_gate_exps.weight` (Q4_K), `blk.48.ffn_up_exps.weight` (Q4_K), `blk.48.ffn_down_exps.weight` (Q8_0).
**Target:** A raw binary blob per expert in the layout expected by `native_expert_grouped`.

1.  **Layout Requirement:**
    *   `include/strata/kernels/iq_kernels.hpp` line 41 defines `NativeExpertLayout`.
    *   `src/kernels/cuda/iq_kernels.cu` line 1423 (`native_expert_layout`) calculates offsets:
        *   `up_off = n_ff * gu_row`
        *   `down_off = 2 * up_off`
    *   This implies the per-expert blob layout is: `[Gate Rows | Up Rows | Down Rows]`.
2.  **Conversion Steps:**
    *   For each expert `e` (0 to 511):
        *   Extract `gate_e` from `ffn_gate_exps` (Q4_K).
        *   Extract `up_e` from `ffn_up_exps` (Q4_K).
        *   Extract `down_e` from `ffn_down_exps` (Q8_0).
        *   Concatenate bytes: `gate_e_bytes + up_e_bytes + down_e_bytes`.
    *   Write all 512 blobs to `experts_native.bin`.
    *   *Note:* The runtime loader will need to know the types (Q4_K=12, Q8_0=8) and dimensions (H=2560, FF=640) to reconstruct the `NativeExpertLayout`.

### 2.2 Dense Tensor Handling
**Target:** `dense.bin` and `dense.txt` (same format as `tools/mtp_rt.py`).

1.  **Name Mapping:**
    *   `blk.48.nextn.eh_proj.weight` $\rightarrow$ Split into `fc_embedding.weight` and `fc_hidden.weight`.
        *   *Action:* Split the `[5120, 2560]` tensor into two `[2560, 2560]` tensors.
        *   *Uncertainty:* Verify row order (Embedding first vs Hidden first).
    *   `blk.48.attn_q_proj.weight` $\rightarrow$ `self_attn.q_proj.weight`
    *   `blk.48.ffn_gate_shexp.weight` $\rightarrow$ `mlp.shared_expert.gate_proj.weight`
    *   `blk.48.ffn_gate_inp.weight` $\rightarrow$ `mlp.gate.weight` (Router)
2.  **Quantization:**
    *   **Q8_0 Set:** Convert `fc_embedding`, `fc_hidden`, `self_attn.*`, `mlp.shared_expert.*` to Q8_0 using `q8_0()` from `tools/mtp_rt.py` (lines 34-45).
    *   **BF16 Set:** Convert Hyper-connection and Router tensors to BF16.
    *   **F32 Norms:** Keep `nextn.enorm`, etc., as F32. **Do NOT add 1.0** (sidecar already includes Gemma +1 offset).

---

## 3. Runtime Design (`src/core/mtp.cpp`)

### 3.1 Loader Modifications
Current code (lines 180-200) assumes Q2_0. New logic:

1.  **Metadata Detection:**
    *   Read a new metadata file (e.g., `meta.txt`) or infer from `dense.txt` presence of specific keys.
    *   If `expert_format == "q4_k_q8_0"`:
        *   Set `native_expert = true`.
        *   Calculate `NativeExpertLayout L = native_expert_layout(12, 8, 2560, 640)`.
        *   Verify `native_expert_supported(12, 8, 2560, 640)` (line 1417 in `iq_kernels.cu`).
2.  **Memory Allocation:**
    *   Allocate `experts_` with size `g.n_expert * L.bytes`.
    *   Load `experts_native.bin` into `experts_`.
3.  **Scratch Allocation:**
    *   Allocate scratch buffer using `native_expert_scratch_bytes(cap_entries, n_ff)` (line 1437 in `iq_kernels.cu`).
    *   Store pointer in `mtp.hpp`.

### 3.2 Execution Dispatch
Current code (lines 524-534) uses `moe_grouped_s2`. New logic:

1.  **Activation Quantization:**
    *   `native_expert_grouped` expects **Q8_1** activations (fp16 scale/sum), per `iq_kernels.hpp` line 7.
    *   Current code uses `quantize_q8_0_scaled` (line 532).
    *   **Change:** If `native_expert`, call `quantize_q8_1_rows` (line 23 in `iq_kernels.hpp`) instead.
2.  **Kernel Call:**
    *   If `native_expert`:
        ```cpp
        native_expert_grouped(L, grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, 
                              T*K, T*K, x_q8_1, scratch_, parts_, cs);
        ```
    *   Else (Legacy Q2_0):
        ```cpp
        moe_grouped_s2(grp_ptr_, grp_start_, grp_counts_, hit_dst_, hit_slot_, 
                       T*K, T*K, hit_xq_, hit_xs_, hit_scratch_, parts_, cs);
        ```

---

## 4. Compatibility and Memory Implications

### 4.1 VRAM Estimates
*   **Legacy Q2_0 Drafter:**
    *   Experts: ~0.71 GB.
    *   Dense: ~0.18 GB.
    *   Total: ~0.89 GB.
    *   `kDrafterMib`: 1000 MiB.
*   **Native Q4_K/Q8_0 Drafter:**
    *   **Experts:**
        *   Gate/Up (Q4_K): $512 \times 2560 \times 640 \times 2 \times (144/256) \approx 940$ MB.
        *   Down (Q8_0): $512 \times 640 \times 2560 \times (34/32) \approx 890$ MB.
        *   Total Experts: **~1.83 GB**.
    *   **Dense:** ~0.18 GB (Q8_0/BF16/F32).
    *   **Scratch:** ~5 MB.
    *   **Total:** **~2.01 GB**.
*   **Action:** Increase `kDrafterMib` in `src/program/generate.cpp` (line 2299) from 1000 to **2200 MiB**.

### 4.2 Hardware Impact
*   **RTX 5070 Ti (16 GB):** Comfortable.
*   **RTX 4060 (8 GB):** Tight. The 2.0 GB drafter consumes 25% of VRAM. `--split-skip-if-fits` (line 2139 in `generate.cpp`) will likely fail, forcing a layer split. This may degrade performance due to cross-GPU communication overhead.

### 4.3 Kernel Compatibility
*   `native_expert_supported` (line 1417 in `iq_kernels.cu`) must return `true` for Q4_K/Q8_0 at H=2560, FF=640.
    *   `is_iq(12)` and `is_iq(8)` are true (line 1269).
    *   `n_embd % 256 == 0` (2560 % 256 = 0).
    *   `n_ff % 32 == 0` (640 % 32 = 0).
    *   **Status:** Likely supported, but untested for this specific dimension pair.

---

## 5. Validation Plan

1.  **Static Checks:**
    *   Verify `eh_proj` split produces two valid Q8_0 tensors.
    *   Verify expert blob layout matches `NativeExpertLayout` offsets.
2.  **Numerical Correctness:**
    *   Run a single forward pass with the new runtime.
    *   Compare logits with a reference llama.cpp run using the same sidecar.
    *   **Critical:** Check for "garbage" output, which would indicate incorrect axis mapping (Gate/Up swap) or activation format mismatch (Q8_0 vs Q8_1).
3.  **Memory Profile:**
    *   Monitor VRAM usage on RTX 4060. Ensure `kDrafterMib` update prevents OOM during cache sizing.
4.  **Performance:**
    *   Measure decode speed. Native Q4_K kernels may be slower than Q2_0 `moe_grouped_s2` due to higher bandwidth requirements.

---

## 6. Remaining Uncertainties

1.  **`eh_proj` Row Order:** Is it `[Embedding; Hidden]` or `[Hidden; Embedding]`? Incorrect splitting will corrupt the drafter.
2.  **Q8_1 Activation Overhead:** Switching to Q8_1 activations for the native path may require changes to `quantize_q8_0_scaled` call sites and buffer sizes.
3.  **Kernel Performance:** `native_expert_grouped` for Q4_K/Q8_0 at these dimensions is untested in Strata. It may be significantly slower than the specialized Q2_0 kernel.
4.  **RTX 4060 Viability:** The 2.0 GB drafter may make the 8 GB card unusable for long contexts without aggressive splitting.

## Conclusion
This proposal requires **C++ runtime changes** to support native Q4_K/Q8_0 experts. It is the only way to meet the requirement of retaining expert quantization. The primary risks are numerical correctness (axis mapping) and performance regression (kernel efficiency). The memory increase is substantial and must be accounted for in VRAM budgeting.
