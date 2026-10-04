# Proposal: Integration of External Q4_K_M MTP Sidecar via Re-quantization to Legacy Q2_0 Runtime

## Executive Summary
This proposal outlines the integration of the external `mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` sidecar into Strata’s existing MTP runtime.

**Strategy:** **Offline Re-quantization (Option A)**.
We will not modify the C++ runtime (`src/core/mtp.cpp`) or GPU kernels (`src/kernels/cuda/iq_kernels.cu`). Instead, we will create a new Python tool that converts the sidecar’s Q4_K/Q8_0 tensors into the **legacy Q2_0 expert blob format** and **Q8_0/BF16/F32 dense format** expected by the current engine.

**Rationale:**
1.  **Runtime Stability:** The current runtime (`mtp.cpp`) is hard-coded to expect Q2_0 experts (`moe_grouped_s2`) and specific Q8_0 dense tensors. Modifying it to support native Q4_K (`native_expert_grouped`) requires significant changes to memory layout, kernel dispatch, and VRAM budgeting (`kDrafterMib`), introducing high regression risk.
2.  **Compatibility:** Re-quantizing to Q2_0 allows the sidecar to run on the existing, validated Q2_0 pipeline without any C++ changes.
3.  **Memory:** While Q4_K is larger than Q2_0, the re-quantized Q2_0 output will be smaller than the native Q4_K sidecar, fitting within the existing VRAM budget (with minor constant adjustments).

---

## 1. Affected Files

### New Files
*   `tools/mtp_rt_sidecar.py`: A new converter tool. It reads the external GGUF, maps tensor names, handles norm offsets, splits fused projections, and re-quantizes experts to Q2_0.

### Modified Files
*   `src/program/generate.cpp`: Update `kDrafterMib` constant to reflect the slightly larger dense tensor footprint (if any) and ensure safety margins for the new source.
*   `setup.py` (Optional): Add logic to detect if a sidecar is present and invoke `mtp_rt_sidecar.py` instead of `mtp_rt.py` during setup.

### Unmodified Files
*   `src/core/mtp.cpp`: No changes.
*   `include/strata/kernels/cpu/expert.hpp`: No changes.
*   `src/kernels/cuda/iq_kernels.cu`: No changes.

---

## 2. Conversion Design (`tools/mtp_rt_sidecar.py`)

The converter must transform the sidecar’s GGUF structure into the `experts.bin` and `dense.bin`/`dense.txt` format produced by `tools/mtp_rt.py`.

### 2.1 Expert Handling (Q4_K/Q8_0 $\rightarrow$ Q2_0)
**Source:** `blk.48.ffn_gate_exps.weight` (Q4_K, shape `[2560, 640, 512]`), `blk.48.ffn_down_exps.weight` (Q8_0, shape `[640, 2560, 512]`).
**Target:** `experts.bin` (512 blobs of Q2_0 layout).

1.  **Dequantization:**
    *   Load Q4_K gate/up and Q8_0 down tensors.
    *   Dequantize to FP32. Note GGUF shape is innermost-first; ensure correct reshaping to `[NE, H, FF]` and `[NE, FF, H]`.
2.  **Re-quantization:**
    *   Use the `q2_0` function from `tools/mtp_pack.py` (lines 58-82) to quantize the FP32 experts to Q2_0.
    *   **Critical:** The `q2_0` function expects input shape `[..., n]`. Ensure the expert matrices are flattened correctly per expert.
3.  **Blob Layout:**
    *   Use the `blob_of` logic from `tools/mtp_rt.py` (lines 48-58) to interleave gate/up rows and pack scales/codes into the `BLOB` format defined in `include/strata/kernels/cpu/expert.hpp`.
    *   Write to `experts.bin`.

### 2.2 Dense Tensor Handling
**Target:** `dense.bin` and `dense.txt`.

1.  **Name Mapping:**
    *   `blk.48.nextn.eh_proj.weight` $\rightarrow$ Split into `fc_embedding.weight` and `fc_hidden.weight`.
        *   *Action:* The sidecar tensor is `[5120, 2560]`. The first 2560 rows correspond to embedding inputs, the next 2560 to hidden inputs (based on prompt description "concatenating the embedding input columns followed by hidden input columns").
        *   *Verification:* This assumption **must** be validated against a reference llama.cpp run. If the order is interleaved or reversed, the split will fail.
    *   `blk.48.attn_q_proj.weight` $\rightarrow$ `self_attn.q_proj.weight`
    *   `blk.48.attn_k_proj.weight` $\rightarrow$ `self_attn.k_proj.weight`
    *   `blk.48.attn_v_proj.weight` $\rightarrow$ `self_attn.v_proj.weight`
    *   `blk.48.attn_o_proj.weight` $\rightarrow$ `self_attn.o_proj.weight`
    *   `blk.48.ffn_gate_shexp.weight` $\rightarrow$ `mlp.shared_expert.gate_proj.weight`
    *   `blk.48.ffn_up_shexp.weight` $\rightarrow$ `mlp.shared_expert.up_proj.weight`
    *   `blk.48.ffn_down_shexp.weight` $\rightarrow$ `mlp.shared_expert.down_proj.weight`
    *   `blk.48.ffn_gate_inp.weight` $\rightarrow$ `mlp.gate.weight` (Router)
    *   `blk.48.ffn_gate_inp_shexp.weight` $\rightarrow$ `mlp.shared_expert_gate.weight`

2.  **Quantization Strategy:**
    *   **Q8_0 Set:** The tensors listed in `tools/mtp_rt.py` lines 29-31 (`fc_embedding`, `fc_hidden`, `self_attn.*`, `mlp.shared_expert.*`) must be converted to Q8_0.
        *   *Action:* Dequantize sidecar Q4_K/Q6_K/Q8_0 tensors to FP32, then apply `q8_0()` from `tools/mtp_rt.py` (lines 34-45).
    *   **BF16 Set:** Hyper-connection tensors (`hc_attn_down`, `hc_ffn_up`, etc.) and Router weights (`mlp.gate.weight`, `mlp.shared_expert_gate.weight`) should be kept as BF16.
        *   *Action:* Dequantize to FP32, then cast to BF16.
    *   **F32 Norms:** `nextn.enorm`, `nextn.hnorm`, `hc_head_norm`, etc.
        *   *Action:* Keep as F32. **Do NOT add 1.0.** The sidecar already includes the Gemma +1 offset. The current `mtp_rt.py` adds 1.0 (line 86); the new converter must skip this step for these tensors.

3.  **Indexer Tensors:**
    *   The sidecar contains unused indexer tensors. The converter should **ignore** them.
    *   *Check:* Verify `src/core/mtp.cpp` does not require `self_attn.indexer.index_qk_proj.weight`. `mtp.cpp` line 201-203 does *not* list it as required. `mtp_rt.py` line 30 includes it in Q8 set, but if the runtime doesn't load it, it can be omitted from `dense.txt`.

### 2.3 Output Generation
*   Write `experts.bin` (Q2_0 blobs).
*   Write `dense.bin` (Q8_0/BF16/F32 blobs).
*   Write `dense.txt` (Index: `name kind rows cols offset bytes`).

---

## 3. Runtime and Memory Considerations

### 3.1 Runtime Compatibility
*   **No C++ Changes:** The output format is identical to the current `mtp_rt.py` output. The engine (`mtp.cpp`) will load `experts.bin` and `dense.bin` without modification.
*   **Kernel Dispatch:** The engine will continue to use `moe_grouped_s2` for experts and `iq_mmvq`/`bf16_gemv` for dense tensors.

### 3.2 Memory Implications
*   **Expert Size:**
    *   Sidecar Q4_K Experts: ~1.42 GB.
    *   Converted Q2_0 Experts: ~0.71 GB (same as current).
    *   *Result:* VRAM usage for experts remains unchanged.
*   **Dense Size:**
    *   Sidecar Dense: Mixed Q4_K/Q6_K/Q8_0.
    *   Converted Dense: Q8_0/BF16/F32.
    *   *Result:* Q8_0 is larger than Q4_K. Dense tensor size may increase by ~10-20%.
*   **VRAM Budget (`kDrafterMib`):**
    *   Current `kDrafterMib` is 1000 MiB (`generate.cpp` line 2299).
    *   New Drafter Size: ~0.71 GB (experts) + ~0.22 GB (dense) + overhead = ~0.93 GB.
    *   *Action:* Increase `kDrafterMib` to **1200 MiB** to provide a safety margin for the slightly larger dense tensors and potential alignment padding.
    *   *Impact on RTX 4060 (8 GB):* The drafter is loaded on the last stage. A 1.2 GB drafter is significant. The `--split-skip-if-fits` logic (line 2139) must be tested to ensure it correctly accounts for this. If the drafter prevents the split from being skipped, performance may degrade.

---

## 4. Validation Plan

### 4.1 Static Checks (Converter)
1.  **Shape Verification:** Assert that `eh_proj` splits correctly into two `[2560, 2560]` matrices.
2.  **Norm Offset Check:** Compare a sample norm weight from the sidecar (F32) with the output `dense.bin` (F32). They must be **identical** (no +1.0 added).
3.  **Router Check:** Compare `ffn_gate_inp` (F32 in sidecar) with output `mlp.gate.weight` (BF16). Ensure no norm offset was applied.

### 4.2 Dynamic Checks (Engine)
1.  **Load Test:** Run `strata generate` with the new MTP folder. Verify no "missing tensor" or "shape mismatch" errors.
2.  **Acceptance Rate:** Compare draft acceptance rate against the legacy Q2 MTP (from BF16 checkpoint).
    *   *Expectation:* Acceptance rate should be **similar or slightly lower** due to double quantization (Q4_K $\rightarrow$ FP32 $\rightarrow$ Q2_0). If it is significantly lower, the `eh_proj` split or norm handling is incorrect.
3.  **Logit Comparison:** Generate a single token with `--draft-only` (if available) or compare logits from the drafter against a reference llama.cpp run using the same sidecar. Discrepancies indicate tensor mapping errors.

### 4.3 Memory Profile
1.  Monitor VRAM usage on RTX 5070 Ti and RTX 4060.
2.  Verify that `kDrafterMib` update prevents OOM during cache sizing.

---

## 5. Remaining Uncertainties

1.  **`eh_proj` Split Order:** The assumption that `eh_proj` rows are `[Embedding; Hidden]` is based on the prompt description. If the actual layout is `[Hidden; Embedding]` or interleaved, the drafter will produce garbage. **Mitigation:** Validate against a known-good llama.cpp output.
2.  **Norm Offset State:** The prompt states sidecar norms "already include" the offset. If this is incorrect, the drafter will have incorrect scaling. **Mitigation:** Visual inspection of norm values (should be ~1.0, not ~0.0).
3.  **Q2_0 Accuracy Loss:** Re-quantizing Q4_K to Q2_0 may introduce significant error. If acceptance rates drop below acceptable thresholds, **Option B (Native Q4_K)** must be revisited, requiring C++ changes.
4.  **Indexer Requirements:** If `mtp.cpp` or `verify.cpp` implicitly requires indexer tensors (not listed in `required` array but used in code), the converter must generate dummy or real indexer tensors. **Mitigation:** Code review of `mtp.cpp` for `indexer` references.

## Conclusion
This proposal prioritizes **correctness and stability** by leveraging the existing Q2_0 runtime. It avoids complex C++ modifications at the cost of potential accuracy loss from re-quantization. The primary risk is the `eh_proj` split logic, which must be rigorously validated. If validation fails, the project must pivot to implementing native Q4_K support in the runtime.
