This is a review of the preceding proposal from the perspective of a skeptical Strata maintainer. The proposal contains several critical technical errors regarding the current runtime architecture, tensor semantics, and memory management. It conflates the *packer* (offline tool) with the *runtime* (online engine) and assumes capabilities that do not exist in the supplied source.

### 1. Critical Error: Misunderstanding of `eh_proj` and Input Projections

**Proposal Claim:** "The sidecar has a single `eh_proj`. Conversion must split this tensor into embedding and hidden parts... or modify the runtime to accept a combined projection."
**Maintainer Review:** **Incorrect and Dangerous.**

*   **Source Fact:** In `tools/mtp_rt.py` (lines 29-31), the runtime expects two distinct tensors: `fc_embedding.weight` and `fc_hidden.weight`.
*   **Source Fact:** In `src/core/mtp.cpp` (lines 201-203), the loader explicitly checks for `fc_embedding.weight` and `fc_hidden.weight` as separate Q8_0 entities.
*   **Sidecar Fact:** The sidecar has `blk.48.nextn.eh_proj.weight` with shape `[5120, 2560]`. The prompt states: "each output row concatenating the embedding input columns followed by hidden input columns."
*   **Correction:** You cannot simply "split" this tensor in the converter and expect the runtime to work without modification if the runtime logic assumes separate matrix multiplications. More importantly, the *semantics* of `eh_proj` in the sidecar (NextN architecture) likely differ from the legacy `fc_embedding`/`fc_hidden` split in the current Strata MTP implementation.
    *   The current Strata MTP (`mtp.cpp`) likely performs `emb @ fc_embedding` and `hidden @ fc_hidden` separately or in a fused kernel that expects two distinct pointers.
    *   If the sidecar provides a single fused matrix, the converter *must* split it into two separate Q8_0 blobs to satisfy the existing `dense.txt` index format. However, you must verify that the *order* of rows in `eh_proj` matches the concatenation order expected by the legacy runtime. If the legacy runtime expects `[Embedding_Weights; Hidden_Weights]` and the sidecar provides `[Embedding_Cols; Hidden_Cols]` (transposed or interleaved), a naive split will produce garbage.
    *   **Missing Requirement:** A rigorous check of the row/column ordering in `eh_proj` vs. the legacy `fc_embedding`/`fc_hidden` layout. The proposal assumes a simple split is sufficient; it is not.

### 2. Critical Error: Expert Kernel Dispatch and `native_expert_grouped`

**Proposal Claim:** "Option B (Native Q4_K)... requires modifying `src/core/mtp.cpp` to detect the expert type and switch from `moe_grouped_s2` to `native_expert_grouped`."
**Maintainer Review:** **Underestimated Complexity.**

*   **Source Fact:** `src/core/mtp.cpp` line 533 calls `moe_grouped_s2`. This function is specific to the Q2_0 blob layout defined in `include/strata/kernels/cpu/expert.hpp`.
*   **Source Fact:** `include/strata/kernels/iq_kernels.hpp` defines `native_expert_grouped` (line 59). This function expects `NativeExpertLayout` (line 41), which describes raw GGUF blocks (gate rows, up rows, down rows).
*   **Source Fact:** The current `mtp.cpp` loader (lines 180-200) reads `experts.bin` as a flat array of `g.n_expert * BLOB` bytes. It does *not* parse a `NativeExpertLayout`.
*   **Correction:** Implementing Option B is not just a "switch" in `mtp.cpp`. It requires:
    1.  A completely new loader path in `mtp.cpp` that reads the sidecar's expert tensors directly into a `NativeExpertLayout` compatible buffer.
    2.  Modifying the `moe_grouped_s2` call site to conditionally call `native_expert_grouped`.
    3.  Ensuring `native_expert_supported` (line 1417 in `iq_kernels.cu`) returns true for Q4_K/Q8_0 at H=2560, FF=640. The proposal assumes this works; it is untested.
    4.  **Crucially:** The proposal suggests reusing `iq_kernels.hpp`. However, `mtp.cpp` currently uses `moe_grouped_s2` which is likely in a different compilation unit or namespace. Linking and ensuring the correct kernels are instantiated for the MTP dimensions is a significant build/runtime change, not a simple flag toggle.

### 3. Critical Error: Norm Offset Handling

**Proposal Claim:** "The sidecar norms already include the Gemma +1 offset... The new converter must NOT add 1.0."
**Maintainer Review:** **Partially Correct but Incomplete.**

*   **Source Fact:** `tools/mtp_rt.py` line 86: `arr = (data.astype(np.float32) + 1.0)`. This applies the offset to *all* F32 tensors identified as norms.
*   **Sidecar Fact:** The prompt states: "source GGUF norm scales already include their Gemma +1 offset."
*   **Correction:** If we convert the sidecar to the *existing* Q2_0 runtime format (Option A), we must ensure the converter *does not* add 1.0. However, the proposal fails to address **which** tensors are norms.
    *   `mtp_rt.py` treats *any* F32 tensor as a norm (line 85: `if int(t.tensor_type) == 0`).
    *   The sidecar has F32 tensors for `nextn.enorm`, `nextn.hnorm`, `hc_head_norm`, etc., but also `ffn_gate_inp` (Router) and `ffn_gate_inp_shexp` (Shared Expert Router) which are **F32** but are **NOT** norms. They are projection weights.
    *   **Mistake:** If the converter blindly applies the "no +1.0" rule to all F32 tensors, it is correct for norms. But if it blindly applies the "add +1.0" rule (like `mtp_rt.py` does), it will corrupt the Router weights (`ffn_gate_inp`).
    *   **Missing Requirement:** The converter must distinguish between *Norm* F32 tensors and *Projection* F32 tensors. `mtp_rt.py` does not do this; it assumes all F32 are norms. The sidecar has F32 projections. The converter must explicitly exclude `ffn_gate_inp` and `ffn_gate_inp_shexp` from any norm offset logic.

### 4. Memory Implications: `kDrafterMib` Constant

**Proposal Claim:** "If the drafter is 1.5 GB, this constant must be increased."
**Maintainer Review:** **Correct, but the estimate is likely wrong.**

*   **Source Fact:** `src/program/generate.cpp` line 2299: `const int64_t kDrafterMib = 1000;`.
*   **Calculation:**
    *   Current Q2_0 Experts: ~0.71 GB.
    *   Current Dense: ~0.18 GB.
    *   Total: ~0.89 GB. The constant 1000 MiB (~0.97 GB) is a safe upper bound.
    *   Sidecar Q4_K Experts: Q4_K is ~4.5 bits/weight. Q2_0 is ~2.25 bits/weight. Expert size doubles to ~1.42 GB.
    *   Sidecar Dense: If converted to Q8_0 (as proposed in Option A), dense size remains ~0.18 GB.
    *   Total Option A: ~1.6 GB.
    *   Total Option B (Native Q4_K): Experts ~1.42 GB + Dense (Q4_K/Q6_K mixed, likely smaller than Q8_0 but larger than Q2_0) ~0.15 GB. Total ~1.57 GB.
*   **Correction:** The drafter size will increase by ~700 MiB. `kDrafterMib` must be increased to at least 1700-1800 MiB.
*   **Impact:** On the RTX 4060 (8 GB), this is a massive regression. The `--split-skip-if-fits` logic (line 2139) will likely fail to skip the split, forcing a layer split even when it might fit, or causing OOM if the split logic doesn't account for the larger drafter on the last stage. The proposal underestimates the impact on the 8 GB card.

### 5. Missing Requirement: Tensor Name Mapping Completeness

**Proposal Claim:** "Map sidecar names to Strata names... `blk.48.attn_q_proj.weight` -> `self_attn.q_proj.weight`."
**Maintainer Review:** **Incomplete.**

*   **Source Fact:** `mtp.cpp` line 201-203 lists required tensors: `fc_embedding`, `fc_hidden`, `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`, `mlp.shared_expert.gate_proj`, `mlp.shared_expert.up_proj`, `mlp.shared_expert.down_proj`.
*   **Sidecar Fact:** The sidecar has `blk.48.attn_k/q_norm`, `attn_k/v/q/output`. It has `ffn_gate/up/down_shexp`.
*   **Missing:**
    *   **Indexer:** `mtp_rt.py` line 30 includes `self_attn.indexer.index_qk_proj.weight` in the Q8 set. The sidecar prompt says "There are unused indexer tensors." The converter must *ignore* these, but the runtime *requires* them?
    *   **Check:** `mtp.cpp` line 201-203 does *not* list `indexer` in the `required` array. However, `mtp_rt.py` line 30 *does* include it in the `Q8` set. If the runtime doesn't require it, why does the packer quantize it? If the runtime *does* use it (e.g., in `mtp.cpp` lines not shown, or in `verify.cpp`), the sidecar's "unused" indexer tensors might be needed.
    *   **Hyper-Connections:** The sidecar has `hc_attn/ffn_down/up/inject/norm`. The current `mtp_rt.py` does *not* list these in the `Q8` set (line 29-31). It keeps them as BF16 (line 96). The sidecar has them as Q4_K/Q6_K/Q5_0. The converter must convert these to BF16 to match the current runtime's expectation (since `mtp.cpp` likely expects BF16 for non-Q8 tensors). The proposal mentions this but doesn't emphasize that *all* non-Q8 dense tensors must be converted to BF16, not just kept as-is.

### 6. Verified Source Facts vs. Untested Expectations

| Item | Status | Evidence |
| :--- | :--- | :--- |
| **Q2_0 Blob Layout** | **Verified** | `expert.hpp` lines 45-51 define `BLOB` and offsets. `mtp_rt.py` line 57 asserts this size. |
| **Runtime Requires Q8_0 for Dense** | **Verified** | `mtp.cpp` line 204 checks `q8(n)` for specific tensors. `mtp_rt.py` line 92 converts them to Q8_0. |
| **Norms Need +1.0 Offset** | **Verified** | `mtp_rt.py` line 86 adds 1.0 to F32 tensors. |
| **Sidecar Norms Have Offset** | **Assumed** | Prompt states "source GGUF norm scales already include their Gemma +1 offset." |
| **Sidecar F32 Projections Exist** | **Verified** | Prompt states `ffn_gate_inp` and `ffn_gate_inp_shexp` are F32. |
| **`native_expert_grouped` Supports Q4_K** | **Unverified** | `iq_kernels.hpp` line 51 says `native_expert_supported` checks types. `iq_kernels.cu` line 1417 implements it. It *likely* supports Q4_K, but `mtp.cpp` does not call it. |
| **`eh_proj` Split is Trivial** | **Unverified** | No source code shows how `fc_embedding` and `fc_hidden` are used relative to `eh_proj`. The row/column order is a guess. |
| **RTX 4060 Can Handle 1.6 GB Drafter** | **Unverified** | Depends on KV cache size and other allocations. `kDrafterMib` is currently 1000. |

### Corrected Proposal Summary

1.  **Abandon Option B (Native Q4_K) for now.** It requires deep runtime changes to `mtp.cpp` and `iq_kernels.cu` integration that are out of scope for a "proposal and review."
2.  **Refine Option A (Re-quantize to Q2_0):**
    *   **Converter:** Create `tools/mtp_rt_sidecar.py`.
    *   **Experts:** Dequantize Q4_K/Q8_0 experts to FP32, then re-quantize to Q2_0 using `mtp_pack.py::q2_0`.
    *   **Dense:**
        *   Identify `eh_proj` and split it into `fc_embedding` and `fc_hidden` (verify row order!).
        *   Convert all dense tensors (Attention, Shared Expert) to Q8_0 using `mtp_rt.py::q8_0`.
        *   Convert Hyper-Connection tensors to BF16.
        *   **Crucial:** For F32 tensors, check if they are Norms or Projections.
            *   If Norm (`enorm`, `hnorm`, etc.): Do **not** add 1.0 (sidecar already has it).
            *   If Projection (`ffn_gate_inp`): Keep as F32 (or convert to BF16 if runtime expects BF16 for non-Q8? `mtp_rt.py` keeps non-Q8 as BF16, but F32 norms are F32. The runtime `mtp.cpp` likely expects F32 for norms and BF16 for others. The sidecar's F32 projections must be converted to BF16 to match the "BF16" kind in `dense.txt` if they are not in the Q8 list).
    *   **Indexer:** Ignore sidecar indexer tensors. Ensure the runtime does not require them (check `mtp.cpp` for `indexer` usage).
3.  **Runtime:** No changes to `mtp.cpp` or kernels.
4.  **Memory:** Update `kDrafterMib` in `generate.cpp` to ~1800 MiB. Test on RTX 4060.
5.  **Validation:**
    *   Verify `eh_proj` split order against a reference llama.cpp run.
    *   Verify norm values (no double offset).
    *   Verify router weights (no accidental offset).

This corrected approach minimizes runtime risk by leveraging the existing Q2_0 pipeline, at the cost of increased conversion time and potential accuracy loss from double quantization (Q4_K -> FP32 -> Q2_0).
