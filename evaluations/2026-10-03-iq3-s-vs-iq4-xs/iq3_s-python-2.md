This review identifies critical errors in the preceding proposal regarding tensor naming, expert layout, and runtime kernel assumptions. The proposal incorrectly assumes that the sidecar's expert tensors are fused or can be trivially mapped to the existing `Q2_0` blob layout, and it overlooks the specific structural differences between the `mtp.*` namespace and the `blk.48.*` namespace in the sidecar.

### 1. Critical Mistake: Expert Tensor Structure and Naming

**Proposal Claim:** The proposal suggests mapping `blk.48.ffn_gate_exps.weight` and `blk.48.ffn_up_exps.weight` to the expert gate/up roles, implying they can be processed similarly to `mtp.layers.0.mlp.experts.gate_up_proj`.

**Source Fact:**
*   **Current Pipeline (`tools/mtp_rt.py` lines 70-71):** Expects a single fused tensor `mtp.layers.0.mlp.experts.gate_up_proj` with shape `[512, 1280, 2560]` (NE, 2*FF, H). The `blob_of` function (lines 48-58) explicitly splits this fused tensor into gate and up rows (`gub[:FF]` and `gub[FF:]`) before interleaving them.
*   **Sidecar Specimen:** The sidecar provides **separate** tensors:
    *   `blk.48.ffn_gate_exps.weight`: Shape `[2560, 640, 512]` (H, FF, NE).
    *   `blk.48.ffn_up_exps.weight`: Shape `[2560, 640, 512]` (H, FF, NE).
    *   `blk.48.ffn_down_exps.weight`: Shape `[640, 2560, 512]` (FF, H, NE).

**Correction:**
The conversion tool cannot simply "read" the sidecar experts. It must:
1.  Load `ffn_gate_exps` and `ffn_up_exps` separately.
2.  **Transpose/Reshape** them to match the expected `[NE, FF, H]` or `[NE, H, FF]` orientation depending on the GGUF axis order (GGUF is innermost-first, so `[2560, 640, 512]` means 512 experts, each with 640x2560 weights).
3.  **Concatenate** gate and up into a fused `[NE, 2*FF, H]` structure *before* passing to the `blob_of` logic, OR modify `blob_of` to accept two separate arrays.
4.  The proposal's suggestion to "Dequantize... Requantize to Q2_0" is valid for format conversion, but the **layout mismatch** (separate vs. fused) is a hard blocker for the existing `blob_of` function which expects a single `gu` array.

### 2. Critical Mistake: Dense Weight Naming and `nextn` Handling

**Proposal Claim:** Map `blk.48.nextn.eh_proj.weight` to `fc_embedding.weight`.

**Source Fact:**
*   **Current Pipeline (`src/core/mtp.cpp` lines 201-204):** Requires `fc_embedding.weight` and `fc_hidden.weight` as separate Q8_0 tensors.
*   **Sidecar Specimen:** Provides `blk.48.nextn.eh_proj.weight` with shape `[5120, 2560]`. The description states: "each output row concatenating the embedding input columns followed by hidden input columns."
*   **Runtime Logic (`src/core/mtp.cpp` lines 258-260):** The runtime allocates `mixed_` (T*N) and uses `fc_embedding` and `fc_hidden` separately in the prefill/decode paths (implied by the separate tensor requirements).

**Correction:**
The sidecar's `eh_proj` is a **fused** projection (Embedding + Hidden). The current runtime expects **separate** `fc_embedding` and `fc_hidden` weights.
*   **Conversion Requirement:** The conversion tool must **split** `blk.48.nextn.eh_proj.weight` into two separate tensors:
    *   `fc_embedding.weight`: First 2560 columns (or rows, depending on orientation) of `eh_proj`.
    *   `fc_hidden.weight`: Last 2560 columns (or rows) of `eh_proj`.
*   **Verification:** The shape `[5120, 2560]` implies 5120 outputs (2560 emb + 2560 hidden) and 2560 inputs. The split must respect the "concatenating... embedding... followed by hidden" order.

### 3. Missing Requirement: Hyper-Connection (HC) Tensor Mapping

**Proposal Claim:** Mentions "Hyper-connection down/inject use Q4_K or Q6_K" but does not address the specific tensor names or the `hc_head_down/up/norm` fields.

**Source Fact:**
*   **Current Pipeline (`tools/mtp_rt.py` lines 9-11):** Keeps hyper-connection weights as BF16.
*   **Sidecar Specimen:** Uses `Q4_K` or `Q6_K` for `hc_attn/ffn_down/up/inject/norm` and `nextn.hc_head_down/up/norm`.
*   **Runtime (`src/core/mtp.cpp` lines 258-260):** Allocates `inj_`, `inj2_`, `lo_`, `rs_`, `bo_` buffers. The loader (`dense.txt`) expects specific names.

**Correction:**
The proposal fails to specify how to map the sidecar's `hc_*` tensors to the runtime's expected names. The conversion tool must:
1.  Identify `blk.48.hc_attn_down.weight`, `blk.48.hc_attn_up.weight`, etc.
2.  Rename them to match the `dense.txt` schema (e.g., `hc_attn_down.weight`).
3.  **Quantization Decision:** The current runtime expects BF16 for HC weights (line 96 in `mtp_rt.py`). The sidecar has Q4_K/Q6_K.
    *   *Option A:* Dequantize Q4_K/Q6_K to BF16 during conversion. This increases memory usage but maintains runtime compatibility.
    *   *Option B:* Modify the runtime to support Q4_K/Q6_K for HC weights. This is a C++ change, not a conversion change.
    *   *Recommendation:* For a "proposal and review only" exercise, **Option A** (Dequantize to BF16) is the safer path to avoid C++ changes, but it negates the memory savings of the sidecar's quantization.

### 4. Mistake: Kernel Support Assumptions

**Proposal Claim:** Suggests using `native_expert_grouped` for Q4_K experts if modifying C++.

**Source Fact:**
*   **`include/strata/kernels/iq_kernels.hpp` (lines 49-51):** `native_expert_supported` checks `gu_type` and `d_type`.
*   **`src/kernels/cuda/iq_kernels.cu` (lines 1417-1421):** `native_expert_supported` returns true if `qg > 0 && qd > 0`.
*   **`src/kernels/cuda/iq_kernels.cu` (lines 1273-1288):** `gu_qk` and `d_qk` use `STRATA_GU_FMTS` and `STRATA_D_FMTS` macros. The provided excerpt does **not** show the definition of these macros.
*   **`src/core/mtp.cpp` (lines 530-534):** The current MTP runtime uses `moe_grouped_s2`, **not** `native_expert_grouped`. `moe_grouped_s2` expects the custom `experts.bin` blob layout (Q2_0).

**Correction:**
The proposal incorrectly assumes `native_expert_grouped` is the drop-in replacement. The current MTP path uses `moe_grouped_s2`. To use Q4_K experts natively, one would need to:
1.  Change `src/core/mtp.cpp` to call `native_expert_grouped` instead of `moe_grouped_s2`.
2.  Verify that `STRATA_GU_FMTS` includes `Q4_K` (type 12). The `is_iq` function (line 1269) includes `t == 12`, so `Q4_K` is likely supported by the *dequantization* kernels, but `native_expert_grouped` might have specific constraints not visible in the excerpt.
3.  **Crucially:** The `moe_grouped_s2` kernel is optimized for the specific `Q2_0` blob layout. Switching to `native_expert_grouped` requires changing the data layout from the custom blob to raw GGUF blocks (or a different native layout), which is a significant refactor.

### 5. Missing Requirement: `nextn` Norms and Offsets

**Proposal Claim:** Mentions "source GGUF norm scales already include their Gemma +1 offset."

**Source Fact:**
*   **`tools/mtp_rt.py` (lines 85-87):** Applies `+1.0` to F32 norm weights.
*   **Sidecar Specimen:** States "source GGUF norm scales already include their Gemma +1 offset."

**Correction:**
If the sidecar norms *already* include the +1 offset, the conversion tool must **NOT** apply the `+1.0` offset again. The current `mtp_rt.py` logic (line 86) blindly adds 1.0. A new conversion tool must detect this metadata or be configured to skip the offset for the sidecar. Applying it twice would result in `w + 2.0`, breaking the model.

### 6. Memory Implications Correction

**Proposal Claim:** Estimates Q4_K MTP VRAM at ~4-5 GB.

**Source Fact:**
*   **Sidecar Experts:**
    *   Gate/Up: 512 experts * 2 * (2560 * 640 / 256) * 144 bytes = 512 * 2 * 10 * 144 = 1,474,560 bytes/expert? No.
    *   Calculation:
        *   Elements per expert gate: 2560 * 640 = 1,638,400.
        *   Blocks per expert gate: 1,638,400 / 256 = 6,400 blocks.
        *   Bytes per expert gate: 6,400 * 144 = 921,600 bytes.
        *   Gate + Up: 1,843,200 bytes.
        *   Down: 640 * 2560 = 1,638,400 elements. Blocks: 1,638,400 / 32 = 51,200 blocks. Bytes: 51,200 * 34 = 1,740,800 bytes.
        *   Total per expert: ~3.58 MB.
        *   Total 512 experts: ~1.83 GB.
*   **Dense Weights:**
    *   Attention (Q,K,V,O): 4 * 2560 * 2560 * (Q4_K ~0.5 bytes/val) ≈ 13 MB.
    *   Shared Expert: 3 * 2560 * 640 * (Q4_K ~0.5) ≈ 2.5 MB.
    *   Hyper-connections: Significant size, likely ~50-100 MB.
    *   `eh_proj`: 5120 * 2560 * 0.5 ≈ 6.5 MB.
*   **Total VRAM:** ~2.0 - 2.5 GB for weights. Plus KV cache and buffers.
*   **Comparison:** Current Q2_0 experts are ~0.7 GB. The Q4_K sidecar is ~2.5x larger in expert weights.

**Correction:** The proposal's estimate of 4-5 GB was too high for weights alone, but the **relative increase** is significant. The RTX 4060 (8 GB) might struggle if the main model uses >5.5 GB. The RTX 5070 Ti (16 GB) is safer.

### 7. Verified Source Facts vs. Untested Expectations

| Item | Status | Evidence |
| :--- | :--- | :--- |
| **Expert Format Mismatch** | **Verified** | `mtp_rt.py` expects `Q2_0` fused; Sidecar has `Q4_K` separate. |
| **Dense Naming Mismatch** | **Verified** | `mtp.cpp` expects `fc_embedding`; Sidecar has `nextn.eh_proj`. |
| **Norm Offset Double-Add** | **Verified Risk** | `mtp_rt.py` adds +1; Sidecar says +1 is included. |
| **`native_expert_grouped` Support** | **Untested** | `is_iq` includes Q4_K, but `moe_grouped_s2` is used in MTP. Switching kernels is untested. |
| **VRAM Fit on 4060** | **Untested** | Depends on main model size and KV cache. |
| **Acceptance Rate Gain** | **Untested** | Q4_K vs Q2_0 accuracy impact is not measured. |

### Final Recommendation

The preceding proposal is **insufficient** because it ignores the **structural** differences (fused vs. separate experts, fused vs. separate embedding projections) and the **metadata** conflict (norm offsets).

**Corrected Proposal Steps:**
1.  **New Conversion Tool:** `tools/mtp_rt_sidecar.py`.
2.  **Expert Handling:**
    *   Load `blk.48.ffn_gate_exps` and `blk.48.ffn_up_exps`.
    *   Dequantize to FP32.
    *   Concatenate into fused `[NE, 2*FF, H]` layout.
    *   Requantize to `Q2_0` (to match existing `moe_grouped_s2` kernel) OR keep as `Q4_K` and modify C++ to use `native_expert_grouped` (high risk).
    *   *Recommendation:* Requantize to `Q2_0` for compatibility, accepting precision loss.
3.  **Dense Handling:**
    *   Split `blk.48.nextn.eh_proj` into `fc_embedding` and `fc_hidden`.
    *   Rename `blk.48.self_attn.*` to `self_attn.*`.
    *   Rename `blk.48.mlp.shared_expert.*` to `mlp.shared_expert.*`.
    *   Rename `blk.48.hc_*` to `hc_*`.
    *   **Do NOT** add +1 to norms (check sidecar metadata).
    *   Dequantize Q4_K/Q6_K dense weights to BF16 (to match current `dense.bin` BF16 expectation) or Q8_0 (if MMVQ supports it).
4.  **Validation:**
    *   Verify `dense.txt` line count matches expected tensor count.
    *   Verify `experts.bin` size is `512 * BLOB`.
    *   Run a short generation to check for NaNs (indicating norm offset errors).
