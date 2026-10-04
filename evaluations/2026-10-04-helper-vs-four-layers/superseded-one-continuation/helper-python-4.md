### Revised Proposal: Native Q4_K_M MTP Integration (Mixed Precision Experts)

This proposal corrects the previous review to strictly adhere to the requirement: **Gate/Up must remain Q4_K, Down must remain Q8_0.** It focuses on the specific technical gaps in `src/core/mtp.cpp` and `include/strata/kernels/iq_kernels.hpp` regarding mixed-type expert support and dense tensor handling.

#### 1. Critical Correction: Expert Layout and Dispatch

**The Constraint:**
The sidecar provides `ffn_gate_exps` (Q4_K), `ffn_up_exps` (Q4_K), and `ffn_down_exps` (Q8_0).
The current runtime (`src/core/mtp.cpp`) uses `moe_grouped_s2` which expects a legacy Q2_0 blob.
The target kernel is `native_expert_grouped` (`include/strata/kernels/iq_kernels.hpp` line 59).

**Source Fact:**
`native_expert_grouped` accepts a `NativeExpertLayout` struct (line 41) which explicitly supports separate `gu_type` and `d_type`.
`native_expert_layout` (line 1423 in `iq_kernels.cu`) calculates offsets based on these types.

**Correction to Previous Proposal:**
The previous proposal suggested converting Down to Q4_K if mixed support was missing. **This is now forbidden.**
Instead, we must verify and enable the mixed path.

**Implementation Requirement:**
1.  **Conversion Tool (`tools/mtp_rt_native.py`):**
    *   Extract raw Q4_K blocks for Gate and Up.
    *   Extract raw Q8_0 blocks for Down.
    *   Concatenate into a single blob per expert: `[Gate Blocks] [Up Blocks] [Down Blocks]`.
    *   **Crucial:** Do not interleave Gate/Up rows as in the legacy Q2_0 blob. `native_expert_grouped` expects contiguous blocks for each role.
    *   Write to `experts_native.bin`.

2.  **Runtime Loader (`src/core/mtp.cpp`):**
    *   Detect "Native" mode.
    *   Construct `NativeExpertLayout L`.
    *   Set `L.gu_type = 12` (Q4_K).
    *   Set `L.d_type = 8` (Q8_0).
    *   Set `L.n_embd = 2560`, `L.n_ff = 640`.
    *   Call `native_expert_layout(12, 8, 2560, 640)` to get correct byte offsets.
    *   **Dispatch Change:** Replace `moe_grouped_s2` with `native_expert_grouped`.
    *   **Activation Format:** `native_expert_grouped` expects `x_q8_1` (Q8_1 activations, 32 values/scale). The current MTP runtime uses `hit_xq_` which is Q8_0 (32 values/scale, no sum).
    *   **Correction:** The MTP runtime must be updated to generate **Q8_1** activations for the expert layer if `native_expert_grouped` strictly requires Q8_1. Check `iq_kernels.hpp` line 23: `quantize_q8_1_rows`. The current `mtp.cpp` line 532 uses `quantize_q8_0_scaled`. **This is a mismatch.** We must either:
        *   A) Update `mtp.cpp` to use `quantize_q8_1_rows` for the expert input.
        *   B) Verify if `native_expert_grouped` can accept Q8_0 (unlikely given the name and header comments). **Assume A is required.**

#### 2. Dense Tensor Handling and Naming

**Sidecar Specimen vs. Strata Expectations:**

| Sidecar Name | Type | Strata Expected Name | Strata Expected Type | Action |
| :--- | :--- | :--- | :--- | :--- |
| `blk.48.self_attn.q_proj.weight` | Q4_K | `self_attn.q_proj.weight` | Q4_K | Preserve Q4_K. Update loader to pass type 12 to `iq_mmvq`. |
| `blk.48.self_attn.k_proj.weight` | Q4_K | `self_attn.k_proj.weight` | Q4_K | Preserve Q4_K. |
| `blk.48.self_attn.v_proj.weight` | Q4_K | `self_attn.v_proj.weight` | Q4_K | Preserve Q4_K. |
| `blk.48.self_attn.o_proj.weight` | Q4_K | `self_attn.o_proj.weight` | Q4_K | Preserve Q4_K. |
| `blk.48.mlp.shared_expert.gate_proj.weight` | Q4_K | `mlp.shared_expert.gate_proj.weight` | Q4_K | Preserve Q4_K. |
| `blk.48.mlp.shared_expert.up_proj.weight` | Q4_K | `mlp.shared_expert.up_proj.weight` | Q4_K | Preserve Q4_K. |
| `blk.48.mlp.shared_expert.down_proj.weight` | Q8_0 | `mlp.shared_expert.down_proj.weight` | Q8_0 | Preserve Q8_0. |
| `blk.48.nextn.eh_proj.weight` | Q4_K | `fc_embedding.weight` + `fc_hidden.weight` | Q4_K | **Split** into two Q4_K tensors. |
| `blk.48.mlp.gate.weight` | F32 | `mlp.gate.weight` | BF16 | **Convert** F32 -> BF16. |
| `blk.48.*.norm.weight` | F32 | `*.norm.weight` | F32 | **Do NOT add +1.0**. Sidecar is pre-offset. |
| `blk.48.hc_*` | Q4_K/Q6_K | `hc_*` | BF16 | **Convert** to BF16 to avoid complex dispatch. |

**Correction on `eh_proj` Split:**
*   Shape: `[5120, 2560]`.
*   Strata expects `fc_embedding` `[2560, 2560]` and `fc_hidden` `[2560, 2560]`.
*   The sidecar description says: "each output row concatenating the embedding input columns followed by hidden input columns."
*   This implies the **inner dimension** (2560) is the concatenation axis? No, "output row" implies the first dimension (5120) is the output. "Concatenating... columns" implies the second dimension (2560) is the input.
*   Wait, `eh_proj` maps `[Embedding + Hidden] -> Output`.
*   If Input is `[Emb(2560) + Hid(2560)] = 5120`, and Output is `2560`.
*   Shape `[5120, 2560]` in GGUF (innermost first) means `rows=2560, cols=5120`?
*   GGUF shapes are innermost first. `[5120, 2560]` usually means `dim0=5120, dim1=2560`.
*   If it's a linear layer $W^T x$, and $x$ is 5120, $W$ is $2560 \times 5120$.
*   Strata's `fc_embedding` is likely $2560 \times 2560$.
*   **Action:** Split the 5120 dimension into two 2560 dimensions.
*   **Q4_K Block Alignment:** 5120 / 256 = 20 blocks. Split at 2560 (10 blocks). Safe.

#### 3. Activation and Scratch Formats

**Fact:** `native_expert_grouped` (line 59 in `iq_kernels.hpp`) takes `const void* x_q8_1`.
**Fact:** `mtp.cpp` line 532 uses `quantize_q8_0_scaled`.
**Correction:**
The MTP runtime must be modified to use **Q8_1** quantization for the expert input activations.
*   Replace `quantize_q8_0_scaled` with `quantize_q8_1_rows` (from `iq_kernels.hpp` line 23).
*   Update `hit_xq_` buffer size to accommodate Q8_1 blocks (34 bytes per 32 values, same as Q8_0, but includes a sum field).
*   **Scratch:** `native_expert_scratch_bytes` (line 54) must be called with the correct `cap_entries` and `n_ff`. The current `hit_scratch_` allocation in `mtp.cpp` line 273 uses `moe_hit_grouped_scratch_bytes`. This must be changed to `native_expert_scratch_bytes`.

#### 4. Compatibility Checks

1.  **Kernel Support:**
    *   Verify `native_expert_supported(12, 8, 2560, 640)` returns `true`.
    *   `n_embd % 256 == 0` (2560 % 256 = 0) -> OK.
    *   `n_ff % qd == 0` (640 % 32 = 0) -> OK.
    *   **Risk:** Does the CUDA kernel `launch_gu` (line 1455 in `iq_kernels.cu`) handle `gu_type=12` and `d_type=8` simultaneously? The switch statement in `native_expert_grouped` (line 1454) switches on `L.gu_type`. It likely calls a templated kernel. We must ensure the template instantiation exists for `Q4_K` gate/up and `Q8_0` down. If not, this proposal fails.

2.  **Dense MMVQ:**
    *   Verify `iq_mmvq` supports Q4_K for `n_embd=2560`.
    *   If not, convert dense attention weights to Q8_0 in the conversion tool.

#### 5. Byte and Memory Estimates

**Experts (512 experts):**
*   Gate/Up (Q4_K): $2 \times 2560 \times 640 \times \frac{4.5}{8}$ bytes $\approx 1.84$ MB/expert.
*   Down (Q8_0): $2560 \times 640 \times \frac{8}{8}$ bytes $\approx 1.64$ MB/expert.
*   Total per expert: $\approx 3.48$ MB.
*   Total Experts: $3.48 \times 512 \approx 1.78$ GB.

**Dense:**
*   Attention (Q4_K): $\approx 0.15$ GB.
*   Shared Expert (Q4_K/Q8_0): $\approx 0.05$ GB.
*   Router/Hyper (BF16): $\approx 0.02$ GB.
*   Norms/Embeddings: $\approx 0.05$ GB.
*   Total Dense: $\approx 0.27$ GB.

**Total MTP VRAM:** $\approx 2.05$ GB.
**Legacy Q2_0 MTP VRAM:** $\approx 0.9$ GB.
**Delta:** +1.15 GB.

**Impact on RTX 4060 (8 GB):**
*   If the drafter is on the 4060, it consumes 2.05 GB.
*   Main model (IQ4_XS) + KV Cache must fit in remaining ~5.95 GB.
*   This is tight. The drafter should be forced to the 16 GB card.

#### 6. Validation Plan

1.  **Unit Test:** Verify `native_expert_layout` returns correct offsets for Q4_K/Q8_0.
2.  **Conversion Test:**
    *   Split `eh_proj` and verify the two resulting tensors sum to the original when dequantized.
    *   Verify norms are not double-offset.
3.  **Runtime Test:**
    *   Load native MTP.
    *   Check for CUDA kernel launch errors (missing template instantiation for Q4_K/Q8_0).
    *   Compare logits against llama.cpp reference.

#### 7. Remaining Uncertainties

1.  **Kernel Instantiation:** Does `iq_kernels.cu` actually compile a kernel for `gu_type=12, d_type=8`? The `STRATA_GU_FMTS` macro (line 1276) must include Q4_K. If it only includes Q4_0/Q8_0, this will fail at runtime or compile time.
2.  **Q8_1 Activation:** Does `native_expert_grouped` strictly require Q8_1? If it can take Q8_0, we save the conversion overhead. The header says `x_q8_1`, so assume strict.
3.  **Performance:** Q4_K MMVQ may be slower than Q8_0 MMVQ due to dequantization complexity. Benchmark required.
