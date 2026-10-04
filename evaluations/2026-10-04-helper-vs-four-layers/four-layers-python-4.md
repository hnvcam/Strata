# Review and Correction: Q4_K_M MTP Adaptation Proposal

This review corrects the previous proposal against the clarified requirement: **Routed experts must remain Q4_K (gate/up) and Q8_0 (down).** The previous proposal correctly identified the need for a native expert path but contained critical errors in tensor mapping, axis interpretation, and runtime integration details.

## 1. Critical Corrections to Tensor Mapping and Axes

### A. Input Projection Splitting (`nextn.eh_proj`)
*   **Previous Error:** Assumed splitting `nextn.eh_proj` [5120, 2560] into two [2560, 2560] matrices was straightforward.
*   **Correction:**
    *   **Axis Check:** GGUF shapes are innermost-first. `nextn.eh_proj.weight` shape `[5120, 2560]` means **5120 output rows** and **2560 input columns**.
    *   **Spec Fact:** "each output row concatenating the embedding input columns followed by hidden input columns." This phrasing is ambiguous in the spec but standard for fused projections. Usually, `y = W @ x`. If `x` is `[emb; hidden]`, then `W` is `[Out, In_emb + In_hidden]`.
    *   **Runtime Expectation:** `mtp.cpp` expects `fc_embedding` and `fc_hidden`. These are likely applied as `y_emb = W_emb @ x_emb` and `y_hid = W_hid @ x_hid`.
    *   **Mapping:** The sidecar `nextn.eh_proj` is likely a single matrix $W_{fused}$ of shape $[5120, 2560]$. Wait, if input is 2560 (emb) + 2560 (hidden) = 5120, the shape should be $[2560, 5120]$ (Out x In).
    *   **Spec Re-read:** "blk.48.nextn.eh_proj.weight is Q4_K, shape [5120,2560]".
        *   If Shape is `[5120, 2560]` (Innermost first: 5120 is innermost? No, GGUF is innermost first. So 5120 is the *first* dimension listed, which is the innermost/fastest varying? No, `gguf_reader.py` line 101 reads `n_dims` then `shape`. Standard GGUF is `[cols, rows]` or `[inner, outer]`?
        *   `mtp_pack.py` line 169: "ggml order is innermost-first".
        *   If shape is `[5120, 2560]`, and it's a weight matrix $W$, usually $W$ is $[Out, In]$. If innermost is first, then $In=5120, Out=2560$? Or $Out=5120, In=2560$?
        *   Let's check `fc_embedding`. In `mtp_rt.py` line 29, it's in `Q8`. In `mtp.cpp` line 201, it's required.
        *   If `eh_proj` is `[5120, 2560]`, and it projects *from* 5120 (emb+hid) *to* 2560 (hidden state), then the shape is `[2560, 5120]` in standard math, but `[5120, 2560]` in GGUF (innermost first).
        *   **Conclusion:** The sidecar provides a **single fused projection** from 5120 inputs to 2560 outputs.
        *   **Runtime Conflict:** `mtp.cpp` likely expects two separate projections or a specific layout. If it expects `fc_embedding` [2560, 2560] and `fc_hidden` [2560, 2560], we cannot simply split `eh_proj` [5120, 2560] because the input dimension is fused.
        *   **Correction:** The converter **cannot** split `eh_proj` into two independent weights if the runtime applies them to separate inputs. The runtime must be changed to accept a single fused weight `nextn.eh_proj` and apply it to a concatenated input `[emb; hidden]`. **This is a major runtime change.**
        *   **Alternative:** If the runtime *can* be fed a concatenated input, the converter just renames `nextn.eh_proj` to a new name (e.g., `fc_fused`) and the runtime must be updated to concatenate `emb` and `hidden` before the MMV.

### B. Expert Tensor Names
*   **Previous Error:** Assumed `ffn_gate_exps` maps to `gate`.
*   **Correction:**
    *   Sidecar: `blk.48.ffn_gate_exps.weight` [2560, 640, 512].
    *   Strata Runtime: Expects experts to be indexed by ID. The layout in `NativeExpertLayout` is per-expert.
    *   **Mapping:** The converter must iterate over the 512 experts. For expert `e`, extract the slice `[:, :, e]` from `ffn_gate_exps`.
    *   **Axis Check:** Shape `[2560, 640, 512]`. Innermost is 2560? No, `gguf_reader` says innermost first. So 2560 is the fastest varying. This implies the tensor is stored as `[Expert, FF, Emb]` or `[Emb, FF, Expert]`?
    *   Standard GGUF for MoE: `experts.gate_up_proj` is often `[NE, 2*FF, H]`.
    *   Sidecar Spec: `[2560, 640, 512]`. If innermost is 2560, then the memory layout is `Expert[512] -> FF[640] -> Emb[2560]`.
    *   `NativeExpertLayout` expects `[gate rows | up rows | down rows]` per expert.
    *   **Converter Task:** For each expert `e`, extract `gate[e]` (shape 2560x640), `up[e]`, `down[e]`. Reorder into the native blob.

## 2. Expert Layout and Dispatch Corrections

### A. Native Expert Layout
*   **Fact:** `native_expert_layout` (`iq_kernels.hpp` line 41) defines `[gate | up | down]`.
*   **Correction:** The converter must produce `experts_native.bin` where each expert block is:
    1.  `gate` rows (Q4_K blocks)
    2.  `up` rows (Q4_K blocks)
    3.  `down` rows (Q8_0 blocks)
*   **Dispatch:** `mtp.cpp` must call `native_expert_grouped` with `gu_type=12` (Q4_K) and `d_type=8` (Q8_0).

### B. Activation Format
*   **Fact:** `native_expert_grouped` requires `q8_1` activations.
*   **Correction:** `mtp.cpp` line 532 `quantize_q8_0_scaled` must be replaced with `quantize_q8_1_rows`. The buffer `hit_xq_` must be resized to `N/32 * 36` bytes.

## 3. Compatibility Checks

1.  **Kernel Support:** Verify `native_expert_supported(12, 8, 2560, 640)` returns true.
2.  **Norm Offset:** Verify sidecar norms are pre-offset. Converter must **not** add 1.0.
3.  **Router:** Sidecar `ffn_gate_inp` is F32. Runtime expects BF16. Converter must cast to BF16.
4.  **HC Weights:** Sidecar HC weights are quantized. Runtime expects BF16. Converter must dequantize to BF16.

## 4. Byte/Memory Estimates

*   **Experts:**
    *   Gate/Up (Q4_K): $512 \times 2 \times (2560 \times 640 / 256) \times 144 \text{ bytes} \approx 1.47 \text{ GB}$.
    *   Down (Q8_0): $512 \times (2560 \times 640 / 32) \times 34 \text{ bytes} \approx 0.87 \text{ GB}$.
    *   Total Experts: **~2.34 GB**.
*   **Dense:**
    *   Attention/Shared (Q4_K/Q6_K/Q8_0): ~200 MB.
    *   HC (BF16 after dequant): ~50 MB.
    *   Norms/Router: ~10 MB.
    *   Total Dense: **~260 MB**.
*   **Total MTP VRAM:** **~2.6 GB**.

## 5. Final Proposal Summary

1.  **Converter (`mtp_rt_v2.py`):**
    *   Extract experts into `NativeExpertLayout` (Q4_K/Q8_0).
    *   **Do not split** `nextn.eh_proj`. Rename to `fc_fused` or similar.
    *   Dequantize HC to BF16.
    *   Cast Router to BF16.
    *   Copy Norms (no +1.0).
    *   Write `dense_native.txt` with correct types.
2.  **Runtime (`mtp.cpp`):**
    *   Detect native format.
    *   **Change Input Path:** Concatenate `emb` and `hidden` into a 5120-dim vector. Apply `fc_fused` (Q4_K) via `iq_mmvq`.
    *   **Change Activation:** Use `q8_1` for experts.
    *   **Change Expert Kernel:** Use `native_expert_grouped`.
    *   **Change Dense Kernels:** Use `iq_mmvq` for Q4_K/Q6_K dense tensors.
3.  **Validation:**
    *   Test on 5070 Ti first.
    *   Verify `native_expert_supported`.
    *   Check VRAM on 4060 (likely OOM).

**Uncertainty:** The exact axis order of `nextn.eh_proj` and whether the runtime can easily be modified to accept a fused input. If not, the sidecar is **incompatible** without significant runtime refactoring of the input stage.
