### 8. Missing Requirement: Tensor Name Mapping and "Shared" Dependency Resolution

**Assumption in Proposal:** The converter can simply map sidecar names to runtime names.
**Source Fact:** The sidecar is explicitly described as a **"shared sidecar"** where "the target model supplies the embedding and output projection."
*   `src/core/mtp.cpp` (lines 201-204) hard-requires `fc_embedding.weight` and `fc_hidden.weight` to be present in the loaded `dense.bin`/`dense.txt`.
*   The sidecar specimen description lists `blk.48.nextn.eh_proj.weight` but **does not list** `fc_embedding` or `fc_hidden`.
*   **Critical Gap:** If the sidecar GGUF does not contain these tensors, the current `mtp_rt.py` logic (which iterates over all tensors in the GGUF) will fail to produce the required entries in `dense.txt`.
*   **Correction:** The converter (`tools/mtp_rt_q4.py`) cannot operate on the sidecar file in isolation. It must either:
    1.  **Require the main model GGUF path** as an argument to extract `fc_embedding` and `fc_hidden` (and potentially `output.weight` if the draft head uses it directly, though `mtp.cpp` seems to use a separate `dhead_` logic).
    2.  **Synthesize** these tensors if they are identical to the main model's embeddings (which is typical for MTP), but this requires reading the main model's GGUF header/data.
    3.  **Fail explicitly** if these tensors are missing, rather than producing a broken `dense.bin`.

### 9. Missing Requirement: `eh_proj` Shape and Concatenation Logic

**Assumption in Proposal:** Standard dense tensor handling.
**Source Fact:** The sidecar spec states: `blk.48.nextn.eh_proj.weight` is Q4_K, shape `[5120, 2560]`, with "each output row concatenating the embedding input columns followed by hidden input columns."
*   `mtp.cpp` expects `fc_embedding` and `fc_hidden` as separate tensors (lines 201-202).
*   The sidecar provides a **single fused** `eh_proj` tensor.
*   **Correction:** The converter must **split** the `eh_proj` tensor into two separate tensors:
    1.  `fc_embedding.weight`: The first 2560 columns (or rows, depending on GGUF axis order, but spec says innermost first, so likely the first half of the inner dimension).
    2.  `fc_hidden.weight`: The second 2560 columns.
*   **Quantization Issue:** The sidecar `eh_proj` is **Q4_K**. The current `mtp.cpp` loader (line 204) checks for `q8_0` for these specific tensors.
    *   If we keep them as Q4_K, we must update `mtp.cpp` to accept Q4_K for `fc_embedding`/`fc_hidden`.
    *   If we dequantize/requantize to Q8_0 in the converter, we lose precision and add conversion time.
    *   **Recommendation:** Update `mtp.cpp` to accept Q4_K for these tensors, consistent with the "Native" dense path.

### 10. Missing Requirement: Router and Hyper-Connection Type Handling

**Assumption in Proposal:** Dense tensors are handled generically.
**Source Fact:**
*   Sidecar: `ffn_gate_inp` (Router) is **F32**. `ffn_gate_inp_shexp` is **F32**.
*   Sidecar: Hyper-connection `down/inject` are Q4_K/Q6_K. `up` matrices are **Q5_0**.
*   `mtp.cpp` (line 526) uses `bf16_gemv_fp32_mmvf` for the router (`mlp.gate.weight`). This kernel likely expects BF16 or F32.
*   `mtp.cpp` (lines 258-259) allocates buffers for hyper-connections (`inj_`, `inj2_`, `lo_`).
*   **Correction:**
    1.  **Router:** The sidecar router is F32. The current code expects BF16 (implied by `bf16_gemv...`). We must ensure the converter writes F32 router weights and `mtp.cpp` dispatches to an F32-compatible GEMV kernel, or the converter converts F32 to BF16. Given F32 is available, using F32 is more accurate.
    2.  **Hyper-Connections:** The sidecar uses mixed types (Q4_K, Q6_K, Q5_0). The current `mtp.cpp` likely assumes a uniform type (BF16 or Q8_0) for these small matrices. We must verify if `mtp.cpp`'s hyper-connection kernels support mixed quantization types. If not, the converter must **dequantize all hyper-connection weights to BF16 or F32** to ensure compatibility, as implementing mixed-precision kernels for these small ops is high-effort/low-reward.

### 11. Missing Requirement: `dense.txt` Format Extension

**Assumption in Proposal:** `dense.txt` can carry new kinds.
**Source Fact:** `src/core/mtp.cpp` (lines 161-163) parses `dense.txt` with `is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes;`.
*   The `kind` string is currently used to select the kernel (e.g., `q8_0`, `bf16`, `f32`).
*   **Correction:** We must extend the parser and the subsequent dispatch logic in `mtp.cpp` to recognize `q4_k`, `q6_k`, `q5_0`, and `q8_0` (for down experts if they were dense, but they are experts).
*   **Crucial:** The `rows` and `cols` in `dense.txt` must reflect the **logical** shape (e.g., 2560x2560), not the **quantized block** shape. The current `mtp_rt.py` writes logical rows/cols. The new converter must do the same.

### 12. Validation Plan: Specific Tests

1.  **Unit Test: Converter Output**
    *   Input: Sidecar GGUF + Main Model GGUF (for embeddings).
    *   Output: `experts.bin`, `dense.bin`, `dense.txt`.
    *   Check: `dense.txt` contains `fc_embedding` (Q4_K), `fc_hidden` (Q4_K), `mlp.gate.weight` (F32).
    *   Check: `experts.bin` size matches `NativeExpertLayout` calculation for 512 experts.

2.  **Integration Test: Kernel Dispatch**
    *   Load the new RT files.
    *   Verify `mtp.cpp` logs "Native Expert Mode" (new log line).
    *   Verify `iq_mmvq` is called for `fc_embedding` (via profiler or debug log).
    *   Verify `native_expert_grouped` is called for experts.

3.  **Numerical Test: Draft Acceptance**
    *   Run a fixed prompt with Q2_0 MTP and Q4_K_M MTP.
    *   Compare acceptance rates. Q4_K_M should be higher.
    *   Compare draft token IDs. They should differ due to quantization noise, but the *sequence* should be similar.

4.  **Memory Test:**
    *   Monitor VRAM usage on the 8GB card.
    *   Ensure `kDrafterMib` adjustment prevents OOM.

### Final Conclusion for Maintainer

The proposal is **technically feasible** but **significantly underestimated** in scope. It is not a "sidecar import" but a **new runtime mode** for the MTP layer.

**Key Blockers to Resolve Before Implementation:**
1.  **Dependency on Main Model:** The converter must access the main model's GGUF to extract `fc_embedding`/`fc_hidden` if they are not in the sidecar.
2.  **Activation Format Change:** Switching from Q8_0 to Q8_1 activations for dense/expert kernels is a breaking change for the current `mtp.cpp` flow and requires careful refactoring.
3.  **CPU Support:** Native Q4_K experts have no CPU kernel. The feature must be GPU-only or require a slow fallback.
4.  **Static Memory Reservation:** `kDrafterMib` must become dynamic.

**Recommendation:**
Proceed with a **prototype** that:
1.  Assumes the sidecar *does* contain `fc_embedding`/`fc_hidden` (if not, add main model dependency).
2.  Dequantizes all dense tensors to **BF16** in the converter (to avoid implementing Q4_K dense dispatch in `mtp.cpp` initially). This simplifies the runtime change to only the **Expert** path.
3.  Implements only the **Native Expert** path in `mtp.cpp`, keeping
