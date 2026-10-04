dynamic types for these small, non-bottleneck tensors if the runtime does not already support them. This avoids modifying `src/core/mtp.cpp` kernel dispatch logic for minor weights.

### 8. Missing Requirement: `dense.txt` Format and Loader Parsing

**The Proposal Claims:**
> "Update `dense.txt` to include the actual GGML type ID... Replace calls to `q8()`... with a generic `get_tensor(type, offset)`"

**The Reality (Source Evidence):**
*   **`src/core/mtp.cpp` lines 158–165:** The loader parses `dense.txt` using `std::istringstream`. It expects: `name kind rows cols offset bytes`.
*   **`tools/mtp_rt.py` line 12:** The docstring says `kind = q8_0 | bf16 | f32`.
*   **The Conflict:** The current parser likely treats `kind` as a string. If we write integers (e.g., `12` for Q4_K), the existing `is >> t.kind` will read it as a string "12". The helper functions `q8()` and `bf16()` in `mtp.cpp` (not fully shown but implied by usage) likely check `if (t.kind == "q8_0")`. They will **fail** or return null for "12".
*   **Missing Requirement:** We must either:
    1.  **Modify the Loader:** Change `src/core/mtp.cpp` to parse `kind` as an integer and update all helper calls (`q8`, `bf16`, etc.) to use a unified `get_tensor_ptr(type, offset)`. This is a significant code change.
    2.  **Keep String Kinds:** Write `dense.txt` with string kinds like `q4_k`, `q8_0`, `bf16`. Then update the loader to recognize these new strings and map them to GGML types. This is less invasive but still requires loader changes.

**Correction:**
The proposal must explicitly state that **`src/core/mtp.cpp` must be modified** to handle new `kind` strings (e.g., `q4_k`) or integer types. The previous proposal's claim that we can just "pass the correct type" ignores that the *loader* currently filters and validates based on hardcoded string kinds.

### 9. Missing Requirement: Alignment and Padding in `dense.bin`

**The Proposal Claims:**
> "Ensure alignment (256 bytes) for GPU memory mapping."

**The Reality (Source Evidence):**
*   **`tools/mtp_rt.py` lines 97–99:**
    ```python
    pad = (-off) % 256
    f.write(b"\0" * pad)
    off += pad
    ```
    The current tool pads to 256 bytes.
*   **Sidecar Specimen:** GGUF files have their own alignment (usually 32 or 64 bytes, defined in header).
*   **The Conflict:** When extracting tensors from the sidecar GGUF, we must ensure that the *destination* `dense_native.bin` maintains the 256-byte alignment required by Strata's CUDA kernels (specifically for `iq_mmvq` which may use vectorized loads).
*   **Missing Requirement:** The conversion tool must replicate the 256-byte padding logic of `mtp_rt.py` when writing `dense_native.bin`. If we just dump raw GGUF blocks, they might be misaligned for Strata's kernels.

**Correction:**
The conversion tool must:
1.  Read tensor data from the sidecar.
2.  Calculate the current offset in `dense_native.bin`.
3.  Pad to the next 256-byte boundary.
4.  Write the tensor data.
5.  Record the *padded* offset in `dense.txt`.

### 10. Missing Requirement: Handling of `nextn` Prefix and Layer Index

**The Proposal Claims:**
> "Map `blk.48.*` names to Strata's internal expected names"

**The Reality (Source Evidence):**
*   **Sidecar Specimen:** Tensors are named `blk.48.nextn.eh_proj.weight`, `blk.48.self_attn.q_proj.weight`, etc.
*   **Strata Expectation:** `fc_embedding.weight`, `self_attn.q_proj.weight`, `mlp.shared_expert.gate_proj.weight`.
*   **The Conflict:** The sidecar uses `blk.48.` prefix and `nextn.` for some tensors. Strata expects no prefix (or `mtp.layers.0.` which is stripped in `mtp_rt.py` line 83).
*   **Missing Requirement:** The mapping logic must be robust.
    *   `blk.48.self_attn.q_proj.weight` -> `self_attn.q_proj.weight`
    *   `blk.48.nextn.eh_proj.weight` -> Split into `fc_embedding.weight` and `fc_hidden.weight`
    *   `blk.48.mlp.shared_expert.gate_proj.weight` -> `mlp.shared_expert.gate_proj.weight`
    *   **Crucial:** Verify that the sidecar does *not* have a `blk.48.` prefix on the *shared expert* or *router* tensors if they are named differently (e.g., `blk.48.ffn_gate_inp`). The specimen says "Router ffn_gate_inp... are F32". It does not explicitly say `blk.48.ffn_gate_inp`. We must assume the prefix is consistent or handle both cases.

**Correction:**
Implement a strict name normalization function in the conversion tool that strips `blk.48.` and `nextn.` prefixes and maps specific sidecar names to Strata's expected names. Fail loudly if a required tensor is missing.

### Final Corrected Proposal Summary

1.  **Conversion Tool (`tools/mtp_rt_native.py`):**
    *   Read sidecar GGUF using `gguf_reader.py`.
    *   **Experts:** Extract Q4_K gate/up and Q8_0 down. Concatenate into `[Gate][Up][Down]` blob per expert. Write to `experts_native.bin`.
    *   **Dense:**
        *   Split `eh_proj` (Q4_K) into `fc_embedding` and `fc_hidden` (Q4_K) via block slicing.
        *   Convert F32 router to BF16.
        *   Convert Q4_K/Q6_K/Q5_0 hyper-connections to BF16 (to avoid runtime kernel changes).
        *   Keep Q4_K/Q8_0 for attention/shared expert if `iq_mmvq` supports them; otherwise convert to Q8_0.
        *   **Skip `+1.0`** for norms.
        *   Write to `dense_native.bin` with **256-byte alignment**.
    *   **Index:** Write `dense.txt` with string kinds (`q4_k`, `bf16`, `f32`) and correct offsets.

2.  **Runtime Changes (`src/core/mtp.cpp`):**
    *   **Loader:** Detect `dense_native.bin` (via a marker in `dense.txt` or filename).
    *   **Parsing:** Update `dense.txt` parser to handle new `kind` strings (`q4_k`, etc.).
    *   **Dispatch:** Replace hardcoded `q8()`/`bf16()` calls with a generic `get_tensor_ptr(kind, offset)` that returns a pointer and type.
    *   **Kernels:** Update `iq_mmvq` calls to pass the dynamic type. Verify `native_expert_grouped` supports mixed Q4_K/Q8_0. If not, force all experts to Q4_K in conversion.

3.  **Validation:**
    *   **Unit Test:** Verify `eh_proj` split produces correct Q4_K blocks.
    *   **Integration:** Run MTP forward pass. Compare logits against a Q8_0-converted version of the same sidecar (using old tool) to ensure numerical consistency.
    *   **Memory:** Monitor VRAM on 8 GB card. Expect failure or heavy swapping. Recommend forcing drafter to 16 GB card.

4.  **What Cannot Be Concluded:**
    *   Whether `native_expert_grouped` CUDA kernel actually implements the mixed Q4_K/Q8_0 path correctly (source only shows struct, not kernel body).
    *   Whether `iq_mmvq` has optimized paths for Q4_K at `n_embd=2560`.
    *   Exact VRAM overhead of `native_expert_grouped` scratch buffers vs legacy Q2_0.
