# Final comparison: 4060 expert helper versus fully cached layers 0–3

The helper produced the better marketing article in this sample and decoded the Python sequence 2.3% faster. The four-layer split read the identical initial source prompt 12.8% faster and finished the matched Python sequence 16.2% sooner, while producing 16.5% fewer output tokens. Its final proposal handled activation and scratch buffers better, but both proposals contain blocking omissions. These results do not establish one placement as the best for every workload.

Measured 2026-10-04 on the Intel i5-13500, 64 GB RAM, RTX 5070 Ti 16 GB and RTX 4060 8 GB, Linux. Only the two requested placements were run. Both used the current patched build/strata executable, SHA-256 20e4e2d88234dc02c8d2e4e5a0ced9fa352b1cc60cd9ec7f4177f66b8caa0903.

## Confirmed placements

| Setup | RTX 5070 Ti | RTX 4060 |
|---|---|---|
| Expert helper | All 48 layers, output head, Q2 MTP; 2513 cached experts (5.66 GiB) | Computes 2800 additional experts (6.31 GiB), returns results through pinned host rows |
| Four-layer split | Layers 4–47, output head, Q2 MTP; 2939 cached experts (6.58 GiB) | Layers 0–3; all 2048 experts cached (4.99 GiB) |

Startup logs confirmed these placements and full 248320-token draft heads on every server start. Main-model experts absent from the GPU caches use the CPU/RAM path. Prompt buffers can temporarily borrow stage cache slots, which the engine refills before decode.

## Saved tasks and matched settings

The exact LoopDay marketing brief, five Python proposal/review prompts, source snapshot and rubric were copied from ../2026-10-03-iq3-s-vs-iq4-xs without changing their bytes. The same material was also used in the saved 27B comparison. The Python task is a Q4_K_M MTP import/runtime proposal based on supplied source; it is not an executable unit-test suite. Only the final corrected proposal is scored. No proposed implementation was applied.

IQ4_XS, installed Q2 MTP with all 248320 draft IDs, 131072 context capacity, int8 KV, prefill=512, spec=4, cutoff=.70, 1248 MiB reserve, PCIe fraction=.20, suffix drafting disabled. Sampling: temperature=.2, top_p=.95, top_k=20, seed=1234; thinking disabled. No additional warmup or benchmark prompts. The system/user messages and named settings match between placements; each conversation retains its own assistant answers.

The helper stage-2 review reached 4096 tokens. The saved paired-continuation rule required an identical continuation for both placements; the four-layer first continuation then reached 2048 tokens. Both received a second identical continuation with a 6144-token budget. This second budget is the only output-limit change. Both continuation turns read cold histories. The helper stages 3–5 produced before the second continuation were archived in superseded-one-continuation and regenerated with the matched external sequence. All final deliverables completed. The 16 matched requests include both continuations; the three superseded helper requests are excluded from task rates and times. [Full adjustment record](test-adjustments.json).

## Quality results

| Task | Expert helper | Four-layer split |
|---|---:|---:|
| Marketing article | 88/100; 771 words | 75/100; 974 words |
| Corrected Python proposal | 51/100 | 56/100 |

These are subjective rubric scores for one sampled article and one technical conversation per placement. They are not general quality rankings. The concrete errors below carry more weight than a small score difference.

### Marketing

The helper article follows the brief more closely: 771 words meets the 600–800 requirement, and it includes the account/internet requirements for optional sync. The four-layer article is 974 words, omits those sync requirements, and adds unprovided notification/weekly-review behavior. Both correctly state the prices, Free project/task restrictions, platforms, calendar/AI limitations and download CTA. Both use familiar productivity phrases and feature-oriented sections. [Category scores and evidence](marketing-review.json).

### Corrected technical proposals

**Expert helper: 51/100.**

- Identifies legacy Q2 expert layout incompatibility and retains supplied Q4_K gate/up and Q8_0 down bytes without the old forbidden fallback.
- Uses contiguous gate/up/down native expert layout and native_expert_supported/layout/grouped helpers.
- Skips already-offset norms, converts F32 router and Q4_K/Q6_K/Q5_0 hyper-connections to BF16, and converts attention/shared dense projections to Q8_0.
- Identifies the Q8_1 activation requirement and proposes a mode-specific quantizer switch. Does not claim implementation or executed tests.

Remaining issues:

- **Under-sized Q8_1 activation buffer:** Section 4.2 sizes Q8_1 as 34 bytes per 32 values. The supplied tools/gguf_reader.py BLOCK_GEOMETRY defines Q8_1 as 36 bytes per 32 values; Q8_0 is 34. The stated native input buffer size is wrong.
- **Fixed expert-address stride and native scratch allocation omitted:** Native allocation and dispatch are changed, but the cpu::BLOB stride in moe_group_resident is unaddressed. Scratch is mentioned only in a memory-leak test. The supplied src/core/mtp.cpp:530 uses the fixed Q2 stride. The native path needs its 3584000-byte expert stride and native_expert_scratch_bytes sizing.
- **Incomplete source tensor mapping:** Naming normalization strips blk.48 and refers to self_attn.*, mlp.shared_expert.* and mlp.gate.weight without mapping the supplied attn_*, ffn_*_shexp and hc_* names. The 1-D shared gate is omitted. A complete map must include blk.48.attn_q/k/v/output, ffn_gate/up/down_shexp, ffn_gate_inp_shexp and the loader's exact hyper-connection names.
- **Fused projection axis remains imprecise:** Describes an inner dimension of 2560 and block-boundary 10, without stating the decoded 2560-by-5120 matrix and per-row input-column split. Its scope review incorrectly suggested splitting decoded rows. The specimen is innermost-first: decoded rows are 2560, input columns are 5120. Split the first and last 2560 input columns in each output row. The block boundary is compatible, but shape semantics need an explicit correction.
- **Incomplete format, import and last-GPU budgeting checks:** Detects native mode from q4_k dense kinds and gives about 2.2 GB for MTP, without versioned geometry/tokenizer/bounds checks, staged imports, exact expert bytes, full head/KV/cache accounting or generate.cpp budget changes. Native experts alone use 1835008000 bytes (1750 MiB), up 1075 MiB from Q2 experts. The full draft head, KV/work buffers, main-model dense/cache and reserve are additional. MTP placement follows the last-stage order.

The regenerated final incorporates the activation-format switch and removes the forbidden down-requantization fallback. It preserves the wrong 34-byte Q8_1 size from its review and does not resolve the address stride, scratch sizing or exact tensor mappings.

**Four-layer split: 56/100.**

- Correctly identifies the legacy Q2 layout and activation mismatch and preserves Q4_K gate/up and Q8_0 down expert bytes.
- Specifies Q8_1 quantization, correctly sizes its blocks at 36 rather than 34 bytes, and explicitly uses native_expert_scratch_bytes.
- Allocates experts using native layout bytes and dispatches native_expert_grouped with native activations and scratch.
- Names the actual ffn_gate_inp router source, converts it to BF16, handles Q5_0 hyper-connections, skips unused indexer tensors and avoids a second norm offset.
- Uses separate native dense/expert files and an explicit native index detection path. Does not claim changes or executed tests.

Remaining issues:

- **Incorrect source tensor names and incomplete mapping:** Uses blk.48.self_attn.q_proj.weight as a sidecar source and mapping example. It leaves the shared-gate and exact hyper-connection destination names unspecified. The supplied sidecar source is blk.48.attn_q.weight, with attn_k/v/output companions. The converter also needs ffn_gate_inp_shexp and exact loader hyper-connection names. The named conversion example would not find its source tensor.
- **Fixed expert address stride omitted:** Changes expert allocation to NativeExpertLayout::bytes but does not change the cpu::BLOB stride passed to moe_group_resident. The native path must group expert addresses using its 3584000-byte stride rather than the 1382400-byte Q2 stride in the supplied src/core/mtp.cpp:530.
- **Fused projection split is not specified per output row:** Says block-aligned raw slicing is safe but does not state the decoded 2560-by-5120 shape and input-column split. Its preceding scope review first reverses the axes and then wrongly says independent projections cannot represent the fused input projection. The supplied geometry is unambiguous: split the first and last 2560 input columns in each of 2560 output rows. Block alignment alone does not specify the correct slicing procedure.
- **Wrong expert storage estimate and incomplete last-GPU budget:** Estimates about 2.3 GB for experts and 2.6 GB total MTP, with about 840 MB for current Q2. It does not include the full draft head, complete KV/workspace, target expert cache or generate.cpp budget update. Exact native expert storage is 1835008000 bytes (1750 MiB), not about 2.3 GB. The measured Q2 draft layer is 949 MiB plus a separate 497.3 MiB full head in these runs. Complete per-stage budgeting is required before any fit claim.
- **Incomplete converter and compatibility validation:** Proposes checking output types/size and kernel execution but omits complete source/target geometry and tokenizer checks, tensor offset/bounds validation, versioned descriptors and staged import safety. Header changes for added state/scratch are not listed. Shared sidecar compatibility must be checked before consuming target embedding/head weights. Native file detection alone does not validate the runtime binary contract or a safe import.

The final correctly incorporates the 36-byte Q8_1 buffer and native scratch changes, unlike the helper final. Its review mishandles the fused projection's axes and linear decomposition; the final resumes a split proposal without explicitly resolving those errors. Its incorrect expert byte estimate remains.

Both proposals need corrections before implementation. Exact reference facts: Q8_1 is 36 bytes per 32 values; Q8_0 is 34. Native expert stride is 3584000 bytes, with 1835008000 bytes for 512 experts (1750 MiB), 1075 MiB above legacy Q2 expert storage. Decoded eh_proj is 2560 output rows by 5120 input columns, split into embedding and hidden input columns per row. The full draft head, KV/work buffers, target cache and reserve require additional memory. The Q2 draft layer in these runs logged 949 MiB plus a separate 497.3 MiB full draft head. [Saved reference facts](review-facts.json), [source snapshot](source-snapshot.md).

## Observed performance

| Setup / task | Output tokens | Decode tok/s | Prefill seconds | Request wall seconds | MTP acceptance |
|---|---:|---:|---:|---:|---:|
| helper / marketing | 951 | 41.36 | 3.56 | 26.74 | 74.39% |
| helper / python | 20087 | 43.81 | 612.23 | 1075.29 | 87.71% |
| four-layers / marketing | 1190 | 42.52 | 3.15 | 31.37 | 70.94% |
| four-layers / python | 16767 | 42.84 | 505.64 | 901.32 | 87.35% |

The first Python request uses identical cold input in both placements:

| Setup | Input tokens | Prefill tok/s | Prefill seconds | Decode tok/s | Request wall seconds |
|---|---:|---:|---:|---:|---:|
| helper | 24796 | 190.8 | 129.96 | 41.2 | 220.87 |
| four-layers | 24796 | 215.3 | 115.17 | 40.2 | 202.52 |

Decode rates divide total output tokens by summed generation time; warmup is absent. Prefill rates count only tokens actually read, and reused tokens are reported separately. Request wall times include prefill and generation, excluding startup. Different answer lengths and assistant histories make task wall times observed outcomes rather than equal-output speed comparisons. The continuation restarts and original cold scope-review restart also add source-reading work. This is one run per placement, not a full 128K test or a significance claim. [Every request, prefill rate, reused token count and startup/elapsed time](timing.md).

## State and artifacts

Production config, original and patched engines, expert profile, full draft vocabulary and watched source files were unchanged by hash. The temporary servers stopped and production remains unloaded.

```text
index, name, memory.used [MiB]
0, NVIDIA GeForce RTX 5070 Ti, 2 MiB
1, NVIDIA GeForce RTX 4060, 2 MiB
```

[Verification](comparison-verification.json), [timings and counters](summary.json), [full run manifest including superseded work](run-manifest.json), [rubric scores](scores.json). Completed articles: [helper](helper-marketing-1.md), [four layers](four-layers-marketing-1.md). Scored final proposals: [helper](helper-python-5.md), [four layers](four-layers-python-5.md). Exact request and response JSON files are saved beside each output.
