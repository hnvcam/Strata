# Qwen3.8 27B UD-Q4_K_M: replay of the saved original tasks

The 27B model wrote the most accurate article of these three saved responses, but produced the weakest final technical proposal. **Flash-Next IQ4_XS still performed best on this particular code-review task.** None of the three technical proposals is ready to implement.

| Model | Marketing score | Final code-proposal score | Marketing decode tokens/s | Code decode tokens/s, five turns |
|---|---:|---:|---:|---:|
| Flash-Next IQ3_S | 80/100 | 45/100 | 57.80 | 54.88 |
| Flash-Next UD-IQ4_XS | 74/100 | 58/100 | 42.24 | 40.23 |
| **Qwen3.8 27B UD-Q4_K_M** | **84/100** | **35/100** | **41.83** | **51.50** |

Scores are subjective reviewer judgments using the unchanged saved rubric. This table compares the actual responses from one invented marketing task and one five-turn technical task, not general model capability. Prior scores and timings are preserved in the [original report](../2026-10-03-iq3-s-vs-iq4-xs/REPORT.md).

## Method and controls

The user supplied an already-loaded endpoint at `http://127.0.0.1:1234/v1`. It reported `/media/hnvcam/AI/LLAMA_Models/Qwen3.8-27B-UD-Q4_K_M.gguf`, 27,320,697,856 parameters, Q4_K Medium and a loaded context of **110,592 tokens**. Server build fingerprint: `b10964-b29c606e2`.

The tasks, source snapshot and rubric were copied byte for byte from the earlier comparison. The marketing task was the same 600–800-word article for fictional app LoopDay. The code task used the same actual project Python source and engine excerpts and asked for a Q4_K_M MTP import/runtime proposal, retaining Q2 compatibility. The requested implementation target inside that prompt remains the existing IQ4_XS Strata installation; it was not changed to the newly loaded 27B model.

The same five technical prompts were used in sequence: initial proposal, skeptical review, first final proposal, clarification/review requiring preservation of Q4_K gate/up and Q8_0 down, then corrected final proposal. The 27B model received that clarification even though its initial primary design already aimed to retain those types, to preserve the same external inputs and turn count. No model-specific coaching or extra correction prompt was added.

The final run used temperature **0.2**, top_p **0.95**, top_k **20**, seed **1234**, thinking disabled and the same output limits: 2048 tokens for marketing and 6144/4096/6144/4096/6144 for code. There were no truncations, continuations or separate reasoning outputs.

One implicit sampling difference was discovered: this server defaults `min_p` to 0.05, while the earlier Strata configs had no sampling overrides and the engine defaults it to 0. The requests were replayed with **explicit `min_p=0`**. An active `/slots` snapshot verifies the accepted seed, temperature, top_p, top_k, min_p and closed thinking prefix. The first run is archived in `superseded-default-min-p/` and excluded from the reported metrics. All six answer texts were byte-identical before and after this correction.

The same system/user messages at each corresponding turn are verified in [comparison-verification.json](comparison-verification.json). Later conversation bodies include each model's own answers, so assistant history necessarily differs. A matching seed across engines does not imply matching random draws.

**No project `bench/` runner or public benchmark suite was executed. No proposed code change was applied.** The runner only called the supplied inference endpoint and wrote evaluation artifacts. The user-loaded server was left running as loaded; model/server settings were not changed. Watched production source, configs, engine and full draft vocabulary hashes remained unchanged.

## Marketing: 84/100

Read the article: [qwen27b-marketing-1.md](qwen27b-marketing-1.md).

| Category | Maximum | Score |
|---|---:|---:|
| Product accuracy | 25 | 23 |
| Audience fit and persuasion | 25 | 20 |
| Structure and readability | 20 | 18 |
| Concrete, original language | 15 | 11 |
| Instruction compliance | 15 | 12 |
| **Total** | **100** | **84** |

The article accurately covers the supplied platforms, offline features, Free project/task limits, lack of an account requirement, optional Pro cloud sync and its account/internet requirements, prices and calendar/AI limitations. It includes a headline, sections, concrete freelance situations and the requested download CTA. It invents neither desktop-browser support nor time-tracking/billing functionality, unlike the respective earlier IQ4_XS and IQ3_S articles.

It contains **818 whitespace-separated words including headings**, slightly above the requested 800-word maximum. The same counting method gave 897 for IQ3_S and 986 for IQ4_XS. This is a marginal length miss; a body-only count would be shorter, but the saved scoring method includes headings for every model.

Its weaker points are the opening's blame-heavy “you failed at structure,” generic productivity phrasing, an assumed app prompt for priorities, and the unsupported assurance that no obligation will slip through the cracks. The difference from IQ3_S is modest; the stronger conclusion is that this answer is closer to the supplied product brief than the earlier IQ4_XS answer.

## Technical proposal: 35/100

Read the scored final answer: [qwen27b-python-5.md](qwen27b-python-5.md). Turns 1–4 are retained beside it.

| Category | Maximum | Score |
|---|---:|---:|
| Current code diagnosis | 15 | 12 |
| Python conversion correctness | 25 | 5 |
| Engine integration correctness | 25 | 7 |
| Compatibility, memory and validation | 25 | 6 |
| Clarity and honesty | 10 | 5 |
| **Total** | **100** | **35** |

The final answer correctly recognizes that the current loader and `moe_grouped_s2` expert path are specialized for Q2. It identifies the native layout, recommends `[gate | up | down]` expert storage and native dispatch, retains a Q2 branch, proposes converting large dense projections to Q8_0 and hyper-connection weights to BF16, and calls for checking shapes, types and architecture. It presents proposed validation rather than claiming tests were executed. These earn substantial diagnosis credit.

However, the final proposal still has these concrete problems:

1. **It omits the fused projection entirely.** The supplied `nextn.eh_proj.weight` must be decoded as a NumPy matrix `[2560,5120]` and split on input columns into embedding and hidden projections. Without that step, the loader's required `fc_embedding.weight` and `fc_hidden.weight` are not produced from this specimen.
2. **It reverses expert row axes.** Its packing example says gate/up have 2560 rows of width 640, while they actually have **640 rows of width 2560 per expert**. Down has **2560 rows of width 640**, the reverse of its example. It also sketches dequantization while claiming raw quantized-block preservation without explaining how those paths fit together.
3. **It invents source names and leaves mappings incomplete.** The sidecar uses `blk.48.attn_q.weight` and related attention names, not `blk.48.self_attn.*`. The proposal does not enumerate the required hyper-connection, shared-gate and norm destination names. Referring broadly to `mtp.layers.0.*` does not establish the actual stripped names needed in `dense.txt`.
4. **It misreads the router's runtime type.** It explicitly says the runtime expects F32 router weights. `src/core/mtp.cpp:526` calls `bf16("mlp.gate.weight")`; the supplied F32 router must therefore become BF16, without a norm offset. The one-dimensional F32 shared gate also needs explicit handling. The answer does not explicitly bypass the existing blanket F32 +1 conversion for this already-offset GGUF's norms, and omits the supplied Q5_0 hyper-connection up weights.
5. **It does not solve native activations, scratch or expert stride.** The existing routed-expert activation quantizer produces the legacy format. Native expert execution requires Q8_1, with **36 bytes per 32 values**, native scratch sizing and a per-expert stride based on `NativeExpertLayout.bytes`. These changes are absent. `moe_group_resident` would otherwise continue using the legacy `cpu::BLOB` stride.
6. **Its arithmetic is wrong even for its printed formulas.** It reports gate+up as 2,304,000 bytes and down as 1,382,400 per expert. The correct values are **1,843,200** and **1,740,800**; total **3,584,000 bytes per expert**, **1,835,008,000 bytes / 1750 MiB for 512 experts**. Its total expert estimate is only about 3% high because the component errors partly cancel, but it would allocate and lay out the wrong component boundaries if those figures were followed.
7. **It declares the GPU fit acceptable without a full budget.** Its claim that approximately 14 GB remains for the main model ignores the complete dense/KV/workspace and full-vocabulary draft-head accounting, last-stage GPU placement, startup budget and expert-cache interaction. It also claims Q8_0 dense storage is larger than BF16 in its estimate, contrary to their bytes per element for the same matrix. Tokenizer/target compatibility, precise file bounds and staged import validation remain unspecified.
8. **Some claimed reusable helpers do not support the supplied formats.** `tools/mtp_pack.py`'s `dequant` helper handles that packer's Q2_0/Q4_0/Q8_0 encodings; it cannot decode Q4_K as the proposal suggests. Decoder support must cover the actual GGUF tensor types, including Q4_K, Q6_K and Q5_0.

Review did not reliably improve this response. Turn 2 incorrectly rejected reuse of `native_expert_grouped` and asserted that a new kernel was necessary because main-model execution differs from MTP. Later turns returned to native reuse, but kept an inconsistent executive recommendation to implement a new kernel or adapt the existing one. The final answer retained the important initial conversion errors and omissions despite the explicit fourth-turn request to check axes, activations, scratch and bytes.

Compared with this answer, IQ4_XS explicitly identified Q8_1 activation conversion and native scratch, correctly handled the F32 router as BF16, computed about 1.83 GB for experts and identified `generate.cpp`'s drafter budget. Its own proposal still contains blocking errors, as documented in the original report, but it covers more of the supplied implementation contract. IQ3_S also recognized the native activation requirement, although it gave incorrect block sizes and other details.

## Speed recorded during the requested tasks

| Task | Generated tokens | Generation time | Weighted decode rate | Request wall time |
|---|---:|---:|---:|---:|
| Marketing | 994 | 23.76 s | 41.83 tokens/s | 26.03 s |
| Code, five turns combined | 16,081 | 312.26 s | 51.50 tokens/s | 315.80 s |

The weighted rate is total reported `predicted_n` divided by total reported `predicted_ms`, matching the previous report's method. It includes the endpoint's speculative generation. The active server reports **`none,draft-mtp`**; the earlier Strata targets both used the original Q2 MTP. The current endpoint's drafter quantization and placement were not established, and server settings were not changed.

On these responses, the 27B code decode rate is about **28% higher than the earlier IQ4_XS run**, and about **6% below IQ3_S**. Marketing generation is close to IQ4_XS and slower than IQ3_S. These are observations of the loaded setups and generated texts, not isolated architecture/quantization speed measurements.

**Wall time is affected by cached prompts.** The corrected replay reused 402 cached marketing prompt tokens and 155,805 cached tokens summed across the five code requests, processing only 4 and 399 prompt tokens respectively. This arose from the preceding sampling-correction run. Therefore the 315.80-second total must not be presented as a controlled end-to-end speed win against the older Strata totals. Decode rate is reported separately so that cached prefill does not dominate the comparison. Output lengths, histories, context allocation, engine build and MTP setup also differ.

The endpoint additionally reported 600 accepted of 1182 drafted tokens for marketing (**50.76%**) and 11,773 of 12,918 for code (**91.14%**). These counts belong to its own drafter and these responses; they are not a Q2-versus-Q4 MTP comparison and do not measure answer correctness.

## Saved evidence and limits

[test-inputs.json](test-inputs.json), [scope-clarification.json](scope-clarification.json), [source-snapshot.md](source-snapshot.md), [source-manifest.json](source-manifest.json) and [rubric.json](rubric.json) preserve the exact earlier test inputs. Every corrected request/response is saved as `qwen27b-*-request.json` / `*-response.json`, with Markdown answers beside them. [run-manifest.json](run-manifest.json), [scores.json](scores.json), [comparison-verification.json](comparison-verification.json), [sampling-correction.json](sampling-correction.json), [endpoint-models.json](endpoint-models.json), [endpoint-props.json](endpoint-props.json) and [endpoint-slots.json](endpoint-slots.json) preserve provenance and checks. Use the corrected files at this directory's top level for the reported result; the superseded subdirectory is an audit record.

This test uses English, thinking disabled, one seed and two invented tasks. It compares a different 27B architecture and quantization recipe served by llama.cpp with previously recorded Flash-Next variants served by Strata. It does not measure actual OpenCode tool use, editing, debugging or general agent reliability. The result supports a preference for the saved IQ4_XS proposal in this specific exercise, not a universal coding-model ranking.
