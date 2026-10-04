# IQ3_S versus IQ4_XS: two original tasks

Date: 2026-10-03. IQ3_S wrote the more accurate marketing article. IQ4_XS produced the stronger technical proposal, but **neither proposal is ready to implement**. These are judgments about the saved responses to these two tasks, not a general ranking of the models.

| Task | IQ3_S | IQ4_XS | Judgment |
|---|---:|---:|---|
| Marketing article | 80/100 | 74/100 | IQ3_S followed the product facts more closely |
| Q4 MTP proposal, final turn | 45/100 | 58/100 | IQ4_XS handled formats and memory better; both contain blocking errors |

Scores use the categories and weights saved in [rubric.json](rubric.json). They are subjective reviewer scores; the concrete errors below matter more than the exact point differences. No aggregate score is used because the tasks measure different abilities.

## What was tested

The marketing task asked for a 600–800-word launch article for **LoopDay**, a fictional offline-capable task-planning app for freelancers. The complete brief specified its features, iOS/Android availability, Free/Pro restrictions, pricing and limitations. Neither model received outside product information.

The technical task supplied actual Python files and relevant engine excerpts from this project, together with an explicit external Q4_K_M MTP specimen. Each model was asked to propose and review an import/runtime adaptation while retaining legacy Q2 support. Models received a fixed source snapshot rather than live filesystem tools. This measures reasoning from that snapshot, including Python conversion and necessary engine changes.

Each model completed one article and **five technical turns**: initial proposal, self-review, first final proposal, scope clarification/review, corrected final proposal. Both initially recommended converting the experts back to Q2. The initial wording left that option open, so I explicitly clarified to both that Q4_K gate/up and Q8_0 down must retain their supplied quantization. Both received the same two additional prompts. That clarification was an evaluation intervention, and is preserved in [scope-clarification.json](scope-clarification.json); initial requantization preference is not treated as a standalone failure against the later clarified requirement.

There was no model-specific correction or coaching. Both had the same system and user messages at corresponding turns. Later histories necessarily included each model's own prior answers, so entire multi-turn request bodies differ in their assistant messages. All 12 responses ended with `finish_reason=stop`; none was truncated.

**No project `bench/` runner or public benchmark suite was used.** The saved runner sends only these invented tasks to temporary local Strata servers.

## Shared settings and verification

Hardware: Intel i5-13500, 64 GB system RAM, RTX 5070 Ti 16 GB and RTX 4060 8 GB. Models ran sequentially.

| Setting | Both models |
|---|---|
| Engine | Original `engine/strata`, SHA-256 `c6dfda3202fe723de920fd2510e2c386db1e26ea34d3bb22a62f88dbe19cc7ac` |
| GPU order / layer split | `[1,0]`: 4060 first, 5070 Ti last; fixed split 2 |
| MTP | Original Q2 runtime, on last GPU; full 248,320-token draft vocabulary |
| Context / prefill / KV | 131,072 / 512 / int8 |
| Speculation | `spec=4`, `spec-min-p=0.70`, `suffix-draft=0` |
| Cache / reserve / PCIe fraction | auto / 1248 MiB / 0.20 |
| Sampling | temperature 0.2, top_p 0.95, top_k 20, seed 1234 |
| Thinking | `reasoning_effort=none`, `enable_thinking=false` |
| Output limits | Article 2048 tokens; technical turns 6144, 4096, 6144, 4096, 6144 |

Runtime configurations matched except model asset paths, model/tokenizer names and logs. Model metadata checks found the same architecture geometry and tokenizer tokens/merges. The actual files are **GSQ-RCO IQ3_S** and **Unsloth UD-IQ4_XS**: this comparison includes their quantization recipes and is not an isolated change in bit width.

[comparison-verification.json](comparison-verification.json) records matching external messages/settings at every turn, engine/config checks and final service state. Production source, configs, engine and full draft-vocabulary hashes remained unchanged. The temporary server was stopped, and production was left unloaded. Only evaluation artifacts were created for this task.

## Marketing evaluation

| Category | Maximum | IQ3_S | IQ4_XS |
|---|---:|---:|---:|
| Product accuracy | 25 | 20 | 16 |
| Audience fit and persuasion | 25 | 22 | 22 |
| Structure and readability | 20 | 17 | 17 |
| Concrete, original language | 15 | 11 | 12 |
| Instruction compliance | 15 | 10 | 7 |
| **Total** | **100** | **80** | **74** |

Both articles have a relatable late-evening opening, sections, accurate tier prices, an honest calendar/AI limitation and the requested download CTA. Both drift toward a feature walkthrough and use familiar productivity slogans. IQ4_XS provides somewhat more concrete client examples, including logo drafts and a login bug, without a decisive persuasive advantage.

Both missed the word limit: **IQ3_S 897 words; IQ4_XS 986 words**, counted by whitespace over the saved article, including headings. Both remain over 800 even excluding headings. They were not cut off by token limits.

IQ3_S says the weekly review “helps you track your time and effort ... useful for billing,” which suggests capabilities beyond a list of completed tasks. It also assumes an automatic weekly review prompt and calls the app private without supporting product facts. These require editing, but it does not invent an additional supported platform.

IQ4_XS explicitly offers access “from a desktop browser.” The brief provides only iOS and Android, so this is an invented product feature. It also assumes mandatory immediate client tagging and makes unsupported privacy claims. Its longer article and concrete platform invention account for the lower score.

Read the original articles: [IQ3_S](iq3_s-marketing-1.md), [IQ4_XS](iq4_xs-marketing-1.md).

## Technical proposal evaluation

| Category | Maximum | IQ3_S | IQ4_XS |
|---|---:|---:|---:|
| Current code diagnosis | 15 | 14 | 14 |
| Python conversion correctness | 25 | 7 | 10 |
| Engine integration correctness | 25 | 12 | 16 |
| Compatibility, memory and validation | 25 | 6 | 11 |
| Clarity and honesty | 10 | 6 | 7 |
| **Total** | **100** | **45** | **58** |

Both final proposals correctly identify that the existing MTP loader and `moe_grouped_s2` path are specialized for Q2. Both propose retaining Q4_K/Q8_0 expert bytes, assembling `[gate | up | down]` per expert, using `NativeExpertLayout` and `native_expert_grouped`, retaining the legacy branch, converting large dense projections to Q8_0 and avoiding another +1 on the already-offset norms. Both recognize the need for Q8_1 activations and native scratch. Neither claims to have implemented or executed its proposed tests.

**Blocking issues common to both:**

- **Fused projection axes:** the supplied GGUF dimensions `[5120,2560]` are innermost-first. The decoded NumPy matrix is `[2560,5120]`, and each output row contains embedding inputs followed by hidden inputs. The two projections must be split on input columns. IQ3_S explicitly splits rows. IQ4_XS's final answer continues to describe an uncertain “row order” without identifying the column split, despite the input supplying that order. Its preceding review explicitly reversed the axes.
- **Tensor mappings:** the source attention name is `blk.48.attn_q.weight`. IQ3_S instead assumes source `blk.48.self_attn.*`; IQ4_XS writes `blk.48.attn_q_proj.weight`. Both fail to provide a complete correct hyper-connection mapping to the loader's required names, such as `attn_hyper_connection.input_mix_weight_down.weight`. These plans would encounter missing tensors or missing runtime weights.
- **Expert address stride:** changing allocation size and dispatch is insufficient. `moe_group_resident` at `src/core/mtp.cpp:530` still uses the fixed legacy `cpu::BLOB` stride. Neither final proposal explicitly changes that stride to the native layout's bytes. Subsequent expert addresses would therefore be incorrect.
- **Compatibility and budgeting:** neither final proposal fully specifies source/target tokenizer and geometry checks, tensor bounds and exact byte validation, safe staged imports, and the complete last-GPU memory budget including the full draft head, KV/work buffers and main-model expert cache. “Safe” or “comfortable” on the 5070 Ti is not established by their estimates.

**IQ3_S-specific problems:** its final proposal says Q8_1 uses 34 bytes per 32 values and Q8_0 uses 18. The supplied reader gives **36 and 34**, respectively; 18 bytes belongs to Q4_0. Its illustrative `quantize_q8_1_rows` call also places the output pointer in the wrong argument position relative to the supplied declaration. It omits the router/shared-gate conversion details and the supplied Q5_0 hyper-connection tensor. It gives **2.34 GB** for experts even though its own later size-check formula yields **1.835 GB**, and omits the startup cache-budget change from its affected files. These errors remain after review.

**IQ4_XS advantages and remaining problems:** it correctly computes approximately **1.83 GB** of native expert storage, explicitly converts the router to BF16, recognizes the startup drafter budget in `generate.cpp`, and names the native scratch sizing helper. Its preceding review also includes Q5_0 in hyper-connection decoding. However, its suggested fixed **2200 MiB** budget is insufficiently justified: it omits the full head and KV/workspace accounting, and fails to resolve GPU placement from the last-stage rule. Its final mapping list also omits the one-dimensional shared gate. Leaving known tensor axes as uncertainty does not solve the supplied specimen.

The exact routed-expert arithmetic, from the supplied block geometry, is:

| Component | Bytes per expert | Bytes for 512 experts |
|---|---:|---:|
| Gate Q4_K | 921,600 | 471,859,200 |
| Up Q4_K | 921,600 | 471,859,200 |
| Down Q8_0 | 1,740,800 | 891,289,600 |
| **Total** | **3,584,000** | **1,835,008,000 = 1750 MiB** |

Legacy Q2 experts occupy 675 MiB, so the expert-only increase is **1075 MiB**. The unchanged Q2 runtime in this test logged 949 MiB for the draft layer plus a separate **497.3 MiB full draft head** on IQ4_XS. Dense weights, KV and buffers must be counted separately; total Q4 residency cannot be inferred from expert bytes alone. These are reference facts used to review the proposals, not measurements of an implemented Q4 runtime in this exercise.

Read the scored final proposals: [IQ3_S](iq3_s-python-5.md), [IQ4_XS](iq4_xs-python-5.md). All previous proposal/review turns are retained as `*-python-1.md` through `*-python-4.md`.

## Timing observed during these tasks

These figures come from the requests already needed to obtain the answers. No separate speed test was run. Decode rate is total generated tokens divided by total reported generation time; acceptance is accepted draft tokens divided by drafted tokens. **Both targets used the same Q2 MTP**, so these are not Q2 versus Q4 MTP acceptance results.

| Task | IQ3_S decode tokens/s | IQ4_XS decode tokens/s | IQ3_S draft acceptance | IQ4_XS draft acceptance |
|---|---:|---:|---:|---:|
| Marketing | 57.80 | 42.24 | 76.12% (545/716) | 72.07% (596/827) |
| Technical, five turns combined | 54.88 | 40.23 | 87.94% (9386/10673) | 87.25% (10347/11859) |

Request wall time was 20.97 versus 31.75 seconds for marketing, and 497.74 versus 717.54 seconds summed over the five technical turns. These totals exclude model startup and reviewer work. They include prefill; stages 1 and 4 started with cold histories. Output lengths and subsequent assistant histories differed, so wall-time differences do not represent a controlled equal-output speed comparison.

## Reuse and limits

Keep this directory to review the evidence without repeating inference. [test-inputs.json](test-inputs.json), [scope-clarification.json](scope-clarification.json), [source-snapshot.md](source-snapshot.md), [source-manifest.json](source-manifest.json) and [rubric.json](rubric.json) preserve the test conditions. Request/response JSON files preserve exact API inputs, outputs and timing. [scores.json](scores.json), [run-manifest.json](run-manifest.json), [followup-manifest.json](followup-manifest.json) and [comparison-verification.json](comparison-verification.json) preserve scoring and checks.

This was one invented article and one five-turn code task per model, in English, with thinking disabled and one fixed seed. It establishes the errors and strengths of these saved answers. It does not establish statistical significance, general coding capability, behavior in other languages, or that one quantization universally wins. The technical proposals should receive a human correction before implementation; no proposal was applied.

Installation status after testing: the user requested removal of IQ3_S and chose 27B for writing and IQ4_XS for coding. Its GGUF directory, pack and local config/launcher/runtime log were removed; these saved responses, inputs and historical test configs remain as evidence. IQ3_S paths in this report are no longer installed assets. See [cleanup record](../2026-10-03-iq3-s-cleanup.json).
