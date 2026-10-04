# Final suffix layers on the 4060; head/MTP on the 5070 Ti

The requested layout works with **n=44**. Layers 0–43 plus the output head/Q2 MTP run on the 5070 Ti; layers 44–47 run on the 4060 with all 2048 experts cached for decode. Five final layers do not fit fully under the tested settings. This run had faster initial cold prefill, but slower decode than either saved baseline; it does not establish this layout as the best overall.

Measured 2026-10-04 on the i5-13500, RTX 5070 Ti 16 GB + RTX 4060 8 GB and 64 GB RAM, Linux. This evaluation uses the new build/strata, SHA-256 b671507f9b5755983c6a75aa50484fca2d31c9cfd1709b52b1e91ff21238ef62. Previous helper and first-four-layer results use the saved earlier dense-placement build, SHA-256 20e4e2d88234dc02c8d2e4e5a0ced9fa352b1cc60cd9ec7f4177f66b8caa0903. The ordinary two-stage split passed a matched-residency output regression against that build. These are independent runs with different generated answers and histories.

## Placement and engine support

| Setup | RTX 5070 Ti | RTX 4060 |
|---|---|---|
| Saved helper | All 48 layers + head/MTP; 2513 cached experts, 5.66 GiB | Computes 2800 additional experts, 6.31 GiB |
| Saved first four | Layers 4–47 + head/MTP; 2939 cached experts, 6.58 GiB | Layers 0–3, all 2048 experts, 4.99 GiB |
| Requested final four | Layers 0–43 + head/MTP; 2448 cached experts, 5.44 GiB | Layers 44–47, all 2048 experts, 5.22 GiB |

With an explicit split, `--head-device 0` appends a head-only stage on CUDA0 after the last transformer layer. The 4060 transfers the completed residual back for the output head and draft layer. Verifier and prefill now support this empty [48,48) range, and its pending buffers/full draft head are reserved before sizing the first GPU cache. The head-only stage owns no transformer expert cache. There are two hand-offs per verification window. Automatic split search currently does not search this placement.

Use `"gpu": [0, 1]`, `"layer_split": 44` and append `"--head-device", "0"` to the existing args. The tested configuration is [saved here](head-primary-config.json); it references the experimental build, not the production executable. Capacity verification held only 2116/2560 experts for layers 43–47, versus all 2048/2048 for layers 44–47. This boundary depends on model, context and reserve settings. Prompt buffers temporarily borrow cache slots during prefill; the engine refills those slots before decode.

Three fixed-residency English/Python/Vietnamese requests produced identical choices before/after the patch and with the head moved to CUDA0. A separate final-build continuation/branch test also produced identical responses and reused prompt state. Scheduling validation passed 14 cases, dense ownership/accounting passed, and diff whitespace checks passed. [Patch and GPU correctness/capacity evidence](../../bench/results/2026-10-04-head-on-primary/report.md).

## Exact saved evaluation

The six saved input/rubric/source files match the earlier evaluation by hash. All eight requests match both baseline system/user messages, sampling settings and token limits. Each conversation retains its own assistant responses. The marketing task is the same LoopDay 600–800-word article. The Python task is the same five-stage Q4_K_M MTP conversion/runtime proposal and review; it is not an executable Python test suite. Generated proposals were not implemented.

The two saved stage-2 continuation requests were replayed to preserve the same external instruction sequence, even though this run's review finished within its limit. Their budgets remain 2048 and 6144. Four fresh-server phases match the earlier comparison: initial marketing + Python 1/2; first continuation; second continuation + Python 3; final follow-ups 4/5. All eight answers finished normally. The superseded test placing layer 47 back on the 5070 Ti was cancelled at the user's request and is excluded.

IQ4_XS, installed Q2 MTP, all 248320 draft token IDs, context capacity 131072, int8 KV, prefill 512, spec 4, cutoff .70, reserve 1248 MiB, PCIe fraction .20, suffix drafting disabled; temperature .2, top_p .95, top_k 20, seed 1234, thinking disabled. Existing adaptive-cache/draft defaults remain enabled. No additional warmup or throughput prompts were inserted into this task run. [Input/settings verification](comparison-verification.json), [full manifest](run-manifest.json).

## Measured performance

| Setup / task | Output tokens | Decode tok/s | Prefill tok/s | Prefill s | Request wall s |
|---|---:|---:|---:|---:|---:|
| helper / marketing | 951 | 41.36 | 114.00 | 3.56 | 26.74 |
| helper / python | 20087 | 43.81 | 216.13 | 612.23 | 1075.29 |
| four-layers / marketing | 1190 | 42.52 | 129.06 | 3.15 | 31.37 |
| four-layers / python | 16767 | 42.84 | 255.36 | 505.64 | 901.32 |
| head-primary / marketing | 1123 | 38.66 | 138.39 | 2.93 | 32.20 |
| head-primary / python | 15740 | 39.30 | 235.83 | 533.76 | 938.62 |

The identical initial cold Python prompt has 24796 input tokens:

| Setup | Prefill tok/s | Prefill s | Decode tok/s | Request wall s |
|---|---:|---:|---:|---:|
| Saved helper | 190.8 | 129.96 | 41.2 | 220.87 |
| Saved first four | 215.3 | 115.17 | 40.2 | 202.52 |
| Requested final four | 248.0 | 100.00 | 40.8 | 187.69 |

Across the seven Python requests, the new layout decoded at 39.30 tok/s: 10.3% below the saved helper and 8.3% below the saved first-four split. Python request wall time was 938.62 seconds (15m 39s), versus 1075.29 and 901.32 seconds, while producing 15740 tokens versus 20087 and 16767. Different lengths and histories prevent treating those wall times as an equal-work latency ranking. The first-four split completed its sampled sequence sooner despite producing more tokens.

Marketing plus Python request wall time totaled 970.81 seconds (16m 11s). The four phases including startup/shutdown totaled 1181.80 seconds (19m 42s), with 193.86 seconds of model loading. The longest request had 38752 input tokens, including 38665 reused tokens. This is a long conversation/decode evaluation, not a full 128K throughput test or a statistical significance claim.

Rates divide summed token counts by summed engine time; prefill counts only read tokens, with reused tokens reported separately. New MTP acceptance was 70.63% for marketing and 87.16% for Python. The printed expert-pool hit rates are not per-layer combined-GPU coverage: remote-helper work and device-planned fully resident layers can bypass those counters. Full residency of the four 4060 layers is confirmed by startup allocations, not a new per-layer telemetry trace. Per-token latency consistency was not measured. [Every request, reused tokens and startup time](timing.md), [raw counters](summary.json).

## Short 96-token validation and per-request MTP counters

The preliminary full-profile capacity validation also ran three short greedy requests with a 96-token output cap. Adaptation was disabled. These were correctness/capacity smoke tests, separate from the sampled marketing/Python evaluation. The first request included initial graph capture. Python and Vietnamese outputs reached the cap, so their generated code/text was not scored as a complete deliverable.

| Short request | Input tokens | Output tokens | Prefill tok/s | Decode tok/s | Wall s | MTP acceptance |
|---|---:|---:|---:|---:|---:|---:|
| English refrigerator explanation | 22 | 96 | 30.0 | 27.5 | 4.26 | 75.81% |
| Python merge function | 22 | 96 | 41.3 | 35.5 | 3.26 | 94.67% |
| Vietnamese RAM/SSD explanation | 25 | 96 | 39.4 | 33.1 | 3.59 | 83.33% |

[Short-run raw results](../../bench/results/2026-10-04-head-on-primary/head-full-k44-results.json).

MTP acceptance is accepted draft tokens divided by offered draft tokens. It is distinct from the expert-cache hit rate. All main task outputs finished normally.

| Main request | Accepted / offered drafts | MTP acceptance | Logged expert-pool hit rate |
|---|---:|---:|---:|
| marketing 1 | 534 / 756 | 70.63% | 64.3% |
| python 1 | 2173 / 2461 | 88.30% | 59.8% |
| python 2 | 1957 / 2261 | 86.55% | 57.9% |
| python 2-continuation-1 | 770 / 943 | 81.65% | 60.8% |
| python 2-continuation-2 | 923 / 1038 | 88.92% | 56.5% |
| python 3 | 1022 / 1208 | 84.60% | 58.0% |
| python 4 | 1553 / 1752 | 88.64% | 58.0% |
| python 5 | 1477 / 1667 | 88.60% | 56.9% |

The expert-pool counters omit paths that bypass the CPU pool. They cannot be interpreted as combined-GPU coverage or per-layer hit rates; the fully resident 4060 layers can use the device planner. Aggregate MTP rates use summed accepted/offered counts, rather than averaging request percentages.

## Quality under the saved rubric

| Task | Saved helper | Saved first four | Requested final four |
|---|---:|---:|---:|
| Marketing | 88/100; 771 words | 75/100; 974 words | 72/100; 908 words |
| Final Python proposal | 51/100 | 56/100 | 43/100 |

Scores are subjective judgments of one sampled task per placement. They do not establish hardware-dependent quality rankings.

The new marketing article has useful designer/client examples and correct prices, platforms and main limits. It exceeds the word limit by 108 words, omits the account/internet requirements for sync, and adds unsupported enforced evening planning, prompted weekly review and product-design rationale. [Article](head-primary-marketing-1.md), [category scores/evidence](marketing-review.json).

The corrected technical proposal preserves Q4_K/Q8_0 expert bytes, retains a Q2 branch and names the native dispatch/activation helpers. It is not ready for implementation:

- It states 0.57 GB for experts and 0.9–1.1 GB total MTP, then claims a 4060 fit. Experts alone require 1835008000 bytes (1750 MiB), plus the separate head, KV/workspace, main-model allocations and reserve.
- It changes allocation and dispatch without updating the `cpu::BLOB` grouping stride, native scratch allocation or Q8_1 input-buffer sizing. The native expert stride is 3584000 bytes; Q8_1 blocks use 36 bytes per 32 values.
- It omits the fused projection split: decoded eh_proj has 2560 output rows × 5120 input columns, requiring a per-row split into embedding/hidden input columns.
- It leaves actual tensor-name mapping, router/shared-gate formats and HC conversion unresolved, while defaulting to raw formats that current kernels cannot consume.
- It proposes removing an unsupported C++ norm-offset addition and lacks complete geometry/tokenizer/offset validation, import staging and per-stage memory accounting.

The precision follow-up corrected the earlier preference for lossy Q2 conversion, but these blocking details remained. [Scored final proposal](head-primary-python-5.md), [detailed findings](python-review.json), [unchanged reference facts](review-facts.json), [all scores](scores.json). Both saved baseline proposals also had blocking omissions; their existing reviews were reused.

## State

The production executable/configuration, expert profile and custom full draft vocabulary were unchanged by hash. The benchmark executable and watched source files also remained unchanged during the task run. The temporary server stopped and port 1112 is closed. The source patch and experimental build remain available; the production engine was not replaced.
