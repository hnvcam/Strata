# Dense weights limited to each GPU's layers

Measured 2026-10-03 on an Intel i5-13500, RTX 5070 Ti 16 GB, RTX 4060 8 GB, 64 GB RAM.
The 4060 runs layers 0–1; the 5070 Ti runs layers 2–47, output head and Q2 MTP.
IQ4_XS, 131072 context capacity, int8 KV, 512-token prompt chunks, spec=4, cutoff=.70,
1248 MiB reserve, all 248320 draft vocabulary IDs. Suffix drafts disabled in both comparison arms.

## Change

The canonical and native dense loaders omit per-layer weights outside the GPU's range.
The split is chosen from metadata before any dense payload is uploaded; automatic placement prices
canonical allocations, native projections and session state for each candidate range. Shared/global
weights remain resident, including the `blk.1.ple_*` input module. Output head and MTP placement is unchanged.
The two-layer minimum remains in place.

## Memory and startup residency

| GPU/use | Original | Changed |
|---|---:|---:|
| 4060 dense weights | about 4.29 GiB | 235.36 MiB (0.23 GiB) |
| 4060 cached experts in layers 0–1 | 718 / 1024 (1.56 GiB) | 1024 / 1024 (2.22 GiB) |
| 5070 Ti cached experts in layers 2–47 | 2690 (6.06 GiB) | 2771 (6.25 GiB) |
| 4060 free after startup | 986 MiB | 4162 MiB |

The 4060 now has all 512 experts for each of its two layers cached during decode.
Prompt processing may temporarily borrow cache slots for buffers; the engine refills them afterward.
The 5070 Ti's cache still covers only part of its layers' expert sets.

## Correctness and coverage

- CUDA build for sm_89 and sm_120 succeeded. Existing compiler warnings remain.
- `build/dense_placement_test` passed: disjoint two/three-stage ownership, PLE/global retention,
  multi-digit layer names, overflowing names, index alignment, compacted-arena accounting,
  canonical override exclusion, malformed-index refusal.
- Original and changed engines used the same 384-entry profile (8 experts per layer), identical
  physical GPU placement and disabled adaptation. All three greedy English/Python/Vietnamese
  responses matched exactly: [True, True, True].
- Automatic 4060-first placement still chose K=2 and generated three responses successfully.
- Single 5070 Ti startup and three responses succeeded.
- The opposite GPU order, with output head + MTP on the 4060, also passed startup and three requests;
  see `changed-default-order-results.json` for its selected split and residency.

## Fresh speed comparison

Compilation had finished before these two sequential runs. Each loaded a fresh engine, warmed up for
128 generated tokens, then ran the same English/Python/Vietnamese prompts twice at up to 512 output tokens.
The weighted decode rate below is total output tokens divided by summed engine decode time, excluding
warmup and the separate 9851-token cold prompt. Adaptive expert swaps remained enabled in these arms.

| Engine | Short generation | Cold prompt read | Long-request generation |
|---|---:|---:|---:|
| Original | 43.41 tok/s | 41.46 s | 37.6 tok/s |
| Changed | 42.60 tok/s | 40.82 s | 37.3 tok/s |

The measured short-generation change is -1.9%. This is a small local sample,
not a guarantee for other prompts, longer contexts or GPU combinations. The larger cache changes
CPU/GPU expert residency and rounding, so generated text and MTP acceptance can differ in the full-cache arms.
The earlier changed run averaged 43.46 tok/s; compilation overlapped part of that run,
so the fresh runs above are the comparison used here. This is not a Q2-versus-Q4 MTP test.

## Risks and deployment state

The principal correctness risk is omitting a weight still needed by a shared path. Retained PLE/global
weights, identical-residency parity and the hardware checks cover the exercised paths; they do not
establish correctness for every model variant, AMD hardware, vision/control-vector use or a full 128K prompt.
No change to quantization or kernel arithmetic was made. Automatic placement estimates still approximate
MTP/session overhead, so unusual settings require validation.

The changed executable is `build/strata`. Production `engine/strata`, `strata-iq4_xs.json` and the
full Q2 draft vocabulary were unchanged by hash. All temporary servers stopped; final GPU state:

```text
index, name, memory.used [MiB]
0, NVIDIA GeForce RTX 5070 Ti, 2 MiB
1, NVIDIA GeForce RTX 4060, 2 MiB
```

Raw request/response records, startup logs, configs and engine hashes are saved beside this report.
The initial validation manifest refers to the first build; `fresh-manifest.json` records the final build
with the no-fitting-placement guard and final startup wording. Source changes and the new executable
are retained for review; no production runtime configuration was applied.
