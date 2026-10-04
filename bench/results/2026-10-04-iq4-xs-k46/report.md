# Layers 0–45 on the 5070 Ti; layers 46–47, head and MTP on the 4060

Measured 2026-10-04 on the Intel i5-13500, RTX 5070 Ti 16 GB, RTX 4060 8 GB and 64 GB RAM, Linux. The patched executable matches the SHA-256 used for the saved fresh dense-placement and four-layer tests.

## Confirmed placement

| GPU | Main layers | Cached main-model experts | Expert VRAM | Output head / Q2 MTP |
|---|---|---:|---:|---|
| RTX 5070 Ti | 0–45 | 3560 of 23552 | 7.90 GiB | Neither |
| RTX 4060 | 46–47 | All 1024 of 1024 | 3.00 GiB | Both |

The 4060 loads 84.47 MiB canonical and 145.43 MiB native dense weights for its layer range, including retained shared weights. The Q2 draft layer uses 949 MiB and its 248320-token head uses 497.3 MiB. The 4060 keeps its own 0.41 GiB prompt buffers, so its complete expert cache is not borrowed for prompt processing. The 5070 Ti prompt path borrows 185 of its own slots and refills them. The startup logs confirm the requested ranges, head, full draft vocabulary and complete 4060 expert cache.

## Matched evaluation

IQ4_XS, installed Q2 MTP, all 248320 draft token IDs, 131072 context capacity, int8 KV, 512-token prefill chunks, spec=4, cutoff=.70, 1248 MiB reserve, PCIe fraction=.20. Greedy sampling, seed 1234, thinking disabled, suffix drafting disabled; adaptive caching and automatic MTP policy enabled. The original placement benchmark runner was imported and reused.

One fresh engine, one 128-token warmup, the saved English/Python/Vietnamese prompts twice at up to 512 output tokens each, then the identical cold 9851-token maintenance-log prompt with up to 256 output tokens. The request objects match the saved patched K=4 evaluation exactly. Short decode is total output tokens divided by summed engine decode time, excluding warmup and the long request. Acceptance is total accepted drafts divided by total offered drafts.

## New results by language

| Output | Generated tokens | Decode tok/s | MTP acceptance | Expert cache hit rate |
|---|---:|---:|---:|---:|
| english | 858 | 47.52 | 71.6% | 80.7% |
| code | 1024 | 39.17 | 92.6% | 54.0% |
| vietnamese | 974 | 42.02 | 75.4% | 75.8% |
| Combined | 2856 | 42.39 | 81.0% | 70.3% |

## Saved combinations

Only the NEW row was run for this request. All other rows reuse raw results measured 2026-10-03. Pilot rows have one repeat per language and no long prompt. Tuned rows change reserve and prefill. The initial single-GPU cold-prompt result did not reproduce; the recheck is the useful comparison. The compilation-overlapped row is retained for history; use the fresh patched K=2 row for comparison.

| Placement | Engine | Short tok/s | Cold prompt seconds | Cold prompt tok/s | Long decode tok/s |
|---|---|---:|---:|---:|---:|
| NEW: 5070 0–45; 4060 46–47 + head/MTP, full cache | patched | 42.39 | 41.17 | 239.3 | 35.7 |
| 4060 0–3 full cache; 5070 4–47 + head/MTP | patched | 46.74 | 39.62 | 248.6 | 41.1 |
| 4060 0–1 full cache; 5070 2–47 + head/MTP | patched | 42.60 | 40.82 | 241.3 | 37.3 |
| 4060 0–1; 5070 2–47 + head/MTP, fresh | original | 43.41 | 41.46 | 237.6 | 37.6 |
| 5070 alone, recheck | original | 41.69 | 41.45 | 237.6 | 38.2 |
| 5070 all layers + head/MTP; 4060 expert helper | original | 46.61 | 41.70 | 236.2 | 43.7 |
| 4060 0–1; 5070 2–47 + head/MTP, earlier | original | 47.25 | 36.90 | 267.0 | 40.5 |
| 4060 0–3; 5070 4–47 + head/MTP, pilot | original | 46.46 | — | — | — |
| 5070 0–46; 4060 47 + head/MTP, auto | original | 40.73 | 35.02 | 281.3 | 38.4 |
| 5070 0–43; 4060 44–47 + head/MTP, pilot | original | 37.85 | — | — | — |
| 5070 0–46; 4060 47 + head/MTP, tuned 1152 MiB/256 prefill | original | 42.37 | 58.40 | 168.7 | 38.5 |
| 5070 0–46; 4060 47 + head/MTP, tuned 1024 MiB/256 prefill | original | Failed: verification-graph OOM | — | — | — |
| 5070 alone, initial (cold prompt did not reproduce) | original | 41.60 | 14.61 | 674.0 | 55.8 |
| 4060 0–1 full cache; 5070 2–47 + head/MTP (compilation overlapped) | patched | 43.46 | 40.96 | 240.5 | 38.2 |

## Interpretation and state

The new short decode rate differs by -9.3% from 4060 0–3 full cache; 5070 4–47 + head/MTP (46.74 tok/s).
The new short decode rate differs by -0.5% from 4060 0–1 full cache; 5070 2–47 + head/MTP (42.60 tok/s).
The new short decode rate differs by +1.7% from 5070 alone, recheck (41.69 tok/s).
The new short decode rate differs by -9.1% from 5070 all layers + head/MTP; 4060 expert helper (46.61 tok/s).

This small speed sample does not establish quality equivalence or full 128K-context performance. Expert residency and CPU/GPU rounding can change generated text and MTP acceptance. The long-request automatic draft policy can differ across placements. Smoke/parity checks and the separate Q2-versus-Q4 MTP study are not pooled with these speed results.

Production configuration, original engine, patched engine, expert profile, full draft vocabulary and the reused runner were unchanged by hash. The temporary localhost server stopped; production remains stopped.

```text
index, name, memory.used [MiB]
0, NVIDIA GeForce RTX 5070 Ti, 2 MiB
1, NVIDIA GeForce RTX 4060, 2 MiB
```

Raw requests, responses and timings: [patched-k46-results.json](patched-k46-results.json). Test configuration: [patched-k46-config.json](patched-k46-config.json). Hashes and state: [manifest.json](manifest.json). All language-level comparisons: [summary.json](summary.json).
