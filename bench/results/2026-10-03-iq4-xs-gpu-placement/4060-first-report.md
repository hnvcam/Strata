# Final test: RTX 4060 first, RTX 5070 Ti finishes with the head and MTP

Measured 2026-10-03 on the Intel i5-13500, 64 GB RAM, RTX 4060 8 GB (PCIe Gen 3 x4) and RTX 5070 Ti 16 GB (Gen 4 x16), Linux, original engine 0.1.34.

## Requested flow

The temporary config sets `gpu: [1, 0]` and `layer_split: auto`. CUDA0 is therefore the physical RTX 4060; CUDA1 is the physical RTX 5070 Ti. The automatic split selected K=2:

- RTX 4060: first two layers (engine layer numbers 0–1).
- RTX 5070 Ti: remaining 46 layers (2–47), output head and Q2 MTP drafter.

The engine log confirms one stage handoff per verification window from the 4060 to the 5070 Ti. There is no expert-only helper enabled, and no per-layer expert-result round trip from the 5070 Ti to the 4060 and back. Each subsequent generation window starts again on the first stage; this is not a one-time handoff for the entire conversation.

## What fits

With the current dense-weight copies, 128K context capacity and 1248 MiB reserve, the 4060 caches 718 experts (1.56 GiB) for its two layers. Those layers have 1024 experts in total; all their expert weights would take 2.22 GiB. Thus the selected layer range fits with an expert cache and RAM fallback, not with every expert permanently resident. The 5070 Ti caches 2690 experts (6.06 GiB) for its remaining layers.

The existing engine requires the first stage to contain at least two layers. Its automatic placement considers every supported two-card split from K=2 through K=47 using estimated compute and cache-miss costs; it selected the smallest supported early range for this pair. A four-layer pilot checks whether that larger early range helps in practice. It caches only 631 of the first four layers' 2048 experts (1.55 GiB); the larger third-layer expert blobs use more bytes per cached expert.

## Matched settings and method

IQ4_XS, installed Q2 MTP runtime, all 248320 draft IDs, maximum context 131072, int8 KV, spec=4, spec-min-p=.70, prefill=512, VRAM reserve=1248 MiB and PCIe fraction=.20. Original engine and source unchanged. Suffix drafting disabled in every compared test, thinking disabled, greedy sampling. Cache adaptation and automatic MTP policy remain enabled. No reserve reduction or smaller prefill buffers are used.

Each full test loads a fresh engine, warms up for 128 output tokens, then runs the same English/Python/Vietnamese prompts twice with up to 512 output tokens, followed by one cold 9851-token maintenance-log prompt and 256 output tokens. Decode speed is total generated tokens divided by summed engine decode time; warmup and the long request are excluded. The four-layer pilot has one repeat per language and no long request. The single-GPU and helper rows reuse the previously saved matched results; they were not rerun in this final test.

## Results

| Setup | Short decode tok/s | Cold prompt seconds | Cold prompt tok/s | Long-request decode tok/s |
|---|---:|---:|---:|---:|
| 5070 Ti alone, recheck | 41.69 | 41.45 | 237.6 | 38.2 |
| 5070 Ti + 4060 expert-only helper | 46.61 | 41.70 | 236.2 | 43.7 |
| 4060 first two layers, 5070 Ti finishes | 47.25 | 36.90 | 267.0 | 40.5 |
| 4060 first four layers, pilot | 46.46 | — | — | — |

| Setup | English tok/s | Code tok/s | Vietnamese tok/s | First repeat combined tok/s |
|---|---:|---:|---:|---:|
| 5070 Ti alone, recheck | 45.18 | 37.99 | 43.17 | 43.00 |
| 5070 Ti + 4060 expert-only helper | 51.34 | 42.13 | 48.24 | 47.84 |
| 4060 first two layers, 5070 Ti finishes | 53.96 | 42.02 | 48.43 | 49.21 |
| 4060 first four layers, pilot | 49.51 | 42.61 | 48.73 | 46.46 |

## Interpretation and limits

The requested first-two-layer split averaged 47.25 tok/s: 13.3% above the rechecked single GPU and 1.4% above the expert-only helper. The small difference against the helper is not enough to establish a reliable overall winner. The helper was faster on the separate long request's decode phase; the early split read that prompt faster in this measurement.

In the matched first repeat, four early layers averaged 46.46 tok/s versus 49.21 for two early layers. The larger early range did not improve that pilot.

The initial single-GPU cold-prompt result elsewhere in this directory (14.61 seconds) did not reproduce in its recheck (41.45 seconds). Cold-prompt timing is therefore uncertain; one long request per setup does not establish steady prompt throughput. This is a small speed test, not a quality evaluation or a benchmark at a full 128K prompt. Expert residency and CPU/GPU rounding can change generated text and MTP acceptance between placements.

For the requested flow, the tested auto split is a reasonable choice: first two layers on the 4060, all remaining layers and the head/MTP on the 5070 Ti. Avoid assuming that more early layers or eliminating the helper's round trips guarantees higher throughput.

## Saved state

The production config, original engine and installed full draft vocabulary are unchanged. Temporary servers were stopped and both GPUs released. The tested config is `4060-first-5070-head-mtp-config.json`; raw responses, logs, `4060-first-summary.json`, the runner and `4060-first-manifest.json` are saved here.
