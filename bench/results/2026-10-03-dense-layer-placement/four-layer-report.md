# Four complete expert layers on the 4060

Measured 2026-10-03 on the i5-13500, RTX 5070 Ti 16 GB + RTX 4060 8 GB, 64 GB RAM.
Uses the rebuilt engine with dense weights limited to each GPU's assigned layer range.
Physical GPU order: 4060 first, 5070 Ti last. All 248320 Q2 MTP draft vocabulary IDs,
IQ4_XS, 131072 context capacity, int8 KV, 512-token prompt chunks, spec=4, cutoff=.70,
1248 MiB reserve. Suffix drafting is disabled in both compared runs; adaptive expert swaps remain enabled.

## Confirmed placement

| GPU | Main layers | Cached experts | Expert VRAM | Output head / MTP |
|---|---|---:|---:|---|
| RTX 4060 | 0–3 (first four) | All 2048: 512 per layer | 4.99 GiB | Neither |
| RTX 5070 Ti | 4–47 (remaining 44) | 2939 of 22528 | 6.58 GiB | Both |

The first four layers' experts require exactly 5,360,844,800 bytes.
The engine confirmed all 2048 slots were filled. The 4060 had 1556 MiB free after startup.
Generation proceeds through the 4060's first four layers and then the 5070 Ti finishes with
its remaining layers, output head and MTP. No expert-only helper was enabled.
Prompt processing can temporarily borrow cache slots for buffers; they are refilled afterward.

## Speed

The four-layer test loaded a fresh engine, warmed up for 128 generated tokens, then ran the same
English/Python/Vietnamese prompts twice with up to 512 output tokens each, followed by a separate
9851-token cold prompt with up to 256 output tokens. The two-layer row reuses the immediately preceding
saved test with the same final executable and settings. Short generation is weighted by total output
tokens divided by summed engine decode time; warmup and the long request are excluded.

| 4060 placement | 5070 Ti placement | Short generation | Cold prompt read | Long-request generation |
|---|---|---:|---:|---:|
| Layers 0–1, all 1024 experts cached | Layers 2–47 + output head/MTP, 2771 experts cached | 42.60 tok/s | 40.82 s | 37.3 tok/s |
| Layers 0–3, all 2048 experts cached | Layers 4–47 + output head/MTP, 2939 experts cached | 46.74 tok/s | 39.62 s | 41.1 tok/s |

The measured short-generation change is +9.7%. This is a small local sample;
changing the layer placement and expert residency can change CPU/GPU rounding, text and MTP acceptance.
It does not establish quality equivalence or performance at a full 128K prompt.
The old four-layer pilot before the dense-weight change cached only 631 experts; it is a different setup
and should not be used as the result of this complete-cache test.

## Saved configuration and state

`changed-full-cache-k4-config.json` is the tested configuration, using `build/strata`, `gpu: [1, 0]`
and `layer_split: 4`. It is an experiment configuration, not the active production configuration.
Production config, original engine, rebuilt engine and the full Q2 draft vocabulary were unchanged by hash.
All temporary servers stopped. Final state:

```text
index, name, memory.used [MiB]
0, NVIDIA GeForce RTX 5070 Ti, 2 MiB
1, NVIDIA GeForce RTX 4060, 2 MiB
```

Raw prompts, responses, timings and logs are saved in `changed-full-cache-k4-results.json` and its
adjacent engine/server logs. `four-layer-manifest.json` records the engine hashes and model expert sizes.
