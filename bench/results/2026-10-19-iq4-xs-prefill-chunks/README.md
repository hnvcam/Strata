# IQ4_XS prefill chunk comparison — 2026-10-19

Follow-up: [true 6K and 8K fit checks and a fresh 4K control](larger-chunks.md). The saved setting remains 4K.

On the i5-13500, RTX 5070 Ti 16 GB + RTX 4060 8 GB, 64 GB RAM, the existing Strata 0.1.45 engine successfully ran 1,024, 2,048 and 4,096-token prefill chunks. No engine patch or imported llama.cpp fork was needed. The saved local configuration now uses **4,096-token prefill and a 40 GiB resident-expert RAM budget**.

## Matched measurements

Tokens/s; medians of three timed requests per case, after warmup. Both arms used the same **40 GiB expert-RAM budget**, the same 4,242 initial GPU expert slots (1,159 on the 4060 and 3,083 on the 5070 Ti), and the same 40,958 MiB pinned expert allocation. This compares chunk sizes under that RAM cap; the 512 column is not an unchanged automatic-RAM production run.

| Measurement | Chunk 512 | Chunk 4096 | Observed ratio |
|---|---:|---:|---:|
| Prefill, 4,096 input tokens | 205.8 | 644.6 | 3.13x |
| Prefill, 16,384 input tokens | 199.0 | 947.4 | 4.76x |
| Decode, short coding prompt / 322–358 output tokens | 47.1 | 51.8 | 1.10x |

Raw timed rates:

- 512 / 4K: 207.0, 205.8, 204.2.
- 4096 / 4K: 376.3, 644.6, 650.5.
- 512 / 16K: 197.4, 203.7, 199.0.
- 4096 / 16K: 917.7, 947.5, 947.4.
- 512 / decode, after a 16K prompt warmup: 46.2, 47.1, 50.1.
- 4096 / decode, after the prefill suite: 51.8, 51.9, 50.5.

Median time to the first output token at 4K was 19.94 → 6.39 s; at 16K, 82.36 → 17.32 s. These timings come from the native engine pipe, not HTTP.

The prefill improvement was present in every timed request. The first timed 4K request at 4096 was slower than the following two, even after a 4K warmup. File-cache warming still matters: the separate first 4K warmup after load ran at 169.0 → 233.7 tokens/s. The table is not a promise of fully cold-disk speed.

The decode increase is an **observation, not an isolated chunk-size effect**. The adaptive expert cache remained enabled, output lengths differed, and the warmup histories were not identical. The additional 512 decode control ran last, after a full 16K read, to reduce the warming advantage of the 4096 arm. The measured decode context was the short coding prompt, not a 16K ongoing conversation. No general decode speedup is established by these few runs.

## Conditions and checks

- Native UD-IQ4_XS, installed engine 0.1.45; GPU order `[1,0]`, split at layer 5 (4060 layers 0–4, 5070 Ti layers 5–47 and head/MTP).
- Context capacity 131,072; INT8 KV; `--spec 4`, `--spec-min-p 0.70`, `--pcie-frac 0.20`, reserves `700,1152` MiB. Full Q2 MTP vocabulary retained: all 248,320 draft token IDs.
- Two unrelated apps retained about 3,123 MiB on the 4060 throughout; neither was stopped. Results apply to that available-VRAM condition.
- Identical saved 4K/16K token IDs in both prefill arms. A passphrase was embedded midway through repository excerpts. Every retrieval check passed, with zero prompt tokens reused.
- Prompt checkpoints, parked conversations, and disk chunk caching were disabled for these measurements. Disk cache files from normal use were not deleted or modified by the tests.
- Each longer coding output was parsed and executed after inspection. Its own two assertions and two additional odd/even-length median cases passed in all three final decode outputs per arm.
- Both final arms reported the requested `prefill_cap`. The 4K input uses 4,095 batched tokens at cap 4096 because generation begins from the final input token; the 16K cases reported `chunk=4096`. The streaming ring was 96 slots.
- All isolated engines exited normally on input EOF, exit code 0. The production server was already stopped. No running production engine/server was killed or restarted.
- During every measurement, SHA-256 checks confirmed the saved config, installed engine, expert profile and `draft_vocab.bin` were unchanged. Only after completing measurements was the saved config updated.

## RAM and settings

The initial automatic-RAM sweep expanded the pinned allocation from about 43.5 to 49.3 GiB, and a live check showed 8.5 GiB of system swap in use. That changed the test conditions and consumed memory outside the engine, so those exploratory numbers were not used for the matched claim. With the 40 GiB cap, the matched arms had about 10.3 and 9.3 GiB available at READY. Linux swap occupancy still varied; the cap is not a guarantee of zero swap.

The only saved configuration changes are:

```diff
+ --resident-budget-gib 40
- --prefill 512
+ --prefill 4096
```

Context, GPU placement, VRAM reserves, sampling/speculation settings, expert profile paths and saving, and full MTP vocabulary were preserved. The engine/model files were not modified. No defaults in Strata source were changed.

The previous configuration is [original-config.json](original-config.json), the saved replacement is [candidate-config.json](candidate-config.json), and [config.diff](config.diff) shows the exact difference. [application.json](application.json) records application checks. The saved setting takes effect at the user's next normal start; the server was left stopped.

Normal disk chunk-cache keys include chunk size, so 512-token cache entries do not match the new 4096-token keys. Existing entries are retained subject to the normal cache budget; new-size entries warm through normal use. This behavior was checked in `src/core/chunk_cache_disk.cpp`, not measured through an HTTP server here.

Not tested: a full 131K request, multi-user load, long-context decode rates, a broad model-quality evaluation, other model quantizations, other hardware, or the same PC with the other GPU apps closed. Output was not claimed bit-identical across chunk sizes.

## Raw records and reproduction

[summary.json](summary.json), [512 prefill](matched-512-40g.json), [4096 prefill/decode](matched-4096-40g.json), [warmed 512 decode](decode-warm-512-40g.json). Corresponding `.engine.log` files contain startup, memory, I/O and request details; `.protocol.log` files contain every progress, token and DONE line. Prompt fixtures and [run.py](run.py) are saved here.

The exploratory automatic-RAM sweep is saved as `baseline-512.json`, `candidate-1024.json`, `candidate-2048.json`, and `candidate-4096.json`. The earlier, less-warmed decode-only control is `decode-512-40g.json` (46.4 tokens/s median); it is not the decode baseline used above.

After the user has stopped any production Strata engine, run from the repository root using a Python with the `regex` dependency and GPU access:

```sh
python bench/results/2026-10-19-iq4-xs-prefill-chunks/run.py --chunk 512 --tag repeat-512 --ram-budget 40 --runs 3 --warm-prefill
python bench/results/2026-10-19-iq4-xs-prefill-chunks/run.py --chunk 4096 --tag repeat-4096 --ram-budget 40 --runs 3 --warm-prefill --decode-runs 3 --decode-max 512
python bench/results/2026-10-19-iq4-xs-prefill-chunks/run.py --chunk 512 --tag repeat-decode-512 --ram-budget 40 --runs 1 --sizes 16k --decode-runs 3 --decode-max 512
```

The harness reads the saved config for model paths and unchanged settings, overrides only the measured chunk/RAM options, removes profile saving, and disables all prompt caches. It refuses to launch while another Strata engine is running. It never sends a kill/terminate signal or edits the production config.
