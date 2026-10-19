# Follow-up: true 6K and 8K prefill chunks, 2026-10-19

**Keep the saved 4,096-token setting on the current setup.** True 6,144-token chunks fit only after reducing the transfer ring, and their measured prefill speed was lower than the fresh 4K control. An 8,192-token request fell back to 4,096 even with the minimum transfer ring. No production setting was changed during this follow-up.

## Fresh comparison

These are **new follow-up measurements**, not the earlier 512-vs-4096 session's numbers. Each timed case has three requests after warmup, with the same 40 GiB expert-RAM cap. The 4K control was rerun after the 6K tests to check current conditions. Tokens/s:

| Actual maximum chunk | Transfer-ring slots | Prefill, 4K input | Prefill, 16K input | Decode |
|---|---:|---:|---:|---:|
| 4096, fresh control | 96 | 501.3 | 657.4 | 38.6 |
| 6144, tuned to fit | 32 | 443.4 | 578.7 | 42.1 |
| 8192 requested | 96 and 16 tried | Not run at 8192 | Not run at 8192 | Not run at 8192 |

The fresh control's prefill medians were 1.13x and 1.14x the 6K configuration's. This compares two complete configurations: **chunk and ring both differ**. It does not isolate the chunk-size effect from the ring effect.

Raw timed rates:

- 4096 / 4K: 305.9, 529.3, 501.3.
- 6144 / 4K: 447.9, 374.2, 443.4.
- 4096 / 16K: 646.6, 742.7, 657.4.
- 6144 / 16K: 448.1, 578.7, 620.2.
- 4096 / decode: 38.6, 39.0, 37.7.
- 6144 / decode: 41.0, 42.1, 42.9.

Decode used the same short coding prompt as before, with a 128-token generation warmup followed by three full answers, capped at 512 new tokens. The answers had 352–359 output tokens for the new 4K control and 295–352 for 6K. The adaptive expert cache remained enabled. These are observed rates, not evidence that a larger prefill chunk causes faster decode.

The earlier 4096 session measured 644.6 / 947.4 / 51.8 tokens/s; the fresh control here is slower. Cache state, storage activity and other system conditions were not completely controlled. Do not merge those sessions into one table or infer a stable decode change between them. The purpose of this follow-up control is to compare against 6K under the current conditions. There is visible variation and warming within runs; no confidence interval or statistical significance is claimed.

## Fit checks

| Requested chunk | Ring slots | Engine-reported cap | Action |
|---|---:|---:|---|
| 6144 | 96, usual setting | 3072 | Recorded fallback; no timings labeled 6K |
| 8192 | 96, usual setting | 4096 | Recorded fallback; no timings labeled 8K |
| 6144 | 32 | 6144 | Ran all prefill and decode checks |
| 8192 | 16, minimum ring | 4096 | Recorded fallback; no timings labeled 8K |

The first stage is still the RTX 4060 with two other apps retaining about 3,123 MiB of its VRAM. Its expert cache held 1,159 slots / about 2.93 GiB. True 6K with ring 32 borrowed 1,004 of those slots (2.53 GiB), leaving 155 expert slots; 4K with ring 96 borrowed 789 slots (1.99 GiB), leaving 370. The engine enforces a 128-slot floor. That leaves little room to raise the chunk further on this card. The tests did not close the other apps, change GPU placement, or reduce the context or MTP vocabulary.

The 6K startup log also reports that none of its 2,154 borrowed slots across both cards have a RAM copy under the 40 GiB cap; those experts are read from the model file. The logs contain substantial storage reads during the timed prompts. A smaller ring changes transfer overlap, and the larger loan changes which weights must be sourced from the file. These are plausible contributors to the slower result; this experiment does not assign a separate measured cost to each.

## Validation and saved state

Hardware: i5-13500, RTX 5070 Ti 16 GB + RTX 4060 8 GB, 64 GB RAM; installed Strata 0.1.45 and native UD-IQ4_XS. Same 131,072 context, INT8 KV, GPU order `[1,0]`, split at layer 5, full 248,320-token MTP vocabulary, PCIe fraction .20, spec 4, cutoff .70, reserves `700,1152` MiB and expert profile as the earlier report. Both timed starts reported 4,242 expert slots and 40,958 MiB pinned expert RAM.

All retrieval checks passed with zero prompt reuse. All three longer coding outputs per timed configuration passed their own assertions and two extra median cases after inspection. No non-finite or transfer error was reported. Disk prompt caching and conversation checkpoints were disabled for measurements, and the same saved 4K/16K token IDs were used.

Every isolated engine exited normally on input EOF (code 0). SHA-256 checks confirmed the saved config, installed engine, expert profile and full draft vocabulary were unchanged. The production server was already stopped and remains stopped. The saved production config remains **prefill 4096, resident budget 40 GiB, usual ring**; the experimental ring 32 was used only by the isolated test process.

## Records and reproduction

[larger-chunks-summary.json](larger-chunks-summary.json), [fresh 4K control](control-4096-after-6k-40g.json), [true 6K test](tuned-6144-ring32-40g.json), [default 6K fit check](matched-6144-40g.json), [default 8K fit check](matched-8192-40g.json), [minimum-ring 8K fit check](fit-8192-ring16-40g.json). Corresponding engine/protocol logs are saved beside each JSON file.

After the user has stopped any production engine:

```sh
python bench/results/2026-10-19-iq4-xs-prefill-chunks/run.py --chunk 6144 --tag repeat-6k-ring32 --ram-budget 40 --ring 32 --runs 3 --warm-prefill --decode-runs 3 --decode-max 512 --require-chunk
python bench/results/2026-10-19-iq4-xs-prefill-chunks/run.py --chunk 8192 --tag repeat-8k-ring16 --ram-budget 40 --ring 16 --runs 3 --warm-prefill --decode-runs 3 --decode-max 512 --require-chunk
python bench/results/2026-10-19-iq4-xs-prefill-chunks/run.py --chunk 4096 --tag repeat-4k-control --ram-budget 40 --ring 96 --runs 3 --warm-prefill --decode-runs 3 --decode-max 512 --require-chunk
```

`--require-chunk` records a fallback and exits the finite input stream without timing the smaller resolved size. No source/backend changes or production configuration changes were made.
