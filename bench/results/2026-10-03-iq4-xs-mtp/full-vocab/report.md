# Full vocabulary: Q2 versus Q4_K_M on IQ4_XS

Intel i5-13500, RTX 4060 then RTX 5070 Ti; MTP on the 5070 Ti. Both arms use all 248,320 token IDs, the same engine, layer split 2, 131,072 context, three draft tokens and automatic expert cache. Suffix lookup is disabled in both arms. Only the MTP runtime differs.

18 requests per arm, 256 generated tokens each, two repeats per language/condition; thinking disabled. Conditions: greedy cutoff 0.70, greedy cutoff 0, sampled temperature 0.8/top-p 0.95/top-k 20/cutoff 0.70. Decode speed is total generated tokens divided by summed decode time. Acceptance is total accepted drafts divided by total offered drafts. Short prompts and cache adaptation limit generalization.

| Condition | Output | Q2 accepted/offered | Q2 acceptance | Q4 accepted/offered | Q4 acceptance | Q2 tok/s | Q4 tok/s | Acceptance delta (pp) | Speed delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_greedy | english | 267/385 | 69.4% | 273/393 | 69.5% | 49.6 | 45.4 | +0.1 | -8.4% |
| current_greedy | code | 356/396 | 89.9% | 360/400 | 90.0% | 42.8 | 37.4 | +0.1 | -12.8% |
| current_greedy | vietnamese | 207/308 | 67.2% | 230/337 | 68.2% | 42.2 | 41.5 | +1.0 | -1.6% |
| unfiltered_greedy | english | 301/633 | 47.6% | 292/657 | 44.4% | 45.6 | 39.8 | -3.1 | -12.7% |
| unfiltered_greedy | code | 369/435 | 84.8% | 366/444 | 82.4% | 43.6 | 39.5 | -2.4 | -9.4% |
| unfiltered_greedy | vietnamese | 266/738 | 36.0% | 282/693 | 40.7% | 38.9 | 37.0 | +4.6 | -5.0% |
| current_sampled | english | 266/398 | 66.8% | 280/388 | 72.2% | 49.8 | 45.6 | +5.3 | -8.6% |
| current_sampled | code | 353/382 | 92.4% | 363/404 | 89.9% | 38.2 | 37.4 | -2.6 | -2.1% |
| current_sampled | vietnamese | 215/297 | 72.4% | 240/335 | 71.6% | 43.6 | 43.7 | -0.7 | +0.4% |

Visible output token eligibility was 100% in every run. Both engine logs confirm a 248,320-token draft head (497.3 MiB). Larger Q4 MTP weights leave fewer slots for the automatic target expert cache; the same cache setting does not imply equal cache capacity.

No production configuration or installed vocabulary was changed. Their hashes matched before/after. The port 1111 server was left unloaded; the temporary test server was stopped. No configuration was applied and no model was restored.
