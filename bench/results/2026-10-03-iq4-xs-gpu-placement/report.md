# IQ4_XS: RTX 5070 Ti alone versus RTX 5070 Ti + RTX 4060

**Updated:** the requested expert-only helper configuration and a fresh single-GPU recheck are in [helper-report.md](helper-report.md). The recheck did not reproduce the initial 14.61-second prompt result. The recommendation below is superseded by that corrected comparison; the original measurements are retained here.

Measured 2026-10-03 on the Intel i5-13500, 64 GB RAM, RTX 5070 Ti 16 GB and RTX 4060 8 GB, Linux. The 5070 Ti has a x16 link; the 4060 has a PCIe Gen 3 x4 link. Engine 0.1.34.

The single GPU and the default dual-GPU setup use the saved production settings: IQ4_XS, Q2 MTP, all 248,320 draft token IDs, 131,072 maximum context, int8 KV, spec=4, spec-min-p=0.70, prefill=512, VRAM reserve=1248 MiB, PCIe fraction=0.20. Suffix lookup was disabled in every test to isolate MTP. Thinking was disabled and sampling was greedy. Adaptive expert caching and the engine's automatic draft policy stayed enabled.

Each full test loaded a fresh engine, warmed up with 128 generated tokens, then ran the same English, Python and Vietnamese prompts twice, capped at 512 output tokens. Some prose answers ended early. Short-request decode speed is total generated tokens divided by total engine decode time; warmup and the long prompt are excluded. Each full test also ran one identical cold 9,851-token prompt, capped at 256 output tokens. The K=44 pilot used one repeat per language and did not run the long prompt.

The completed tuning attempt changes prefill to 256 and VRAM reserve to 1152 MiB, so it measures a tuning attempt as well as GPU placement. Everything else stays matched. Its automatic split still assigns just the final layer to the 4060. An earlier attempt with a 1024 MiB reserve loaded 93 experts (0.27 GiB) on the 4060 but ran out of memory while instantiating verification graphs in its warmup; it has no valid speed result. No source or engine changes were made.

## Measurements

| Setup | Short decode tok/s | Cold 9,851-token prompt tok/s | Prompt seconds | 5070 cache GiB | 4060 cache GiB (slots) |
|---|---:|---:|---:|---:|---:|
| 5070 Ti alone | 41.60 | 674.0 | 14.61 | 5.66 | 0.00 (0) |
| 5070 Ti + 4060, auto (K=47) | 40.73 | 281.3 | 35.02 | 7.71 | 0.06 (19) |
| 5070 Ti + 4060, four layers on 4060 (K=44, pilot) | 37.85 | — | — | 7.72 | 0.05 (18) |
| 5070 Ti + 4060, 1152 MiB reserve (tuned) | 42.37 | 168.7 | 58.40 | 7.80 | 0.15 (51) |
| 5070 Ti + 4060, 1024 MiB reserve (failed) | Failed: Out of memory while instantiating verification graphs | — | — | — | — |

## Short-request results by output

| Setup | English tok/s | Code tok/s | Vietnamese tok/s | MTP acceptance | Expert cache hit rate |
|---|---:|---:|---:|---:|---:|
| 5070 Ti alone | 44.86 | 38.26 | 42.79 | 80.5% | 61.6% |
| 5070 Ti + 4060, auto (K=47) | 43.59 | 37.17 | 42.80 | 81.0% | 67.3% |
| 5070 Ti + 4060, four layers on 4060 (K=44, pilot) | 41.88 | 35.27 | 37.42 | 82.5% | 64.9% |
| 5070 Ti + 4060, 1152 MiB reserve (tuned) | 46.56 | 39.56 | 42.23 | 80.8% | 68.0% |

## Interpretation and limits

Moving the head and MTP to the 4060 increases the 5070 Ti's expert cache, but the default two-card setup was slightly slower on the short requests and much slower on this long prompt. Two repeats per short prompt are not enough to establish a reliable small decode-speed advantage. The long prompt is one repetitive maintenance-log workload, measured once per setup; it does not establish performance at the full 128K context limit.

For these measured workloads, use the 5070 Ti alone: it read the long prompt 2.40 times as fast as the default split and 4.00 times as fast as the tuned split. The tuned split increased the 4060 cache from 19 to 51 experts but improved short decode by only 1.84% over the single GPU in this small sample, while taking longer to read the prompt. The lower reserve was tested only with suffix drafts disabled; it is not validated for the production setup's larger suffix-draft windows or other workloads.

The engine chooses its draft policy automatically: the single-GPU long-prompt response offered zero MTP drafts, whereas the default two-GPU long-prompt response offered 178 and accepted 143. Those long-prompt decode rates therefore reflect each setup's automatic behavior, rather than equal numbers of draft rounds. They are excluded from the short-request aggregate.

Assigning four layers to the 4060 left its cache at 18 slots and slowed the pilot. A split-stage cache stores experts only for that stage's layers; the 4060 cannot also be a general helper cache for the other stage with the current engine and these two cards.

Generated text can differ between placements because CPU and GPU expert calculations round differently and adaptive caching changes residency. This is a speed comparison, not a quality evaluation.

## Saved state

The production JSON, installed Q2 runtime vocabulary, and engine hashes are unchanged (see `initial-manifest.json` and `manifest.json`). Temporary benchmark servers were stopped. The user had stopped the production server and requested no restoration; it remains stopped. Per-request responses and timings, startup logs, temporary configs, the runner, and `summary.json` are saved alongside this report.
