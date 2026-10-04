# Corrected comparison: head and MTP on the 5070 Ti; 4060 holds extra experts

This measures the requested two-card arrangement. The earlier `report.md` measured layer splits that moved the output head and MTP to the 4060; those splits are a different configuration.

Measured 2026-10-03 on the Intel i5-13500, 64 GB RAM, RTX 5070 Ti 16 GB and RTX 4060 8 GB, Linux, engine 0.1.34. The 5070 Ti uses PCIe Gen 4 x16 under load; the 4060 uses Gen 3 x4.

## Placement and settings

The 5070 Ti runs all 48 main-model layers, the output head, Q2 MTP and its 2513-expert cache (5.66 GiB). The 4060 holds 2800 additional main-model experts (6.31 GiB) and computes those expert rows as needed; it does not run a layer stage, the output head or the MTP drafter. Results return through pinned host rows. Startup verifies one cached slot on the helper.

The temporary config keeps `gpu: 0` so the Python server does not add a layer split. Its engine environment sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0,1`; `--expert-cache-device1 2800` enables only the helper cache on CUDA1.

The single-GPU and helper tests use the same IQ4_XS model, installed Q2 MTP runtime, all 248,320 draft token IDs, 131072 maximum context, int8 KV, spec=4, spec-min-p=.70, prefill=512, VRAM reserve=1248 MiB, PCIe fraction=.20 and automatic primary expert cache. Greedy sampling, thinking disabled, suffix drafts disabled. There are no smaller prompt chunks or lower VRAM reserves in this corrected comparison. The original engine and source are used unchanged.

Each fresh engine runs one 128-token warmup, English/Python/Vietnamese prompts twice at up to 512 output tokens, then one identical cold 9851-token maintenance-log prompt with 256 output tokens. The same request order is used. Short decode speed is total generated tokens divided by summed engine decode time, excluding warmup and the long prompt. The initial single-GPU result is retained, and a fresh single-GPU recheck follows the helper measurement.

## Results

| Setup | Short decode tok/s | Cold prompt seconds | Cold prompt tok/s | Long-request decode tok/s |
|---|---:|---:|---:|---:|
| 5070 Ti alone, initial | 41.60 | 14.61 | 674.0 | 55.8 |
| 5070 Ti alone, recheck | 41.69 | 41.45 | 237.6 | 38.2 |
| 5070 Ti head/MTP + 4060 expert cache | 46.61 | 41.70 | 236.2 | 43.7 |

| Setup | English tok/s | Code tok/s | Vietnamese tok/s | MTP acceptance |
|---|---:|---:|---:|---:|
| 5070 Ti alone, initial | 44.86 | 38.26 | 42.79 | 80.5% |
| 5070 Ti alone, recheck | 45.18 | 37.99 | 43.17 | 80.5% |
| 5070 Ti head/MTP + 4060 expert cache | 51.34 | 42.13 | 48.24 | 81.1% |

## What these measurements support

The requested expert-only helper improved short decode by 11.9% against the pooled single-GPU result (41.64 versus 46.61 tok/s). Every language gained in these samples. The helper logs report 16606–19860 expert entries computed per measured short request, confirming that the additional cache was used.

The fresh single-GPU recheck read the long prompt in 41.45 seconds, close to the helper's 41.70 seconds. The initial single-GPU measurement of 14.61 seconds did not reproduce. These data therefore do not establish a consistent prompt-reading penalty from the helper or a reliable prompt-reading advantage from using one GPU. The earlier recommendation based on that initial prompt measurement was too strong.

For these measured workloads, prefer keeping the head and MTP on the 5070 Ti and using the 4060 for this expert-only cache. It improved short decode, and prompt reading was similar to the fresh single-GPU recheck. This corrects the earlier blanket recommendation to use only the 5070 Ti; a helper cache and a layer split are different arrangements.

This is a small speed comparison, not a quality evaluation or a full 128K-context benchmark. The automatic draft policy stays enabled, so draft counts and generated text can differ between placements. The cause of the first long-prompt measurement's discrepancy is unresolved; more repeated cold-prompt measurements would be needed to establish steady prompt throughput. A long reused conversation was not separately measured.

## State and artifacts

The production config, engine and full draft vocabulary are unchanged. The user stopped the production server; benchmark servers were stopped after measurement. The tested helper config is `5070-head-mtp-4060-experts-config.json`. Raw per-request results, engine/server logs, the runner, `helper-manifest.json`, and `helper-summary.json` are in this directory.
