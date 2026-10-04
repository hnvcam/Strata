# Observed timing: expert helper versus fully cached first four layers

| Setup / task | Output tokens | Decode tok/s | Read tokens | Reused tokens | Prefill tok/s | Prefill seconds | Request wall seconds | MTP acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| helper / marketing | 951 | 41.36 | 406 | 0 | 114.00 | 3.56 | 26.74 | 74.39% |
| helper / python | 20087 | 43.81 | 132319 | 108423 | 216.13 | 612.23 | 1075.29 | 87.71% |
| helper / all | 21038 | 43.69 | 132725 | 108423 | 215.54 | 615.79 | 1102.03 | 87.13% |
| four-layers / marketing | 1190 | 42.52 | 406 | 0 | 129.06 | 3.15 | 31.37 | 70.94% |
| four-layers / python | 16767 | 42.84 | 129123 | 102087 | 255.36 | 505.64 | 901.32 | 87.35% |
| four-layers / all | 17957 | 42.82 | 129529 | 102087 | 254.58 | 508.79 | 932.68 | 86.31% |

| Setup / request | Output tokens | Decode tok/s | Read / reused input tokens | Prefill tok/s | Prefill seconds | Wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| helper-marketing-1 | 951 | 41.40 | 406 / 0 | 114.00 | 3.56 | 26.74 |
| helper-python-1 | 3718 | 41.20 | 24796 / 0 | 190.80 | 129.96 | 220.87 |
| helper-python-2 | 4096 | 39.60 | 67 / 28514 | 41.10 | 1.63 | 105.77 |
| four-layers-marketing-1 | 1190 | 42.50 | 406 / 0 | 129.10 | 3.15 | 31.37 |
| four-layers-python-1 | 3485 | 40.20 | 24796 / 0 | 215.30 | 115.17 | 202.52 |
| four-layers-python-2 | 3612 | 38.40 | 66 / 28282 | 33.40 | 1.98 | 96.60 |
| helper-python-2-continuation-1 | 1903 | 44.10 | 32715 / 0 | 208.50 | 156.94 | 200.75 |
| four-layers-python-2-continuation-1 | 2048 | 45.00 | 31997 / 0 | 268.80 | 119.02 | 165.03 |
| helper-python-2-continuation-2 | 2473 | 46.30 | 34655 / 0 | 233.40 | 148.51 | 202.59 |
| helper-python-3 | 2532 | 45.40 | 99 / 37129 | 77.30 | 1.28 | 57.74 |
| helper-python-4 | 2880 | 44.90 | 39900 / 0 | 231.20 | 172.59 | 237.48 |
| helper-python-5 | 2485 | 51.50 | 87 / 42780 | 65.90 | 1.32 | 50.10 |
| four-layers-python-2-continuation-2 | 712 | 44.00 | 34083 / 0 | 268.80 | 126.81 | 143.55 |
| four-layers-python-3 | 2241 | 45.10 | 819 / 34076 | 204.10 | 4.01 | 54.40 |
| four-layers-python-4 | 2452 | 46.70 | 37276 / 0 | 271.40 | 137.36 | 190.53 |
| four-layers-python-5 | 2217 | 47.50 | 86 / 39729 | 66.10 | 1.30 | 48.71 |

| Setup | Initial startup seconds | First continuation startup | Second continuation startup | Final follow-up startup | Startup + matched request seconds | Actual phase seconds including superseded work and shutdown |
|---|---:|---:|---:|---:|---:|---:|
| helper | 57.05 | 53.54 | 45.13 | 46.81 | 1304.56 | 1722.20 |
| four-layers | 52.28 | 54.23 | 49.26 | 48.24 | 1136.69 | 1153.01 |

Decode rates are output-token-weighted; prefill rates count only input tokens actually read. Stages 1 and 4 read cold histories, matching the saved evaluation procedure. The helper reached the stage-2 output cap; the four-layer first continuation also reached its cap. Both receive two identical saved continuation prompts, with budgets of 2048 then 6144 tokens. The second budget is the only output-limit change from the saved cases. Both continuation requests read cold histories. Continuation time is included. Superseded helper stages 3–5 from the one-continuation attempt are archived and excluded from matched task rates and request times, but included in actual phase elapsed time. Other turns can reuse conversation checkpoints. Output lengths and assistant histories differ, so wall-time and multi-turn prefill differences are observed workload outcomes, not equal-output comparisons. One run per placement, one fixed seed, no statistical significance claim.
