# Head-on-primary task timings

New head-on-primary build versus saved prior build; existing two-stage matched-residency output regression passed. Independent runs, differing generated histories/output lengths. Helper hit rate excludes remote helper work and is not combined-GPU coverage.

| Setup | Task | Output tokens | Read tokens | Reused tokens | Prefill s | Prefill tok/s | Decode s | Decode tok/s | Request wall s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| helper | marketing | 951 | 406 | 0 | 3.56 | 114.00 | 22.99 | 41.36 | 26.74 |
| helper | python | 20087 | 132319 | 108423 | 612.23 | 216.13 | 458.51 | 43.81 | 1075.29 |
| four-layers | marketing | 1190 | 406 | 0 | 3.15 | 129.06 | 27.99 | 42.52 | 31.37 |
| four-layers | python | 16767 | 129123 | 102087 | 505.64 | 255.36 | 391.38 | 42.84 | 901.32 |
| head-primary | marketing | 1123 | 406 | 0 | 2.93 | 138.39 | 29.05 | 38.66 | 32.20 |
| head-primary | python | 15740 | 125876 | 101401 | 533.76 | 235.83 | 400.49 | 39.30 | 938.62 |

| Head-on-primary request | Read / reused | Prefill tok/s | Prefill s | Output | Decode tok/s | Decode s | Wall s | Finish |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| head-primary-marketing-1 | 406 / 0 | 138.40 | 2.93 | 1123 | 38.70 | 29.05 | 32.20 | stop |
| head-primary-python-1 | 24796 / 0 | 248.00 | 100.00 | 3558 | 40.80 | 87.15 | 187.69 | stop |
| head-primary-python-2 | 67 / 28354 | 37.30 | 1.79 | 3161 | 38.30 | 82.58 | 84.99 | stop |
| head-primary-python-2-continuation-1 | 31619 / 0 | 245.80 | 128.61 | 1298 | 40.20 | 32.31 | 161.54 | stop |
| head-primary-python-2-continuation-2 | 32954 / 0 | 229.40 | 143.66 | 1427 | 40.90 | 34.88 | 179.12 | stop |
| head-primary-python-3 | 99 / 34382 | 70.20 | 1.41 | 1633 | 38.00 | 42.95 | 45.06 | stop |
| head-primary-python-4 | 36254 / 0 | 231.10 | 156.87 | 2411 | 39.90 | 60.46 | 217.89 | stop |
| head-primary-python-5 | 87 / 38665 | 61.70 | 1.41 | 2252 | 37.40 | 60.17 | 62.33 | stop |

| Phase | Load s | Elapsed including startup/shutdown s |
|---|---:|---:|
| initial | 46.22 | 355.16 |
| stage2-continuation | 55.49 | 221.11 |
| stage2-second-continuation-and-final | 46.09 | 274.40 |
| followups-final | 46.06 | 331.12 |
