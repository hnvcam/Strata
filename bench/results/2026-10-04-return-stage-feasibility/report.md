# Return-stage placement feasibility

Requested sequence: RTX 5070 Ti layers 0–(n−1), RTX 4060 layers n–46, then RTX 5070 Ti layer 47, output head and Q2 MTP.

Inspected the current patched source and linked its existing allocation-sizing functions. This is a metadata/allocation calculation, not an inference test. Production settings, executable, draft vocabulary and model files were not changed. NVIDIA reported 8188 MiB total on the 4060 and 2 MiB used before inspection.

Settings retained: IQ4_XS, 512 experts per layer, 131072 context, INT8 KV, 1248 MiB VRAM reserve, 96 MiB estimate for verification windows. Dense sizing follows current canonical/native ownership and alignment. Full Q2 draft vocabulary remains 248320 tokens; draft/head allocations would live on the 5070 Ti.

| n | Fully cached 4060 layers | Expert storage (GiB) | Dense (MiB) | Session (MiB) | Total incl. windows/reserve (MiB) | Margin against 8188 MiB (MiB) |
|---|---|---:|---:|---:|---:|---:|
| 40 | 40–46 | 8.167 | 689.81 | 208.16 | 10604.47 | -2416.47 |
| 41 | 41–46 | 7.056 | 598.03 | 205.04 | 9372.07 | -1184.07 |
| 42 | 42–46 | 5.945 | 506.25 | 201.92 | 8139.67 | 48.33 |
| 43 | 43–46 | 4.834 | 414.46 | 198.80 | 6907.27 | 1280.73 |
| 44 | 44–46 | 3.723 | 328.15 | 198.80 | 5683.45 | 2504.55 |
| 45 | 45–46 | 2.612 | 236.36 | 195.69 | 4451.05 | 3736.95 |
| 46 | 46–46 | 1.501 | 144.58 | 192.57 | 3218.65 | 4969.35 |

Four layers (n=43) are the conservative candidate. Five layers (n=42) leave only about 48 MiB before CUDA context/module allocations, residency tables, streams, other allocations and prompt buffers; that is not a validated fit. Six layers exceed total VRAM even before those overheads. Exact maximum requires startup and prompt/decode verification in an engine supporting the requested return stage. Full expert caching refers to decode; prompt buffers may temporarily borrow slots and require refill.

Current engine blockers:

- src/program/generate.cpp rejects CUDA0 among later stages except the single-boundary same-GPU diagnostic case. A request with split boundaries n,47 and split devices 1,0 is rejected.
- RemoteExperts::preflight currently refuses device 0; stage setup calls this function for each later stage.
- The first stage sizes/fills its cache before the later-stage caches. If the last stage shares the first GPU, its cache/windows/prompt requirements must be reserved jointly so the first cache cannot consume its space.
- Per-stage sessions, verifiers and prompt/checkpoint handling need validation when the same physical GPU runs two separated stages. Existing coverage demonstrates distinct GPUs and the two-stage same-GPU diagnostic, not this return layout.

The sequence requires two inter-GPU hand-offs per verify window. Existing two-GPU contiguous layouts require one; no performance benefit for this arrangement has been measured.

No inference benchmark was started and no return-stage engine patch was made during this feasibility inspection. The saved marketing and Python prompts/evaluation remain available for the requested later comparison.

Reproducible calculation: capacity.cpp calls NativeDense::served_names, WeightTable::allocation_bytes, DenseWeightSizes::bytes and session_bytes from the current patched build; raw integer bytes are in capacity.csv.
