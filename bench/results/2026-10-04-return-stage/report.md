# Return to the first GPU: engine patch and validation

The engine accepts an explicit final stage on CUDA0. The tested schedule is layers 0–42 on the RTX 5070 Ti, layers 43–46 on the RTX 4060, and layer 47 plus output head and Q2 MTP on the RTX 5070 Ti. This is a three-stage schedule on two physical GPUs.

Use `--layer-split 43,47 --split-device 1,0` with the 5070 Ti visible as CUDA0 and the 4060 visible as CUDA1. A server configuration uses `gpu: [0,1]`, `layer_split: "43,47"`, and the split-device pair in args. Automatic placement continues to search distinct GPUs.

Changes: the explicit schedule validator accepts a final return to CUDA0; startup reuses that device context; the first cache reserves the returning stage's profiled expert bytes, verification window, any separate prompt allocation, and MTP bind allocation; variable-size cache accounting and the post-allocation shrink check preserve that reservation. Stage weights, session, cache, verifier and prefill objects remain independent. The schedule reports two hand-offs per verification window.

The returning layer 47 is fully cached in the current policy (512 experts, 1.50 GiB). Its cache is reserved before the main-stage cache, which holds 2020 experts (4.49 GiB). This is a conservative allocation; the first and last stages do not yet share one globally ranked expert-cache allocation.

## Validation

- Release build for CUDA architectures 89 and 120 succeeded. Existing unused argmax warning remains.
- Standalone split schedule validator: 14 cases passed. Existing dense placement ownership/allocation test passed.
- Existing two-stage layout: all three saved English/Python/Vietnamese greedy responses identical before and after the patch with identical residency and adaptation disabled.
- Returning layout: all three greedy responses identical to the contiguous two-GPU split with identical residency. This is limited correctness evidence, not a full model equivalence proof.
- Five middle layers (42–46): only 2313 of 2560 expert slots filled (5.37 GiB), so the full-cache requirement fails with unchanged settings.
- Four middle layers (43–46): all 2048 experts filled (4.83 GiB). The 4060 keeps separate 0.41 GiB prompt buffers. Layer 47: all 512 experts filled.
- The full 248320-token draft head loaded on CUDA0; Q2 MTP loaded there. The production executable, config, custom vocabulary and expert profile hashes stayed unchanged.

Settings: IQ4_XS, 131072 context, INT8 KV, 512-token prefill chunks, spec=4, 1248 MiB reserve, PCIe fraction .20, suffix drafts disabled. Validation alone disables adaptation and uses a small fixed-residency profile for parity; capacity checks use the complete production profile.

Measured on the Intel i5-13500, 64 GB RAM, RTX 5070 Ti 16 GB and RTX 4060 8 GB, Linux, 2026-10-04. No production server was started. Temporary test servers shut down after each phase.

Validation build SHA-256: 61094912754afe9a423f97c5e0a62769844356dd04386410ebe5e2fde3c3e199. Final task-evaluation build SHA-256: 1195ee39f01cca496d2f78acfac8c5e228248ab7a92ee47ce657fcd8d65c2fc4. The final rebuild changes the startup label from GPUs to stages; functional return-stage code is the same.

The full marketing and Python proposal/review evaluation is saved separately at ../../../evaluations/2026-10-04-return-stage/. Those performance results are independent requests and must not be inferred from the short validation prompts.

Files: return-stage.patch contains only this change against the prior working tree; before-source preserves the prior source files; before-strata preserves the prior dense-placement binary; validation.json and the per-run JSON/log files contain raw evidence.
