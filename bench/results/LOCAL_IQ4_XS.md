# Saved IQ4_XS experiments on the 5070 Ti + 4060

Measured 2026-10-03 and 2026-10-04 on the Intel i5-13500, 64 GB RAM, RTX 5070 Ti 16 GB and RTX 4060 8 GB, Linux. These are historical experiments, with the exact saved prompts, requests, responses, logs, configs and measurement scripts retained.

The selected deployment is the existing IQ4_XS/Q2 MTP configuration with all 248320 draft token IDs, physical GPU order [1,0] (4060 then 5070 Ti), and layer_split 4: the 4060 runs layers 0–3 with all 2048 experts cached for decode; the 5070 Ti runs layers 4–47 plus output head/MTP. The dense-weight placement fix is in the engine source. Historical configs are records, not portable active defaults; their absolute model paths and executable hashes refer to the original PC/build.

## Reports

- [Full-vocabulary Q2 versus Q4_K_M MTP](2026-10-03-iq4-xs-mtp/full-vocab/report.md).
- [Initial GPU placement](2026-10-03-iq4-xs-gpu-placement/report.md), [4060 first](2026-10-03-iq4-xs-gpu-placement/4060-first-report.md), [expert helper](2026-10-03-iq4-xs-gpu-placement/helper-report.md).
- [Dense-weight allocation fix](2026-10-03-dense-layer-placement/report.md).
- [Layers 46–47 + head/MTP on the 4060](2026-10-04-iq4-xs-k46/report.md).
- [Marketing/Python: expert helper versus fully cached first four layers](../../evaluations/2026-10-04-helper-vs-four-layers/REPORT.md).
- [Marketing/Python: fully cached final four layers, head/MTP on the 5070 Ti](../../evaluations/2026-10-04-head-on-primary/REPORT.md), [engine correctness/capacity](2026-10-04-head-on-primary/report.md).
- [Earlier IQ3_S versus IQ4_XS tasks](../../evaluations/2026-10-03-iq3-s-vs-iq4-xs/REPORT.md), [27B comparison using those same tasks](../../evaluations/2026-10-03-qwen3-8-27b-q4-k-m/REPORT.md).
- [Return-stage feasibility](2026-10-04-return-stage-feasibility/report.md) and [return-stage correctness](2026-10-04-return-stage/report.md). The related [task run](../../evaluations/2026-10-04-return-stage/run-manifest.json) was cancelled by the user and is excluded from final performance comparisons.

The experimental head/return-stage code remains as archived patches and source snapshots in these records; it is not enabled in the fork's current engine source. The 71.7 MiB before-strata executable and Python bytecode caches were excluded. Model weights and runtime engine binaries are not included. The saved manifests retain historical executable hashes. Re-running scripts may need the referenced older build; do not treat them as ready-to-run deployment commands.

The file inventory in [iq4-xs-published-records.json](iq4-xs-published-records.json) preserves the SHA-256 of each copied benchmark/evaluation file. GPU performance measurements used their recorded older builds; rebasing the source fix onto upstream 0.1.38 does not remeasure those results.
