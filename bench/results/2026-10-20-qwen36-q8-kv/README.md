# Qwen3.6 shared engine, Q8 runtime and validation — 2026-10-20

## Active profile

The fork tracks one local Qwen3.6 configuration at
[`strata-qwen36-ud-q4_k_xl.json`](../../../strata-qwen36-ud-q4_k_xl.json):

- UD-Q4_K_XL on RTX 5070 Ti (GPU 0), with the remaining experts in system RAM.
- 131,072 context capacity, Q8 KV (`--kv int8`), 2,048-token prefill chunks.
- 13 pool workers plus the host thread, `--mtp 2`, full draft vocabulary.
- Greedy sampling with presence/frequency penalties off and repetition penalty 1.
- Localhost only: web chat `http://127.0.0.1:1234/`, API `http://127.0.0.1:1234/v1`.

`./run-qwen36-ud-q4_k_xl.sh` launches this configuration. The user's
`~/Workplace/llama-cli/qwen3.6-35b-a3b.sh` delegates to that same launcher. Both
serve the same engine and defaults. Requests with explicit sampling parameters
can override the defaults; nonzero temperature or active penalties bypass this
upstream Qwen3.6 engine's greedy-only MTP path.

Live verification through the `llama-cli` launcher is saved in
[`runtime-verification.json`](runtime-verification.json) and
[`runtime-startup.txt`](runtime-startup.txt): web root HTTP 200, API context 131072,
and MTP drafts active for requests omitting sampling in both thinking modes.
These functional requests are not a new throughput benchmark.

This is the measured configuration for this Linux desktop, not a change to
engine-wide defaults or other models. The GGUF path is local to this machine.
Flash-Next's configuration, launcher and full 248,320-token Q2 draft list are unchanged.

## Controlled decode measurements

Hardware: Intel i5-13500, RTX 5070 Ti 16 GB, 64 GB RAM, Linux. Each case uses a fresh
process, normal engine warmup, then each of the ten upstream `parity_prompts.ids`
lists once, up to 128 generated tokens each, greedy, MTP 2, 2K prefill, 14 CPU
threads and automatic GPU expert placement. TPS is total generated tokens divided
by summed engine decode time, excluding prefill and model loading.

The prompts contain 20-40 tokens. **Context below is allocated capacity, not a
filled conversation. These are not measurements of decode after a 128K history.**

| Controlled comparison | First run TPS | Second run TPS | Combined TPS | GPU experts/layer |
| --- | ---: | ---: | ---: | ---: |
| Q8, 8K, against FP32 controls | 87.6 | 89.8 | 88.7 | 89/256 |
| FP32, 8K, same comparison | 88.0 | 89.1 | 88.6 | 86/256 |
| Q8, 8K, fresh capacity controls | 95.9 | 99.7 | 97.8 | 89/256 |
| Q8, 128K, same capacity comparison | 78.8 | 83.6 | 81.1 | 62/256 |

The first comparison ran Q8/FP32/FP32/Q8; the second ran 128K/8K/8K/128K. Q8 at
8K matched FP32 within observed variation. Raising Q8 capacity to 128K reduced
overall throughput 17.0% in the second comparison and held fewer GPU experts;
MTP remained active. Per 128K process, 567/794 drafts were accepted and 963 tokens
generated. Each format/capacity's two repeats generated identical token IDs.
Quantization and changing CPU/GPU expert placement changed some outputs across
cases; these comparisons do not establish bit-identical output or general quality.

There is visible variation between sessions, hence the fresh controls. Four runs
per comparison do not establish a maximum rate or a confidence interval. The
observed suite rate is not a minimum for every chat request.

[`decode/results.json`](decode/results.json) indexes all eight raw runs, including
binary and prompt hashes, engine timings and draft counters. Each run's JSON
preserves output token IDs; matching `.txt` files preserve engine startup and
decode logs. No model binary is committed.

To reproduce while production is already stopped by the user:

```sh
.venv/bin/python bench/results/2026-10-20-qwen36-q8-kv/run_decode.py
# Same setup at 8K; FP32 control also uses --kv fp32.
.venv/bin/python bench/results/2026-10-20-qwen36-q8-kv/run_decode.py --context 8192 --kv int8
```

The runner refuses to start while an engine named `strata` is running. It exits
only its own benchmark process through stdin EOF and does not edit saved settings.

## Engine change and memory

`--kv int8` stores K and V in groups of 64 signed bytes with one FP16 scale per
group. Main attention, MTP, batch tables and conversation copies use it. Prefill
gathers it into shared FP16 buffers; decode accumulates in FP32. GDN stays FP32.
Without this opt-in flag the engine retains its FP32 path. Unsupported formats fail
explicitly. Related upstream issue: [#3303](https://github.com/Niko1221/Strata/issues/3303).

For this model's 10 main full-attention layers plus MTP, two KV heads and head
dimension 256, main-plus-MTP KV allocation at 128K is 5.500 GiB FP32 versus
1.418 GiB Q8, a 4.082 GiB saving. At 8K it is 0.344 versus 0.089 GiB. These counts
exclude recurrent state, shared buffers, model weights and expert slots.

The shared CUDA binary was built with `STRATA_MMQ_KQUANTS=ON`; its adjacent
`libstrata_kq.so` supplies the existing upstream K-quant MMQ prefill kernels. Q8
KV and MMQ prefill are separate features. Q8 is this fork's opt-in addition.
No qwen4exp source or shared decode kernel was changed.

## Engine validation retained from the original change

- CUDA compiled for sm_89 and sm_120. Model-free Q8 codes/scales, FP16 gather,
  causal attention, graph capture and batch-table checks passed on RTX 4060 and
  RTX 5070 Ti under Compute Sanitizer with zero errors. Maximum attention absolute
  error against the host reference: 0.0000012493. CTest also passed.
- FP32 opt-out: ten prompts, up to 64 generated tokens, 551 total; output IDs
  exactly matched the backed-up original binary with fixed expert slots.
- Q8 versus FP32: 22 teacher-forced English/Vietnamese rows; mean KL 0.002431,
  maximum 0.022979, same top prediction in all rows. This small check is not a
  general quality benchmark. `validate.py quality` reproduces it; temporary full
  vocabulary dumps were discarded after retaining metrics and inputs.
- Q8 solo/MTP versus batch MTP and session copying passed. Copying reused 99/100
  prompt tokens. An 18,040-token retrieval prompt returned `STRATA-73921`; the
  original 4K-prefill staged-binary check read it at 3,013 TPS. This is historical
  validation of the engine, not the current 2K profile's prefill measurement.
- Flash-Next IQ4_XS old/new shared binary: dual GPU, 128K, INT8 KV, 4K prefill,
  40 GiB resident and full Q2 draft vocabulary; three short requests plus 4K
  retrieval gave identical output IDs, 123 tokens total, and equal expert slots.
- A harder arithmetic smoke test returned incorrect `425` instead of `403` on
  both original and Q8 engines; it is retained as a model failure, not a pass.
- Existing Qwen3.6 server configuration tests passed. HIP configure failed because
  this PC has no ROCm toolchain; HIP compile/execution were not validated. This
  family does not support SYCL. No new engine source was changed for the runtime
  profile update.

`validate.py`, `flashnext_compat.py` and their JSON/log evidence remain for engine
reproduction. Historical raw paths and `installation.json` describe the original
installation and its rollback backup, not the current runtime settings. The
rollback backup under `engine/.runtime-backups/` is preserved.

## Earlier measurements and correction

`bench.json`, `fresh-decode.json` and their logs retain the earlier installation
checks: 128K capacity, nine workers, FP32/native versus Q8/MMQ, 2K versus 4K chunks.
Their short coding prompt was sampled at temperature 0.7 and produced no MTP drafts.
Repeated coding-prompt warmups reached 80.7 TPS with 4K chunks, while generation
after unrelated long prompts was 51.8 TPS. That narrow 80.7 result was wrongly
presented as representative chat speed. It is not the basis for the active profile.

The initial FP32/128K setup still held 23 GPU experts/layer; saying it left no room
for experts was incorrect. An 8K FP32 experiment also does not establish FP32
performance at 128K. Those settings are no longer saved as alternate launchers.
Duplicate configs, temporary benchmark folders, staged binary and diagnostic dumps
were removed; the shared installed binary and the relevant measurements remain.
