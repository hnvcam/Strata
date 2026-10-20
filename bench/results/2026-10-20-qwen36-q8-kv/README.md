# Qwen3.6 Q8 KV and K-quant prefill, 2026-10-20

**Installed setting: Q8 KV, K-quant MMQ, 4,096-token prompt chunks, 131,072 context capacity.** It gives the fastest
prefill of these local cases. The 2K Q8 setting gives faster fresh-chat generation, so both measurements are retained.

Fork change to the shared Strata 0.1.45 executable, installed locally after the user's approval. Related upstream issue:
[Qwen3.6 always allocates FP32 KV, #3303](https://github.com/Niko1221/Strata/issues/3303).
No issue comment or pull request was posted.

## Change

`--kv int8` stores K and V in groups of 64 signed bytes with a two-byte FP16 scale per group. Both main attention
and MTP use it. Prefill reconstructs the cache in the existing shared FP16 buffers; decode reads the Q8 groups
and accumulates in FP32. Batch sessions and conversation copies retain the compressed representation. GDN state
is unchanged. Without the flag the cache remains FP32; unsupported KV formats fail explicitly.

The candidate is also compiled with `STRATA_MMQ_KQUANTS=ON`. The installed build had that option off, so its
Qwen3.6 prefill used native grouped kernels and the CPU pool for every prompt chunk. Startup of the candidate
confirms `prompt experts on MMQ (int8)` and a 1.10 GiB staging pair for streamed experts. These are two changes
to the complete configuration; the performance comparison does not isolate their separate contributions.

The model header contains 10 full-attention main layers plus one MTP layer, two KV heads, and 256 values per
head. At 131,072 cells, K/V allocation including scales changes from **5.50 GiB to 1.418 GiB**, freeing
**4.082 GiB**. Context capacity stays 131,072. Shared FP16 prefill buffers, attention partials, GDN state and
weights are additional allocations.

## Validation

- CUDA build for sm_89 and sm_120, including the new engine and `libstrata_kq.so`.
- Model-free Q8 code/scale, prefill gather, causal split attention, graph capture and batch-session selection
  tests on both the RTX 5070 Ti and RTX 4060: zero errors under Compute Sanitizer memcheck. Maximum attention
  absolute error against the host reference: 0.0000012493. The CMake/CTest target also passed.
- FP32 opt-out compared with the installed binary using the same fixed 23 expert slots, 2,048-token chunk,
  native prefill and 131,072 capacity: all 10 prompts gave identical token IDs, up to 64 generated tokens each
  (551 tokens in total). See `baseline-fp32.json` and `candidate-fp32.json`.
- Q8 versus FP32, fixed slots and native prefill, a shared English/Vietnamese prompt and teacher-forced
  continuation: 22 rows, mean KL 0.002431, maximum 0.022979, top prediction identical in all 22. This small check
  is not a general quality benchmark. Raw logits are the `quality-*.full` files; per-row KL is in `quality.json`.
- Q8 solo/MTP versus two concurrent slots with batch MTP: identical outputs. Copying a held prefix into the
  other session reused 99 of 100 prompt tokens and gave the fresh continuation's output. Simple arithmetic
  returned 4; an 18,040-token retrieval prompt returned `STRATA-73921`. See `integration.json` and its log.
- Final check of the exact staged binary, adjacent K-quant library, automatic expert placement and proposed
  4K chunk config also retrieved `STRATA-73921` from 18,040 tokens (3,013 tok/s prompt reading). See
  `candidate-smoke.json`. Every isolated engine exited normally; the production engine remains stopped.
- A harder arithmetic smoke prompt returned the incorrect `425` instead of `403` on **both the installed
  binary and Q8**. This was preserved as an existing model result, not marked as a correct answer.
- Existing server Qwen3.6 config tests: 4 passed. `git diff --check` passed.
- Before replacing the shared executable, the old and new binaries ran the saved Flash-Next IQ4_XS config:
  GPU order `[1,0]`, split at layer 5, 131,072 context, INT8 KV, 4K prefill, 40 GiB resident RAM and the full
  248,320-token Q2 MTP vocabulary. Profile saving and prompt/conversation caches were disabled only in the test
  processes. Three short requests (up to 64 tokens each) and a 4K retrieval request gave identical token IDs,
  123 generated tokens total. Both builds allocated 5,437 expert slots (2,354 on the primary stage). Saved
  config, launcher, expert profile and draft vocabulary hashes were unchanged. See `flashnext-compat.json`,
  `flashnext_compat.py` and the two engine logs. This checks those requests, not all possible Flash-Next behavior.

HIP configure failed because this PC has no ROCm toolchain; HIP compilation and AMD execution were not
validated. This model's own engine path does not support SYCL. No review request was opened.

## Performance method

Hardware: Intel i5-13500, RTX 5070 Ti 16 GB, 64 GB system RAM, Linux. GPU 0 only; the RTX 4060 continues to host
the user's other apps. Every case uses the same UD-Q4_K_XL GGUF, 131,072 context capacity, nine pool workers
(plus the host thread), `--mtp 2`, and the same saved prompt IDs. All test engines run independently of the
production server and exit on EOF. Expert-cache adaptation stays enabled.

Cases: installed FP32 with 2K chunks; candidate Q8/MMQ with 2K chunks; candidate Q8/MMQ with 4K chunks. Each
process warms with a 64-token generation, then reads two fresh 4,096-token prompts and one fresh 16,384-token
prompt. Timed reads reuse zero cached prompt tokens. Next it generates three answers capped at 256 tokens
from the same 34-token coding prompt, temperature 0.7 and seeds 100/101/102. These sampled generations make
zero MTP draft attempts, matching the original slow sampled path; they do not measure greedy MTP throughput.

Cases run sequentially, not as interleaved pairs. There is visible cache warming and system variation; the
16K read has one timing per case. Results describe these local smoke measurements, not a sustained throughput
guarantee or a 128K-history decode benchmark. The longest functional prompt checked was 18,040 tokens.
Raw timings and aggregates are saved in `bench.json`, `bench.stdout` and the case logs.

## Measurements

Tokens per second; descriptive timings from one local session, without a confidence interval:

| Setting | 4K prompt, median of 2 | 16K prompt, 1 read | Fresh-chat generation, median of 3 | Generation after unrelated long prompts, median of 3 |
| --- | ---: | ---: | ---: | ---: |
| Installed FP32, native prefill, 2K chunk | 221.7 | 276.8 | 68.4 | 54.2 |
| Q8 KV + MMQ, 2K chunk | 1,722.1 | 1,824.1 | 91.2 | 56.0 |
| Q8 KV + MMQ, 4K chunk | 3,304.1 | 3,532.3 | 80.7 | 51.8 |

Fresh-chat timings were measured in separate new processes after a 128-token warmup on the same coding prompt,
with no unrelated long document read before generation. All other generation settings are the same. Their raw
rates were 64.8/68.4/71.0 (installed), 84.2/91.2/91.4 (Q8 2K), and 67.1/80.7/87.4 (Q8 4K).
See `fresh-decode.json`. Startup placed 23 experts per layer in VRAM on the installed build, 62 on Q8 2K and
49 on Q8 4K. The 4K buffers cost some expert slots compared with 2K. Retained routing scores/cache state are a
plausible reason for the slower generation after an unrelated document; these runs do not isolate its exact cost.

The installed 4K setting trades some fresh-chat generation speed for faster prompt reading. It does not guarantee
faster generation after a long unrelated prompt: that case was slightly slower than the installed build here.

## Files and use

The source change is committed with the engine/header/test/docs files. `source.patch` is a local generated copy,
excluded from git. `build-manifest.json` records binary hashes and the protected original binary/config hashes.
Full-vocabulary floating-point dumps are retained locally and excluded from git; the per-row KL is committed.
`validate.py` reproduces the model checks and speed measurements;
it refuses to run while another process named `strata` is running.

`candidate-config.json` and `proposed-production-config.json` preserve the tested candidate and proposed config.
After approval, the tested executable and matching `libstrata_kq.so` were installed in `engine/`; both saved
launchers use the same `engine/strata`. The Qwen3.6 config now selects `--kv int8 --prefill 4096` at 131,072
capacity. The Flash-Next config and full draft vocabulary remain unchanged. Engine-level defaults still use
FP32 for Qwen3.6 unless `--kv int8` is passed; the Q8 setting is explicit in this PC's saved Qwen3.6 config.

The prior binary, library, build metadata and Qwen3.6 config were backed up under `engine/.runtime-backups/`.
`installation.json` records the exact backup path and installed hashes. Generated runtime binaries and the
machine's saved configs remain excluded from git by the repository's existing policy. No production server
was started or restarted; the user starts it with their existing launcher.

After installation the validation scripts use the backed-up original executable for baseline comparisons.
On another PC, set `STRATA_BASELINE_EXE` to an original build and adjust the machine/model paths before running;
the scripts must not silently compare the new shared executable against itself.
