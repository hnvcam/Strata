# Reapply the dense layer placement change

Base Strata commit: `1678de333d0e0711bc414ad992b640e1a37dd814` (0.1.34).
Patch: `2026-10-03-dense-layer-placement.patch`. Contains all nine engine/build/test/documentation files changed for this feature,
including the two new source files. It does not include unrelated AGENTS.md edits, evaluations,
model weights, executable binaries or production configuration changes.

## Apply and rebuild

From the Strata repository root, with the patch stored in `patches/`:

```bash
git apply --check patches/2026-10-03-dense-layer-placement.patch
git apply patches/2026-10-03-dense-layer-placement.patch
cmake --build build --target strata dense_placement_test -j 8
./build/dense_placement_test
```

The existing local build is configured for CUDA 13.0, Release, GPU architectures 89 and 120.
If the build directory is missing, configure it first (the vendored llama.cpp checkout must exist):

```bash
cmake -S . -B build -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
  '-DCMAKE_CUDA_ARCHITECTURES=89;120' -DSTRATA_GGML_DIR="$PWD/third_party/llama.cpp"
```

The rebuilt executable is `build/strata`. Point a config's `exe` at that executable to use it.
Applying this source patch does not replace `engine/strata` or change `strata-iq4_xs.json`.
If the patch is already present, `git apply --reverse --check` succeeds; do not apply it twice.
If a later upstream version has changed the same code, review any application conflicts instead of forcing them.

To remove these source changes later:

```bash
git apply --reverse --check patches/2026-10-03-dense-layer-placement.patch
git apply --reverse patches/2026-10-03-dense-layer-placement.patch
```

## What is restored

Each GPU uploads per-layer canonical/native dense weights only for its assigned range. Shared/global
weights, including PLE inputs, stay resident. Automatic placement prices layer weights and session memory
before upload. Output head and MTP still run on the final GPU; the two-layer first-stage minimum remains.
The patch adds `dense_placement_test` and updates `docs/MULTI_GPU.md`.

Validated locally on i5-13500, 5070 Ti 16 GB + 4060 8 GB, 64 GB RAM, IQ4_XS and Q2 MTP with all
248,320 draft tokens. With 4060 layers 0–3 and 5070 Ti layers 4–47 plus output head/MTP:
all 2,048 experts of the first four layers fit on the 4060 (4.99 GiB), and the 5070 Ti caches 2,939 experts.
The measured six-request generation rate was 46.74 tok/s versus 42.60 for the preceding two-layer run.
These are small local measurements; the source patch does not apply benchmark settings automatically.
Keep the existing full draft vocabulary at the configured Q2 MTP runtime.

## Saved measurements

`2026-10-03-dense-layer-placement-backup.tar.gz` contains this patch, these instructions, the manifest and both GPU-placement
measurement folders under their original relative paths. It includes raw logs, responses, test configs and
benchmark scripts; it excludes binaries, model files and unrelated workspace changes.
Extract it into a Strata checkout to restore the records and documentation's measurement links:

```bash
tar -xzf /path/to/2026-10-03-dense-layer-placement-backup.tar.gz -C /path/to/Strata
```

Extraction restores saved records; applying the patch remains a separate step.
The first-four-layer test config is
`bench/results/2026-10-03-dense-layer-placement/changed-full-cache-k4-config.json`.
It specifies the rebuilt engine, GPU order `[1, 0]`, `layer_split: 4`, full Q2 draft vocabulary,
and suffix drafts disabled for the comparison. Its absolute model paths are for this PC.
The benchmark scripts obtain prompts from the existing saved full-vocabulary Q2 comparison;
those older MTP records are not duplicated in this backup.

## Packaging verification

The patch passed `git apply --check` with whitespace errors rejected, applied to an isolated copy
of the base source, reproduced all nine current files byte for byte, and reversed back to the base exactly.
