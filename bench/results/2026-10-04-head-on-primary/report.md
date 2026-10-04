# Separate head/MTP placement on the first GPU

Measured 2026-10-04 on an i5-13500, 64 GB RAM, RTX 5070 Ti 16 GB + RTX 4060 8 GB, Linux. The selected boundary is **n=44**: layers 0–43 and the output head/Q2 MTP run on the 5070 Ti; layers 44–47 run on the 4060 with all 2048 experts resident for decode.

The five-layer suffix 43–47 held only 2116 of 2560 experts (5.23 GiB). The four-layer suffix held all 2048 experts (5.22 GiB). These are runtime allocations with the installed IQ4_XS model, 131072 context capacity, int8 KV, prefill 512, spec 4, 1248 MiB reserve and the full 248320-token draft vocabulary. Prompt buffers borrow some expert slots during prefill; the engine refills them before decode. Other settings, model variants or VRAM use can change the boundary.

## Engine change

`--head-device 0` with an explicit multi-GPU split appends an empty transformer range [48,48) on the first GPU. It receives the completed residual after layer 47 and runs the output head/MTP. The 4060 stage hands off even though it reaches the last transformer layer. The head-only stage has its own session, verifier and prompt buffers, but no transformer expert cache. Its pending allocations are reserved before sizing the first GPU cache. This path makes two hand-offs per verification window. The option currently requires an explicit split; automatic boundary selection does not search this placement.

```json
{"gpu": [0, 1], "layer_split": 44, "args": ["--head-device", "0"]}
```

This is a fragment to add to the existing configuration, retaining its other arguments. The active production configuration and engine were not replaced.

## Validation

Three fixed-residency English, Python and Vietnamese requests produced identical response choices before and after the patch for the existing contiguous split, and after moving the head to CUDA0. Adaptation was disabled and eight experts per layer were used to isolate placement correctness from different GPU/CPU expert rounding. Full-profile capacity checks then verified four final layers fully cached and five partially cached. The final rebuild adds stricter empty-stage range validation and clearer diagnostics; its separate conversation-state check also produced identical responses for continuation and branching, with 15 prompt tokens reused in each follow-up. [Conversation-state validation](checkpoint-validation.json).

The split scheduling test passed 14 cases, the existing dense ownership/accounting test passed, and `git diff --check` passed. These checks do not establish token-by-token latency consistency or general performance across workloads.

[GPU validation](validation.json), [final build record](build-record.json), [engine patch relative to the source before this task](head-on-primary.patch), [saved-task evaluation](../../../evaluations/2026-10-04-head-on-primary/REPORT.md).
