"""Summarize the first-four-layer test against the saved two-layer run."""
import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('summary', OUT / 'summarize.py')
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def main():
    two = s.load('changed-fresh-k2')
    four = s.load('changed-full-cache-k4')
    assert four['status'] == 'complete', four.get('error')
    manifest = json.loads((OUT / 'four-layer-manifest.json').read_text())
    assert manifest['complete_cache_confirmed']
    a, b = s.rate(two), s.rate(four)
    def long(result):
        t = next(r for r in result['runs'] if r['label'] == 'long-prompt')['response']['timings']
        return t['prompt_ms']/1000, t['predicted_per_second']
    x, y = long(two), long(four)
    summary = {'two_layers_tok_s': a, 'four_layers_tok_s': b, 'percent_change': (b/a-1)*100,
               '4060_cached_experts': 2048, '4060_expert_gib': manifest['expert_bytes_first_four']/2**30,
               '5070_cached_experts': 2939, 'production_unchanged': manifest['production_unchanged'],
               'gpu_after': manifest['gpu_after']}
    (OUT/'four-layer-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (OUT/'four-layer-report.md').write_text(f'''# Four complete expert layers on the 4060

Measured 2026-10-03 on the i5-13500, RTX 5070 Ti 16 GB + RTX 4060 8 GB, 64 GB RAM.
Uses the rebuilt engine with dense weights limited to each GPU's assigned layer range.
Physical GPU order: 4060 first, 5070 Ti last. All 248320 Q2 MTP draft vocabulary IDs,
IQ4_XS, 131072 context capacity, int8 KV, 512-token prompt chunks, spec=4, cutoff=.70,
1248 MiB reserve. Suffix drafting is disabled in both compared runs; adaptive expert swaps remain enabled.

## Confirmed placement

| GPU | Main layers | Cached experts | Expert VRAM | Output head / MTP |
|---|---|---:|---:|---|
| RTX 4060 | 0–3 (first four) | All 2048: 512 per layer | 4.99 GiB | Neither |
| RTX 5070 Ti | 4–47 (remaining 44) | 2939 of 22528 | 6.58 GiB | Both |

The first four layers' experts require exactly {manifest['expert_bytes_first_four']:,} bytes.
The engine confirmed all 2048 slots were filled. The 4060 had 1556 MiB free after startup.
Generation proceeds through the 4060's first four layers and then the 5070 Ti finishes with
its remaining layers, output head and MTP. No expert-only helper was enabled.
Prompt processing can temporarily borrow cache slots for buffers; they are refilled afterward.

## Speed

The four-layer test loaded a fresh engine, warmed up for 128 generated tokens, then ran the same
English/Python/Vietnamese prompts twice with up to 512 output tokens each, followed by a separate
9851-token cold prompt with up to 256 output tokens. The two-layer row reuses the immediately preceding
saved test with the same final executable and settings. Short generation is weighted by total output
tokens divided by summed engine decode time; warmup and the long request are excluded.

| 4060 placement | 5070 Ti placement | Short generation | Cold prompt read | Long-request generation |
|---|---|---:|---:|---:|
| Layers 0–1, all 1024 experts cached | Layers 2–47 + output head/MTP, 2771 experts cached | {a:.2f} tok/s | {x[0]:.2f} s | {x[1]:.1f} tok/s |
| Layers 0–3, all 2048 experts cached | Layers 4–47 + output head/MTP, 2939 experts cached | {b:.2f} tok/s | {y[0]:.2f} s | {y[1]:.1f} tok/s |

The measured short-generation change is {(b/a-1)*100:+.1f}%. This is a small local sample;
changing the layer placement and expert residency can change CPU/GPU rounding, text and MTP acceptance.
It does not establish quality equivalence or performance at a full 128K prompt.
The old four-layer pilot before the dense-weight change cached only 631 experts; it is a different setup
and should not be used as the result of this complete-cache test.

## Saved configuration and state

`changed-full-cache-k4-config.json` is the tested configuration, using `build/strata`, `gpu: [1, 0]`
and `layer_split: 4`. It is an experiment configuration, not the active production configuration.
Production config, original engine, rebuilt engine and the full Q2 draft vocabulary were unchanged by hash.
All temporary servers stopped. Final state:

```text
{manifest['gpu_after'].strip()}
```

Raw prompts, responses, timings and logs are saved in `changed-full-cache-k4-results.json` and its
adjacent engine/server logs. `four-layer-manifest.json` records the engine hashes and model expert sizes.
''')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
