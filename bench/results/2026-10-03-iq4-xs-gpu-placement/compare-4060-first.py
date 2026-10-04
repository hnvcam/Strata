"""Summarize the final 4060-first split alongside the saved matched measurements."""
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
names = ['5070-alone-check', '5070-head-mtp-4060-experts',
         '4060-first-5070-head-mtp', '4060-first-four-layers']
labels = ['5070 Ti alone, recheck', '5070 Ti + 4060 expert-only helper',
          '4060 first two layers, 5070 Ti finishes', '4060 first four layers, pilot']
summary = {}


def rate(runs):
    ts = [r['response']['timings'] for r in runs]
    return sum(t['predicted_n'] for t in ts) * 1000 / sum(t['predicted_ms'] for t in ts)


for name, label in zip(names, labels):
    data = json.loads((OUT / f'{name}-results.json').read_text())
    assert data['status'] == 'complete', name
    short = [r for r in data['runs'] if r['label'] != 'long-prompt']
    long = next((r['response']['timings'] for r in data['runs'] if r['label'] == 'long-prompt'), None)
    primary = re.search(r'expert cache (\d+) slots, ([\d.]+) GiB', data['startup_log'])
    later = re.search(r'CUDA1 runs layers (\d+)-(\d+), expert cache (\d+) slots \(([\d.]+) GiB\)', data['startup_log'])
    item = {'label': label, 'pilot': data['pilot'], 'requests': len(short),
            'decode_tok_s': rate(short),
            'first_repeat_decode_tok_s': rate([r for r in short if r['label'].endswith('-1')]),
            'long_prompt_s': long['prompt_ms'] / 1000 if long else None,
            'long_prompt_tok_s': long['prompt_per_second'] if long else None,
            'long_decode_tok_s': long['predicted_per_second'] if long else None,
            'by_language': {lang: rate([r for r in short if r['label'].startswith(lang)])
                            for lang in ['english', 'code', 'vietnamese']},
            'primary_cache_slots': int(primary[1]), 'primary_cache_gib': float(primary[2]),
            'later_cache_slots': int(later[3]) if later else None,
            'later_cache_gib': float(later[4]) if later else None}
    summary[name] = item

single, helper, first, four = [summary[n] for n in names]
summary['comparison'] = {'first_vs_single_decode_gain': first['decode_tok_s'] / single['decode_tok_s'] - 1,
                         'first_vs_helper_decode_gain': first['decode_tok_s'] / helper['decode_tok_s'] - 1,
                         'first_vs_helper_prompt_time': first['long_prompt_s'] / helper['long_prompt_s'],
                         'four_vs_two_first_repeat_decode_gain': four['first_repeat_decode_tok_s'] / first['first_repeat_decode_tok_s'] - 1}
(OUT / '4060-first-summary.json').write_text(json.dumps(summary, indent=2) + '\n')

lines = ['# Final test: RTX 4060 first, RTX 5070 Ti finishes with the head and MTP', '',
         'Measured 2026-10-03 on the Intel i5-13500, 64 GB RAM, RTX 4060 8 GB (PCIe Gen 3 x4) and '
         'RTX 5070 Ti 16 GB (Gen 4 x16), Linux, original engine 0.1.34.', '',
         '## Requested flow', '',
         'The temporary config sets `gpu: [1, 0]` and `layer_split: auto`. CUDA0 is therefore the physical '
         'RTX 4060; CUDA1 is the physical RTX 5070 Ti. The automatic split selected K=2:', '',
         '- RTX 4060: first two layers (engine layer numbers 0–1).',
         '- RTX 5070 Ti: remaining 46 layers (2–47), output head and Q2 MTP drafter.', '',
         'The engine log confirms one stage handoff per verification window from the 4060 to the 5070 Ti. '
         'There is no expert-only helper enabled, and no per-layer expert-result round trip from the '
         '5070 Ti to the 4060 and back. Each subsequent generation window starts again on the first stage; '
         'this is not a one-time handoff for the entire conversation.', '',
         '## What fits', '',
         'With the current dense-weight copies, 128K context capacity and 1248 MiB reserve, the 4060 caches '
         '718 experts (1.56 GiB) for its two layers. Those layers have 1024 experts in total; all their '
         'expert weights would take 2.22 GiB. Thus the selected layer range fits with an expert cache and '
         'RAM fallback, not with every expert permanently resident. The 5070 Ti caches 2690 experts '
         '(6.06 GiB) for its remaining layers.', '',
         'The existing engine requires the first stage to contain at least two layers. Its automatic '
         'placement considers every supported two-card split from K=2 through K=47 using estimated '
         'compute and cache-miss costs; it selected the smallest supported early range for this pair. '
         'A four-layer pilot checks whether that larger early range helps in practice. It caches only '
         '631 of the first four layers\' 2048 experts (1.55 GiB); the larger third-layer expert blobs '
         'use more bytes per cached expert.', '',
         '## Matched settings and method', '',
         'IQ4_XS, installed Q2 MTP runtime, all 248320 draft IDs, maximum context 131072, int8 KV, spec=4, '
         'spec-min-p=.70, prefill=512, VRAM reserve=1248 MiB and PCIe fraction=.20. Original engine and '
         'source unchanged. Suffix drafting disabled in every compared test, thinking disabled, greedy '
         'sampling. Cache adaptation and automatic MTP policy remain enabled. No reserve reduction or '
         'smaller prefill buffers are used.', '',
         'Each full test loads a fresh engine, warms up for 128 output tokens, then runs the same '
         'English/Python/Vietnamese prompts twice with up to 512 output tokens, followed by one cold '
         '9851-token maintenance-log prompt and 256 output tokens. Decode speed is total generated '
         'tokens divided by summed engine decode time; warmup and the long request are excluded. '
         'The four-layer pilot has one repeat per language and no long request. The single-GPU and '
         'helper rows reuse the previously saved matched results; they were not rerun in this final test.', '',
         '## Results', '',
         '| Setup | Short decode tok/s | Cold prompt seconds | Cold prompt tok/s | Long-request decode tok/s |',
         '|---|---:|---:|---:|---:|']
for name in names:
    s = summary[name]
    pp_s = f'{s["long_prompt_s"]:.2f}' if s['long_prompt_s'] is not None else '—'
    pp_rate = f'{s["long_prompt_tok_s"]:.1f}' if s['long_prompt_tok_s'] is not None else '—'
    long_rate = f'{s["long_decode_tok_s"]:.1f}' if s['long_decode_tok_s'] is not None else '—'
    lines.append(f'| {s["label"]} | {s["decode_tok_s"]:.2f} | {pp_s} | {pp_rate} | {long_rate} |')
lines += ['', '| Setup | English tok/s | Code tok/s | Vietnamese tok/s | First repeat combined tok/s |',
          '|---|---:|---:|---:|---:|']
for name in names:
    s = summary[name]
    r = s['by_language']
    lines.append(f'| {s["label"]} | {r["english"]:.2f} | {r["code"]:.2f} | '
                 f'{r["vietnamese"]:.2f} | {s["first_repeat_decode_tok_s"]:.2f} |')
lines += ['', '## Interpretation and limits', '',
          f'The requested first-two-layer split averaged {first["decode_tok_s"]:.2f} tok/s: '
          f'{100*summary["comparison"]["first_vs_single_decode_gain"]:.1f}% above the rechecked single GPU '
          f'and {100*summary["comparison"]["first_vs_helper_decode_gain"]:.1f}% above the expert-only helper. '
          'The small difference against the helper is not enough to establish a reliable overall winner. '
          'The helper was faster on the separate long request\'s decode phase; the early split read '
          'that prompt faster in this measurement.', '',
          f'In the matched first repeat, four early layers averaged {four["first_repeat_decode_tok_s"]:.2f} '
          f'tok/s versus {first["first_repeat_decode_tok_s"]:.2f} for two early layers. '
          'The larger early range did not improve that pilot.', '',
          'The initial single-GPU cold-prompt result elsewhere in this directory (14.61 seconds) did '
          'not reproduce in its recheck (41.45 seconds). Cold-prompt timing is therefore uncertain; '
          'one long request per setup does not establish steady prompt throughput. This is a small '
          'speed test, not a quality evaluation or a benchmark at a full 128K prompt. Expert residency '
          'and CPU/GPU rounding can change generated text and MTP acceptance between placements.', '',
          'For the requested flow, the tested auto split is a reasonable choice: first two layers '
          'on the 4060, all remaining layers and the head/MTP on the 5070 Ti. Avoid assuming that '
          'more early layers or eliminating the helper\'s round trips guarantees higher throughput.', '',
          '## Saved state', '',
          'The production config, original engine and installed full draft vocabulary are unchanged. '
          'Temporary servers were stopped and both GPUs released. The tested config is '
          '`4060-first-5070-head-mtp-config.json`; raw responses, logs, `4060-first-summary.json`, '
          'the runner and `4060-first-manifest.json` are saved here.']
(OUT / '4060-first-report.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(summary['comparison'], indent=2))
