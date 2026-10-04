"""Compare the new split with saved measurements using the same evaluation."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'bench/results/2026-10-03-iq4-xs-gpu-placement'
DENSE = ROOT / 'bench/results/2026-10-03-dense-layer-placement'


def aggregate(runs):
    timings = [r['response']['timings'] for r in runs]
    tokens = sum(t['predicted_n'] for t in timings)
    ms = sum(t['predicted_ms'] for t in timings)
    offered = sum(t['draft_n'] for t in timings)
    accepted = sum(t['draft_n_accepted'] for t in timings)
    hits = [re.search(r'hit rate: .*?\((\d+) hits / (\d+) lookups\)', r['engine_log']) for r in runs]
    hits = [m for m in hits if m]
    return {'requests': len(runs), 'generated_tokens': tokens, 'decode_ms': ms,
            'decode_tok_s': tokens * 1000 / ms, 'drafts_accepted': accepted,
            'drafts_offered': offered, 'draft_acceptance': accepted / offered if offered else None,
            'expert_hit_rate': sum(int(m[1]) for m in hits) / sum(int(m[2]) for m in hits) if hits else None}


def main():
    result = json.loads((OUT / 'patched-k46-results.json').read_text())
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert result['status'] == manifest['status'] == 'complete'
    assert manifest['files_unchanged'] and manifest['identical_requests_to_saved_k4']
    log = result['startup_log']
    assert 'CUDA1 runs layers 46-47, expert cache 1024 slots (3.00 GiB), 1024 of its 1024 profiled pairs' in log
    assert 'CUDA1 keeps its own prompt buffers' in log and 'no loan' in log
    assert 'CUDA1 holds its weights, session [46, 48) and the head' in log
    assert 'draft head over 248320 tokens' in log
    sources = [
        (OUT, 'patched-k46', 'NEW: 5070 0–45; 4060 46–47 + head/MTP, full cache', 'patched'),
        (DENSE, 'changed-full-cache-k4', '4060 0–3 full cache; 5070 4–47 + head/MTP', 'patched'),
        (DENSE, 'changed-fresh-k2', '4060 0–1 full cache; 5070 2–47 + head/MTP', 'patched'),
        (DENSE, 'original-fresh-k2', '4060 0–1; 5070 2–47 + head/MTP, fresh', 'original'),
        (OLD, '5070-alone-check', '5070 alone, recheck', 'original'),
        (OLD, '5070-head-mtp-4060-experts', '5070 all layers + head/MTP; 4060 expert helper', 'original'),
        (OLD, '4060-first-5070-head-mtp', '4060 0–1; 5070 2–47 + head/MTP, earlier', 'original'),
        (OLD, '4060-first-four-layers', '4060 0–3; 5070 4–47 + head/MTP, pilot', 'original'),
        (OLD, '5070-4060-auto', '5070 0–46; 4060 47 + head/MTP, auto', 'original'),
        (OLD, '5070-4060-k44', '5070 0–43; 4060 44–47 + head/MTP, pilot', 'original'),
        (OLD, '5070-4060-tuned-safe', '5070 0–46; 4060 47 + head/MTP, tuned 1152 MiB/256 prefill', 'original'),
        (OLD, '5070-4060-tuned', '5070 0–46; 4060 47 + head/MTP, tuned 1024 MiB/256 prefill', 'original'),
        (OLD, '5070-alone', '5070 alone, initial (cold prompt did not reproduce)', 'original'),
        (DENSE, 'changed-full-cache-k2', '4060 0–1 full cache; 5070 2–47 + head/MTP (compilation overlapped)', 'patched'),
    ]
    expected = {r['label']: r['request'] for r in result['runs']}
    summary = {'full_4060_cache_confirmed': True, '4060_prompt_cache_loan': False,
               'manifest': manifest, 'comparisons': {}}
    for directory, name, label, engine in sources:
        data = json.loads((directory / f'{name}-results.json').read_text())
        item = {'label': label, 'engine': engine, 'source': str(directory / f'{name}-results.json'),
                'status': data['status'], 'pilot': data.get('pilot', False)}
        if data['status'] == 'complete':
            assert all(r['request'] == expected[r['label']] for r in data['runs']), name
            short = [r for r in data['runs'] if r['label'] != 'long-prompt']
            item['short'] = aggregate(short)
            item['by_language'] = {lang: aggregate([r for r in short if r['label'].startswith(lang)])
                                   for lang in ('english', 'code', 'vietnamese')}
            long = next((r for r in data['runs'] if r['label'] == 'long-prompt'), None)
            item['long'] = long['response']['timings'] if long else None
            item['startup_residency'] = [line for line in data['startup_log'].splitlines()
                                         if 'expert cache ' in line or 'dense weights for layers' in line]
        else:
            item['error'] = data.get('error')
        summary['comparisons'][name] = item
    new = summary['comparisons']['patched-k46']
    summary['short_speed_delta_percent'] = {
        name: (new['short']['decode_tok_s'] / item['short']['decode_tok_s'] - 1) * 100
        for name, item in summary['comparisons'].items() if item['status'] == 'complete' and name != 'patched-k46'}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    lines = [
        '# Layers 0–45 on the 5070 Ti; layers 46–47, head and MTP on the 4060', '',
        'Measured 2026-10-04 on the Intel i5-13500, RTX 5070 Ti 16 GB, RTX 4060 8 GB and 64 GB RAM, Linux. '
        'The patched executable matches the SHA-256 used for the saved fresh dense-placement and four-layer tests.', '',
        '## Confirmed placement', '',
        '| GPU | Main layers | Cached main-model experts | Expert VRAM | Output head / Q2 MTP |',
        '|---|---|---:|---:|---|',
        '| RTX 5070 Ti | 0–45 | 3560 of 23552 | 7.90 GiB | Neither |',
        '| RTX 4060 | 46–47 | All 1024 of 1024 | 3.00 GiB | Both |', '',
        'The 4060 loads 84.47 MiB canonical and 145.43 MiB native dense weights for its layer range, including retained shared weights. '
        'The Q2 draft layer uses 949 MiB and its 248320-token head uses 497.3 MiB. '
        'The 4060 keeps its own 0.41 GiB prompt buffers, so its complete expert cache is not borrowed for prompt processing. '
        'The 5070 Ti prompt path borrows 185 of its own slots and refills them. '
        'The startup logs confirm the requested ranges, head, full draft vocabulary and complete 4060 expert cache.', '',
        '## Matched evaluation', '',
        'IQ4_XS, installed Q2 MTP, all 248320 draft token IDs, 131072 context capacity, int8 KV, '
        '512-token prefill chunks, spec=4, cutoff=.70, 1248 MiB reserve, PCIe fraction=.20. '
        'Greedy sampling, seed 1234, thinking disabled, suffix drafting disabled; adaptive caching and automatic MTP policy enabled. '
        'The original placement benchmark runner was imported and reused.', '',
        'One fresh engine, one 128-token warmup, the saved English/Python/Vietnamese prompts twice at up to 512 output tokens each, '
        'then the identical cold 9851-token maintenance-log prompt with up to 256 output tokens. '
        'The request objects match the saved patched K=4 evaluation exactly. '
        'Short decode is total output tokens divided by summed engine decode time, excluding warmup and the long request. '
        'Acceptance is total accepted drafts divided by total offered drafts.', '',
        '## New results by language', '',
        '| Output | Generated tokens | Decode tok/s | MTP acceptance | Expert cache hit rate |',
        '|---|---:|---:|---:|---:|']
    for label, metrics in [*new['by_language'].items(), ('Combined', new['short'])]:
        lines.append(f'| {label} | {metrics["generated_tokens"]} | {metrics["decode_tok_s"]:.2f} | '
                     f'{metrics["draft_acceptance"] * 100:.1f}% | {metrics["expert_hit_rate"] * 100:.1f}% |')
    lines += ['', '## Saved combinations', '',
              'Only the NEW row was run for this request. All other rows reuse raw results measured 2026-10-03. '
              'Pilot rows have one repeat per language and no long prompt. Tuned rows change reserve and prefill. '
              'The initial single-GPU cold-prompt result did not reproduce; the recheck is the useful comparison. '
              'The compilation-overlapped row is retained for history; use the fresh patched K=2 row for comparison.', '',
              '| Placement | Engine | Short tok/s | Cold prompt seconds | Cold prompt tok/s | Long decode tok/s |',
              '|---|---|---:|---:|---:|---:|']
    for item in summary['comparisons'].values():
        if item['status'] != 'complete':
            lines.append(f'| {item["label"]} | {item["engine"]} | Failed: verification-graph OOM | — | — | — |')
            continue
        long = item['long']
        cells = [f'{long["prompt_ms"] / 1000:.2f}', f'{long["prompt_per_second"]:.1f}',
                 f'{long["predicted_per_second"]:.1f}'] if long else ['—'] * 3
        lines.append(f'| {item["label"]} | {item["engine"]} | {item["short"]["decode_tok_s"]:.2f} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Interpretation and state', '']
    for name in ('changed-full-cache-k4', 'changed-fresh-k2', '5070-alone-check', '5070-head-mtp-4060-experts'):
        item = summary['comparisons'][name]
        lines.append(f'The new short decode rate differs by {summary["short_speed_delta_percent"][name]:+.1f}% from '
                     f'{item["label"]} ({item["short"]["decode_tok_s"]:.2f} tok/s).')
    lines += ['', 'This small speed sample does not establish quality equivalence or full 128K-context performance. '
              'Expert residency and CPU/GPU rounding can change generated text and MTP acceptance. '
              'The long-request automatic draft policy can differ across placements. '
              'Smoke/parity checks and the separate Q2-versus-Q4 MTP study are not pooled with these speed results.', '',
              'Production configuration, original engine, patched engine, expert profile, full draft vocabulary and the reused runner '
              'were unchanged by hash. The temporary localhost server stopped; production remains stopped.', '',
              '```text', manifest['gpu_after'].strip(), '```', '',
              'Raw requests, responses and timings: [patched-k46-results.json](patched-k46-results.json). '
              'Test configuration: [patched-k46-config.json](patched-k46-config.json). '
              'Hashes and state: [manifest.json](manifest.json). '
              'All language-level comparisons: [summary.json](summary.json).']
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'new': new['short'], 'long': new['long'],
                      'delta_percent': summary['short_speed_delta_percent'],
                      'files_unchanged': manifest['files_unchanged']}, indent=2))


if __name__ == '__main__':
    main()
