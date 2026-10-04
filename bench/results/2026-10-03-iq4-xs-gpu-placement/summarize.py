"""Summarize measured timings; no model execution or configuration changes."""
from pathlib import Path
import json
import re

OUT = Path(__file__).resolve().parent
NAMES = ['5070-alone', '5070-4060-auto', '5070-4060-k44', '5070-4060-tuned-safe', '5070-4060-tuned']
LABELS = {'5070-alone': '5070 Ti alone', '5070-4060-auto': '5070 Ti + 4060, auto (K=47)',
          '5070-4060-k44': '5070 Ti + 4060, four layers on 4060 (K=44, pilot)',
          '5070-4060-tuned': '5070 Ti + 4060, 1024 MiB reserve (failed)',
          '5070-4060-tuned-safe': '5070 Ti + 4060, 1152 MiB reserve (tuned)'}


def summarize(runs):
    ts = [r['response']['timings'] for r in runs]
    offered = sum(t['draft_n'] for t in ts)
    accepted = sum(t['draft_n_accepted'] for t in ts)
    hits = [re.search(r'hit rate: .*?\((\d+) hits / (\d+) lookups\)', r['engine_log']) for r in runs]
    hits = [m for m in hits if m]
    return {'requests': len(ts), 'generated_tokens': sum(t['predicted_n'] for t in ts),
            'decode_ms': sum(t['predicted_ms'] for t in ts),
            'decode_tok_s': sum(t['predicted_n'] for t in ts) * 1000 / sum(t['predicted_ms'] for t in ts),
            'drafts_accepted': accepted, 'drafts_offered': offered,
            'draft_acceptance': accepted / offered if offered else None,
            'expert_hit_rate': sum(int(m[1]) for m in hits) / sum(int(m[2]) for m in hits) if hits else None}


summary = {}
for name in NAMES:
    path = OUT / f'{name}-results.json'
    if not path.exists():
        continue
    d = json.loads(path.read_text())
    item = {'status': d.get('status', 'incomplete'), 'pilot': d['pilot'], 'label': LABELS[name]}
    if item['status'] == 'complete':
        short = [r for r in d['runs'] if r['label'] != 'long-prompt']
        item['short'] = summarize(short)
        item['by_language'] = {lang: summarize([r for r in short if r['label'].startswith(lang)])
                               for lang in ('english', 'code', 'vietnamese')}
        long = next((r for r in d['runs'] if r['label'] == 'long-prompt'), None)
        item['long'] = long['response']['timings'] if long else None
        primary = re.search(r'expert cache (\d+) slots, ([\d.]+) GiB', d['startup_log'])
        secondary = re.search(r'CUDA1 runs layers (\d+)-(\d+), expert cache (\d+) slots \(([\d.]+) GiB\)', d['startup_log'])
        item['cache'] = {'primary_slots': int(primary[1]), 'primary_gib': float(primary[2]),
                         'secondary_slots': int(secondary[3]) if secondary else 0,
                         'secondary_gib': float(secondary[4]) if secondary else 0}
    else:
        item['error'] = d.get('error', 'No completed result')
        log_path = OUT / f'{name}-engine.log'
        if log_path.exists() and 'verify: instantiate: out of memory' in log_path.read_text():
            item['error'] = 'Out of memory while instantiating verification graphs'
    summary[name] = item
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

lines = ['# IQ4_XS: RTX 5070 Ti alone versus RTX 5070 Ti + RTX 4060', '',
         'Measured 2026-10-03 on the Intel i5-13500, 64 GB RAM, RTX 5070 Ti 16 GB and RTX 4060 8 GB, Linux. '
         'The 5070 Ti has a x16 link; the 4060 has a PCIe Gen 3 x4 link. Engine 0.1.34.', '',
         'The single GPU and the default dual-GPU setup use the saved production settings: IQ4_XS, Q2 MTP, '
         'all 248,320 draft token IDs, 131,072 maximum context, int8 KV, spec=4, spec-min-p=0.70, '
         'prefill=512, VRAM reserve=1248 MiB, PCIe fraction=0.20. Suffix lookup was disabled in every test '
         'to isolate MTP. Thinking was disabled and sampling was greedy. Adaptive expert caching and '
         'the engine\'s automatic draft policy stayed enabled.', '',
         'Each full test loaded a fresh engine, warmed up with 128 generated tokens, then ran the same English, '
         'Python and Vietnamese prompts twice, capped at 512 output tokens. Some prose answers ended early. '
         'Short-request decode speed is total generated tokens divided by total engine decode time; '
         'warmup and the long prompt are excluded. Each full test also ran one identical cold '
         '9,851-token prompt, capped at 256 output tokens. The K=44 pilot used one repeat per language and '
         'did not run the long prompt.', '',
         'The completed tuning attempt changes prefill to 256 and VRAM reserve to 1152 MiB, so it measures '
         'a tuning attempt as well as GPU placement. Everything else stays matched. Its automatic split '
         'still assigns just the final layer to the 4060. An earlier attempt with a 1024 MiB reserve '
         'loaded 93 experts (0.27 GiB) on the 4060 but ran out of memory while instantiating verification '
         'graphs in its warmup; it has no valid speed result. No source or engine changes were made.', '',
         '## Measurements', '',
         '| Setup | Short decode tok/s | Cold 9,851-token prompt tok/s | Prompt seconds | 5070 cache GiB | 4060 cache GiB (slots) |',
         '|---|---:|---:|---:|---:|---:|']
for name, d in summary.items():
    if d['status'] != 'complete':
        lines.append(f'| {d["label"]} | Failed: {d["error"]} | — | — | — | — |')
        continue
    long, cache = d['long'], d['cache']
    pp = f'{long["prompt_per_second"]:.1f}' if long else '—'
    sec = f'{long["prompt_ms"] / 1000:.2f}' if long else '—'
    lines.append(f'| {d["label"]} | {d["short"]["decode_tok_s"]:.2f} | {pp} | {sec} | '
                 f'{cache["primary_gib"]:.2f} | {cache["secondary_gib"]:.2f} ({cache["secondary_slots"]}) |')
lines += ['', '## Short-request results by output', '',
          '| Setup | English tok/s | Code tok/s | Vietnamese tok/s | MTP acceptance | Expert cache hit rate |',
          '|---|---:|---:|---:|---:|---:|']
for d in summary.values():
    if d['status'] != 'complete':
        continue
    lang, short = d['by_language'], d['short']
    lines.append(f'| {d["label"]} | {lang["english"]["decode_tok_s"]:.2f} | '
                 f'{lang["code"]["decode_tok_s"]:.2f} | {lang["vietnamese"]["decode_tok_s"]:.2f} | '
                 f'{100 * short["draft_acceptance"]:.1f}% | {100 * short["expert_hit_rate"]:.1f}% |')
lines += ['', '## Interpretation and limits', '',
          'Moving the head and MTP to the 4060 increases the 5070 Ti\'s expert cache, but the default '
          'two-card setup was slightly slower on the short requests and much slower on this long prompt. '
          'Two repeats per short prompt are not enough to establish a reliable small decode-speed advantage. '
          'The long prompt is one repetitive maintenance-log workload, measured once per setup; '
          'it does not establish performance at the full 128K context limit.', '',
          'For these measured workloads, use the 5070 Ti alone: it read the long prompt 2.40 times '
          'as fast as the default split and 4.00 times as fast as the tuned split. The tuned split '
          'increased the 4060 cache from 19 to 51 experts but improved short decode by only 1.84% '
          'over the single GPU in this small sample, while taking longer to read the prompt. '
          'The lower reserve was tested only with suffix drafts disabled; it is not validated '
          'for the production setup\'s larger suffix-draft windows or other workloads.', '',
          'The engine chooses its draft policy automatically: the single-GPU long-prompt response offered '
          'zero MTP drafts, whereas the default two-GPU long-prompt response offered 178 and accepted 143. '
          'Those long-prompt decode rates therefore reflect each setup\'s automatic behavior, rather than '
          'equal numbers of draft rounds. They are excluded from the short-request aggregate.', '',
          'Assigning four layers to the 4060 left its cache at 18 slots and slowed the pilot. '
          'A split-stage cache stores experts only for that stage\'s layers; the 4060 cannot also be '
          'a general helper cache for the other stage with the current engine and these two cards.', '',
          'Generated text can differ between placements because CPU and GPU expert calculations round '
          'differently and adaptive caching changes residency. This is a speed comparison, not a quality evaluation.', '',
          '## Saved state', '',
          'The production JSON, installed Q2 runtime vocabulary, and engine hashes are unchanged '
          '(see `initial-manifest.json` and `manifest.json`). Temporary benchmark servers were stopped. '
          'The user had stopped the production server and requested no restoration; it remains stopped. '
          'Per-request responses and timings, startup logs, temporary configs, the runner, and '
          '`summary.json` are saved alongside this report.']
if (OUT / 'helper-report.md').exists():
    lines[2:2] = ['**Updated:** the requested expert-only helper configuration and a fresh single-GPU '
                   'recheck are in [helper-report.md](helper-report.md). The recheck did not reproduce '
                   'the initial 14.61-second prompt result. The recommendation below is superseded '
                   'by that corrected comparison; the original measurements are retained here.', '']
(OUT / 'report.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(summary, indent=2))
