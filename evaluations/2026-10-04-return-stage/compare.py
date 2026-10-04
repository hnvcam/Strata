"""Verify saved external inputs and compare timings with completed prior runs."""
import importlib.util
import json
from pathlib import Path
import socket
import re

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / '2026-10-04-helper-vs-four-layers'
spec = importlib.util.spec_from_file_location('saved_compare', PREVIOUS / 'compare.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
c.OUT = OUT


def load(path):
    return json.loads(path.read_text())


def main():
    manifest = load(OUT / 'run-manifest.json')
    previous = load(PREVIOUS / 'run-manifest.json')
    assert manifest['status'] == 'complete' and manifest['protected_files_unchanged']
    runs = [r for phase in manifest['phases'] for r in phase['runs']]
    verification = {'source_input_hashes_identical': manifest['saved_input_hashes'] == previous['saved_input_hashes'],
                    'matched_requests': [], 'selected_n': manifest['selected_n']}
    assert verification['source_input_hashes_identical']
    for run in runs:
        new = load(OUT / (run['stem'] + '-request.json'))
        for name in ('helper', 'four-layers'):
            old = load(PREVIOUS / (f'{name}-{run["task"]}-{run["stage"]}-request.json'))
            assert {k: v for k, v in new.items() if k != 'messages'} == {k: v for k, v in old.items() if k != 'messages'}
            assert [m for m in new['messages'] if m['role'] != 'assistant'] == [m for m in old['messages'] if m['role'] != 'assistant']
        verification['matched_requests'].append(run['stem'])
    assert len(runs) == 8
    assert all(r['finish_reason'] == 'stop' for r in runs if r['stage'] not in (2, '2-continuation-1'))
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 1112))
    verification['benchmark_port_closed'] = True
    baseline = load(PREVIOUS / 'summary.json')['setups']
    new = {task: c.aggregate([r for r in runs if r['task'] == task]) for task in ('marketing', 'python')}
    new['all'] = c.aggregate(runs)
    new['marketing_word_count'] = len((OUT / 'return-marketing-1.md').read_text().split())
    new['load_s'] = {p['phase']: p['load_s'] for p in manifest['phases']}
    new['phase_elapsed_s'] = {p['phase']: p['elapsed_s_including_stop'] for p in manifest['phases']}
    summary = {'setups': {**baseline, 'return': new},
               'note': 'New return-stage build versus saved prior build; existing two-stage matched-residency output regression passed. Independent runs, differing generated histories/output lengths. Helper hit rate excludes remote helper work and is not combined-GPU coverage.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (OUT / 'comparison-verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    lines = ['# Return-stage task timings', '', summary['note'], '',
             '| Setup | Task | Output tokens | Read tokens | Reused tokens | Prefill s | Prefill tok/s | Decode s | Decode tok/s | Request wall s |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, setup in summary['setups'].items():
        for task in ('marketing', 'python'):
            t = setup[task]
            lines.append(f'| {name} | {task} | {t["output_tokens"]} | {t["prompt_read_tokens"]} | {t["prompt_reused_tokens"]} | {t["prefill_s"]:.2f} | {t["prefill_tok_s"]:.2f} | {t["decode_s"]:.2f} | {t["decode_tok_s"]:.2f} | {t["wall_s"]:.2f} |')
    lines += ['', '| Return request | Read / reused | Prefill tok/s | Prefill s | Output | Decode tok/s | Decode s | Wall s | Finish |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in runs:
        t = r['timings']
        lines.append(f'| {r["stem"]} | {t["prompt_n"]} / {t["cache_n"]} | {t["prompt_per_second"]:.2f} | {t["prompt_ms"]/1000:.2f} | {t["predicted_n"]} | {t["predicted_per_second"]:.2f} | {t["predicted_ms"]/1000:.2f} | {r["wall_s"]:.2f} | {r["finish_reason"]} |')
    lines += ['', '| Phase | Load s | Elapsed including startup/shutdown s |', '|---|---:|---:|']
    for p in manifest['phases']:
        lines.append(f'| {p["phase"]} | {p["load_s"]:.2f} | {p["elapsed_s_including_stop"]:.2f} |')
    (OUT / 'timing.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(new, indent=2))


if __name__ == '__main__':
    main()
