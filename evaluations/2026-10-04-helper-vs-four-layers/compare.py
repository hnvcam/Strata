"""Verify matched external inputs and summarize observed task timing."""
import hashlib
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
SAVED = OUT.parent / '2026-10-03-iq3-s-vs-iq4-xs'


def load(name):
    return json.loads((OUT / name).read_text())


def aggregate(runs):
    ts = [r['timings'] for r in runs]
    tokens = sum(t['predicted_n'] for t in ts)
    decode_ms = sum(t['predicted_ms'] for t in ts)
    prompt_tokens = sum(t['prompt_n'] for t in ts)
    prompt_ms = sum(t['prompt_ms'] for t in ts)
    drafts = sum(t['draft_n'] for t in ts)
    accepted = sum(t['draft_n_accepted'] for t in ts)
    hits = []
    helper_entries = 0
    for run in runs:
        directory = Path(run['archived_directory']) if run.get('superseded') else OUT
        log = (directory / f'{run["stem"]}-engine.log').read_text()
        hits.extend(re.findall(r'hit rate: .*?\((\d+) hits / (\d+) lookups\)', log))
        helper_entries += sum(int(n) for n in re.findall(r'CUDA1: (\d+) expert entries', log))
    return {'requests': len(runs), 'output_tokens': tokens, 'decode_s': decode_ms / 1000,
            'decode_tok_s': tokens * 1000 / decode_ms,
            'prompt_read_tokens': prompt_tokens, 'prompt_reused_tokens': sum(t['cache_n'] for t in ts),
            'prefill_s': prompt_ms / 1000,
            'prefill_tok_s': prompt_tokens * 1000 / prompt_ms if prompt_ms else None,
            'wall_s': sum(r['wall_s'] for r in runs),
            'drafts_offered': drafts, 'drafts_accepted': accepted,
            'draft_acceptance': accepted / drafts if drafts else None,
            'expert_hit_rate': sum(int(h) for h, _ in hits) / sum(int(n) for _, n in hits) if hits else None,
            'helper_expert_entries': helper_entries}


def main():
    manifest = load('run-manifest.json')
    assert manifest['status'] == 'complete' and manifest['protected_files_unchanged']
    verification = {'same_patched_executable': True, 'identical_saved_inputs': {},
                    'matched_external_messages_and_settings': [], 'complete_final_deliverables_after_continuations': True}
    for name, digest in manifest['saved_input_hashes'].items():
        local = hashlib.sha256((OUT / name).read_bytes()).hexdigest()
        original = hashlib.sha256((SAVED / name).read_bytes()).hexdigest()
        assert local == original == digest, name
        verification['identical_saved_inputs'][name] = digest
    summary = {'setups': {}, 'runs': manifest['phases']}
    for setup in ('helper', 'four-layers'):
        all_phases = [p for p in manifest['phases'] if p['setup'] == setup]
        phases = [p for p in all_phases if any(not r.get('superseded') for r in p['runs'])]
        assert len(phases) == 4 and all(p['status'] in ('complete', 'complete_with_truncation') for p in phases)
        runs = [r for p in phases for r in p['runs'] if not r.get('superseded')]
        assert len(runs) == 8
        assert all(r['finish_reason'] == 'stop' or (r['task'] == 'python' and r['stage'] in (2, '2-continuation-1')) for r in runs)
        for run in runs:
            t = run['timings']
            run['prefill_s'] = t['prompt_ms'] / 1000
            run['decode_s'] = t['predicted_ms'] / 1000
        summary['setups'][setup] = {
            'marketing': aggregate([r for r in runs if r['task'] == 'marketing']),
            'python': aggregate([r for r in runs if r['task'] == 'python']),
            'all': aggregate(runs),
            'load_s': {p['phase']: p['load_s'] for p in phases},
            'phase_elapsed_s': {p['phase']: p['elapsed_s_including_stop'] for p in phases},
            'all_phase_elapsed_s_including_superseded': sum(p['elapsed_s_including_stop'] for p in all_phases),
            'superseded': aggregate([r for p in all_phases for r in p['runs'] if r.get('superseded')])
                          if any(r.get('superseded') for p in all_phases for r in p['runs']) else None,
            'marketing_word_count': len((OUT / f'{setup}-marketing-1.md').read_text().split())}
    assert manifest['configs']['helper']['exe'] == manifest['configs']['four-layers']['exe']
    helper_args = list(manifest['configs']['helper']['args'])
    index = helper_args.index('--expert-cache-device1')
    assert helper_args[index + 1] == '2800'
    del helper_args[index:index + 2]
    assert helper_args == manifest['configs']['four-layers']['args']
    continuation = load('test-inputs.json')['continuation_prompt']
    for task, stages in (('marketing', [1]), ('python', [1, 2, '2-continuation-1', '2-continuation-2', 3, 4, 5])):
        for stage in stages:
            a, b = [load(f'{setup}-{task}-{stage}-request.json') for setup in ('helper', 'four-layers')]
            external = lambda req: [m for m in req['messages'] if m['role'] != 'assistant']
            assert external(a) == external(b)
            assert {k: v for k, v in a.items() if k != 'messages'} == {k: v for k, v in b.items() if k != 'messages'}
            if stage not in ('2-continuation-1', '2-continuation-2'):
                original = json.loads((SAVED / f'iq4_xs-{task}-{stage}-request.json').read_text())
                without_continuation = [m for m in external(a) if m['content'] != continuation]
                assert without_continuation == external(original)
                assert {k: v for k, v in a.items() if k not in ('messages', 'model')} == {
                    k: v for k, v in original.items() if k not in ('messages', 'model')}
            verification['matched_external_messages_and_settings'].append(f'{task}-{stage}')
    verification.update(protected_files_unchanged=manifest['protected_files_unchanged'],
                        gpu_after=manifest['gpu_after'],
                        matched_stage2_continuations=2,
                        second_continuation_budget=6144,
                        token_limit_events=[r['stem'] for p in manifest['phases'] for r in p['runs'] if r['finish_reason'] == 'length'])
    summary['deltas'] = {}
    for task in ('marketing', 'python', 'all'):
        a, b = [summary['setups'][setup][task] for setup in ('helper', 'four-layers')]
        summary['deltas'][task] = {'four_layers_decode_change_percent': (b['decode_tok_s'] / a['decode_tok_s'] - 1) * 100,
                                  'four_layers_wall_time_change_percent': (b['wall_s'] / a['wall_s'] - 1) * 100}
    (OUT / 'comparison-verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Observed timing: expert helper versus fully cached first four layers', '',
        '| Setup / task | Output tokens | Decode tok/s | Read tokens | Reused tokens | Prefill tok/s | Prefill seconds | Request wall seconds | MTP acceptance |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for setup, item in summary['setups'].items():
        for task in ('marketing', 'python', 'all'):
            t = item[task]
            lines.append(f'| {setup} / {task} | {t["output_tokens"]} | {t["decode_tok_s"]:.2f} | '
                         f'{t["prompt_read_tokens"]} | {t["prompt_reused_tokens"]} | {t["prefill_tok_s"]:.2f} | '
                         f'{t["prefill_s"]:.2f} | {t["wall_s"]:.2f} | {t["draft_acceptance"] * 100:.2f}% |')
    lines += ['', '| Setup / request | Output tokens | Decode tok/s | Read / reused input tokens | Prefill tok/s | Prefill seconds | Wall seconds |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for phase in manifest['phases']:
        for run in phase['runs']:
            if run.get('superseded'):
                continue
            t = run['timings']
            lines.append(f'| {run["stem"]} | {t["predicted_n"]} | {t["predicted_per_second"]:.2f} | '
                         f'{t["prompt_n"]} / {t["cache_n"]} | {t["prompt_per_second"]:.2f} | '
                         f'{t["prompt_ms"] / 1000:.2f} | {run["wall_s"]:.2f} |')
    lines += ['', '| Setup | Initial startup seconds | First continuation startup | Second continuation startup | Final follow-up startup | Startup + matched request seconds | Actual phase seconds including superseded work and shutdown |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for setup, item in summary['setups'].items():
        startup = item['load_s']
        lines.append(f'| {setup} | {startup["initial"]:.2f} | {startup["stage2-continuation-and-final"]:.2f} | '
                     f'{startup["stage2-second-continuation-and-final"]:.2f} | {startup["followups-final"]:.2f} | '
                     f'{sum(startup.values()) + item["all"]["wall_s"]:.2f} | {item["all_phase_elapsed_s_including_superseded"]:.2f} |')
    lines += ['', 'Decode rates are output-token-weighted; prefill rates count only input tokens actually read. '
              'Stages 1 and 4 read cold histories, matching the saved evaluation procedure. '
              'The helper reached the stage-2 output cap; the four-layer first continuation also reached its cap. '
              'Both receive two identical saved continuation prompts, with budgets of 2048 then 6144 tokens. '
              'The second budget is the only output-limit change from the saved cases. Both continuation requests read cold histories. '
              'Continuation time is included. Superseded helper stages 3–5 from the one-continuation attempt are archived '
              'and excluded from matched task rates and request times, but included in actual phase elapsed time. '
              'Other turns can reuse conversation checkpoints. Output lengths and assistant histories differ, '
              'so wall-time and multi-turn prefill differences are observed workload outcomes, not equal-output comparisons. '
              'One run per placement, one fixed seed, no statistical significance claim.']
    (OUT / 'timing.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'setups': summary['setups'], 'deltas': summary['deltas']}, indent=2))


if __name__ == '__main__':
    main()
