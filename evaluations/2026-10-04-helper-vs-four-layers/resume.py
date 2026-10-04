"""Apply the saved paired continuation rule after the helper's stage-2 cap."""
import copy
import json
from pathlib import Path
import socket
import subprocess
import time

from run import OUT, PORT, api, run_phase, save, sha


def history(name, cases, through):
    messages = [{'role': 'system', 'content': cases['system']}]
    for stage in range(1, through + 1):
        prompt = cases['python_stages'][stage - 1]
        response = json.loads((OUT / f'{name}-python-{stage}-response.json').read_text())
        messages.extend([{'role': 'user', 'content': prompt},
                         {'role': 'assistant', 'content': response['choices'][0]['message']['content']}])
        if stage == 2 and through >= 3:
            response = json.loads((OUT / f'{name}-python-2-continuation-1-response.json').read_text())
            messages.extend([{'role': 'user', 'content': cases['continuation_prompt']},
                             {'role': 'assistant', 'content': response['choices'][0]['message']['content']}])
    return messages


def main():
    cases = json.loads((OUT / 'test-inputs.json').read_text())
    extra = json.loads((OUT / 'scope-clarification.json').read_text())
    manifest = json.loads((OUT / 'run-manifest.json').read_text())
    assert all(sha(p) == h for p, h in manifest['hashes_before'].items())
    memory = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert all(int(x) < 512 for x in memory.splitlines()), memory
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', PORT))
    assert manifest['matched_continuation_required'] == {'setup': 'helper', 'task': 'python', 'stage': 2}
    save(OUT / 'before-continuation-manifest.json', manifest)
    manifest.update(status='running', continuation_plan={
        'python-2': {'paired_prompt': cases['continuation_prompt'],
                     'max_tokens': cases['token_limits']['continuation'],
                     'reason': 'Helper stage 2 reached its saved 4096-token cap. Both setups receive one identical continuation before stage 3.'}},
        continuation_state_note='Both setups restart before the paired stage-2 continuation. This adds one matched cold-history read per setup beyond the original procedure.')
    manifest['phases'][0]['status'] = 'complete_with_truncation'
    manifest['phases'][0]['error'] = 'Stage-2 output cap; original record retained, paired continuation planned'
    manifest.pop('error', None)
    manifest.pop('finished_at', None)
    save(OUT / 'run-manifest.json', manifest)
    system = [{'role': 'system', 'content': cases['system']}]
    try:
        initial = [(i, prompt, limit) for i, (prompt, limit) in enumerate(
            zip(cases['python_stages'][:2], cases['token_limits']['python_stages'][:2]), 1)]
        run_phase('four-layers', manifest['configs']['four-layers'], 'initial', [
            ('marketing', [(1, cases['marketing'], cases['token_limits']['marketing'])], system),
            ('python', initial, system)], manifest, allowed_truncation_stages=(2,))
        for name in ('helper', 'four-layers'):
            cfg = manifest['configs'][name]
            stages = [('2-continuation-1', cases['continuation_prompt'], cases['token_limits']['continuation']),
                      (3, cases['python_stages'][2], cases['token_limits']['python_stages'][2])]
            run_phase(name, cfg, 'stage2-continuation-and-final',
                      [('python', stages, history(name, cases, 2))], manifest)
            stages = [(i, prompt, limit) for i, (prompt, limit) in enumerate(
                zip(extra['prompts'], extra['max_tokens']), 4)]
            run_phase(name, cfg, 'followups', [('python', stages, history(name, cases, 3))], manifest)
        manifest.update(status='complete', matched_continuation_applied=True)
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        after = {str(p): sha(p) for p in manifest['hashes_before']}
        manifest.update(hashes_after=after,
                        protected_files_unchanged=manifest['hashes_before'] == after,
                        finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                        gpu_after=subprocess.check_output(
                            ['nvidia-smi', '--query-gpu=index,name,memory.used', '--format=csv'], text=True))
        save(OUT / 'run-manifest.json', manifest)
        assert manifest['protected_files_unchanged']
        print('Matched continuation run finished; protected files unchanged', flush=True)


if __name__ == '__main__':
    main()
