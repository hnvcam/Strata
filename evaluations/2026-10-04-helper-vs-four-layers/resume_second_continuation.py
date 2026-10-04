"""Complete the paired review with a larger budget and matched downstream histories."""
import json
from pathlib import Path
import shutil
import socket
import subprocess
import time

from run import OUT, PORT, run_phase, save, sha


def history(name, cases, through, continuations):
    messages = [{'role': 'system', 'content': cases['system']}]
    for stage in range(1, through + 1):
        prompt = cases['python_stages'][stage - 1]
        response = json.loads((OUT / f'{name}-python-{stage}-response.json').read_text())
        messages.extend([{'role': 'user', 'content': prompt},
                         {'role': 'assistant', 'content': response['choices'][0]['message']['content']}])
        if stage == 2:
            for i in range(1, continuations + 1):
                response = json.loads((OUT / f'{name}-python-2-continuation-{i}-response.json').read_text())
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
    assert manifest['matched_continuation_required'] == {
        'setup': 'four-layers', 'task': 'python', 'stage': '2-continuation-1'}
    save(OUT / 'before-second-continuation-manifest.json', manifest)
    archive = OUT / 'superseded-one-continuation'
    archive.mkdir(exist_ok=True)
    for stage in (3, 4, 5):
        for path in OUT.glob(f'helper-python-{stage}.*'):
            assert not (archive / path.name).exists()
            shutil.move(str(path), archive / path.name)
        for path in OUT.glob(f'helper-python-{stage}-*'):
            assert not (archive / path.name).exists()
            shutil.move(str(path), archive / path.name)
    review = OUT / 'helper-python-review.json'
    if review.exists():
        shutil.move(str(review), archive / review.name)
    for phase in manifest['phases']:
        for run in phase['runs']:
            if run['setup'] == 'helper' and run['stage'] in (3, 4, 5):
                run.update(superseded=True, superseded_reason='Second paired continuation changes downstream history',
                           archived_directory=str(archive))
        if phase['setup'] == 'four-layers' and phase['phase'] == 'stage2-continuation-and-final':
            phase.update(status='complete_with_truncation', error='Continuation budget reached; second paired continuation planned')
    adjustment = {
        'reason': 'Four-layer first continuation reached 2048 tokens after the helper stage-2 cap. Both setups need a second continuation and matched downstream histories.',
        'prompt': cases['continuation_prompt'], 'second_continuation_max_tokens': 6144,
        'original_continuation_max_tokens': cases['token_limits']['continuation'],
        'all_original_task_and_scope_limits_unchanged': True,
        'extra_cold_restart_before_second_continuation_for_both': True,
        'superseded_outputs_retained': str(archive)}
    save(OUT / 'test-adjustments.json', adjustment)
    manifest.update(status='running', second_continuation_adjustment=adjustment)
    manifest.pop('error', None)
    manifest.pop('finished_at', None)
    manifest.pop('matched_continuation_required', None)
    save(OUT / 'run-manifest.json', manifest)
    try:
        for name in ('helper', 'four-layers'):
            cfg = manifest['configs'][name]
            stages = [('2-continuation-2', cases['continuation_prompt'], 6144),
                      (3, cases['python_stages'][2], cases['token_limits']['python_stages'][2])]
            run_phase(name, cfg, 'stage2-second-continuation-and-final',
                      [('python', stages, history(name, cases, 2, 1))], manifest)
            stages = [(i, prompt, limit) for i, (prompt, limit) in enumerate(
                zip(extra['prompts'], extra['max_tokens']), 4)]
            run_phase(name, cfg, 'followups-final',
                      [('python', stages, history(name, cases, 3, 2))], manifest)
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
        print('Completed paired review with matched histories; protected files unchanged', flush=True)


if __name__ == '__main__':
    main()
