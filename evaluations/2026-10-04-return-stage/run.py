"""Reuse the saved tasks for a 5070 Ti -> 4060 -> 5070 Ti placement."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SAVED = ROOT / 'evaluations/2026-10-03-iq3-s-vs-iq4-xs'
PORT = 1112


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def api(path, body=None, timeout=1800, port=PORT):
    request = urllib.request.Request(f'http://127.0.0.1:{port}' + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode()}') from exc


def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)


def run_phase(name, cfg, phase, jobs, manifest, allowed_truncation_stages=()):
    config_path = OUT / f'{name}-config.json'
    console = (OUT / f'{name}-{phase}-server.log').open('w')
    log = Path(cfg['log'])
    offset = log.stat().st_size if log.exists() else 0
    start = time.monotonic()
    proc = subprocess.Popen([str(ROOT / '.venv/bin/python'), str(ROOT / 'serve/server.py'),
        '--engine', 'strata', '--config', str(config_path), '--port', str(PORT), '--host', '127.0.0.1'],
        cwd=ROOT, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
    entry = {'setup': name, 'phase': phase, 'runs': []}
    manifest['phases'].append(entry)
    save(OUT / 'run-manifest.json', manifest)
    try:
        last = 0
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f'{name} server exited {proc.returncode}')
            try:
                if api('/health', timeout=2).get('loaded'):
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            elapsed = time.monotonic() - start
            if elapsed > 300:
                raise TimeoutError(f'{name} startup')
            if elapsed - last > 15:
                print(f'{name}/{phase}: loading {elapsed:.0f}s', flush=True)
                last = elapsed
            time.sleep(1)
        entry['load_s'] = time.monotonic() - start
        with log.open('rb') as stream:
            stream.seek(offset)
            startup = stream.read().decode()
        entry['startup_log'] = startup
        (OUT / f'{name}-{phase}-startup.log').write_text(startup)
        assert 'draft head over 248320 tokens' in startup
        n = manifest['selected_n']
        assert f'layers 0-{n-1} (CUDA0), {n}-46 (CUDA1), 47-47 (CUDA0)' in startup
        slots = (47-n)*512
        assert f'runs layers {n}-46, expert cache {slots} slots' in startup
        assert f'{slots} of its {slots} profiled pairs' in startup
        assert 'runs layers 47-47, expert cache 512 slots' in startup
        assert 'session [47, 48) and the head' in startup
        print(f'{name}/{phase}: ready in {entry["load_s"]:.2f}s', flush=True)
        for line in startup.splitlines():
            if any(s in line for s in ('expert cache ', 'additional experts', 'draft head over', 'runs layers', 'dense weights for layers')):
                print(line, flush=True)
        save(OUT / 'run-manifest.json', manifest)
        for task, stages, initial_messages in jobs:
            messages = copy.deepcopy(initial_messages)
            for stage, prompt, limit in stages:
                messages.append({'role': 'user', 'content': prompt})
                payload = {**manifest['sampling'], 'model': cfg['model_name'],
                           'messages': copy.deepcopy(messages), 'max_tokens': limit, 'stream': False}
                stem = f'{name}-{task}-{stage}'
                save(OUT / f'{stem}-request.json', payload)
                request_offset = log.stat().st_size
                print(f'{stem}: started', flush=True)
                request_start = time.monotonic()
                response = api('/v1/chat/completions', payload)
                wall = time.monotonic() - request_start
                save(OUT / f'{stem}-response.json', response)
                choice = response['choices'][0]
                text = choice['message'].get('content') or ''
                (OUT / f'{stem}.md').write_text(text + '\n')
                with log.open('rb') as stream:
                    stream.seek(request_offset)
                    request_log = stream.read().decode()
                (OUT / f'{stem}-engine.log').write_text(request_log)
                assert text.strip() and response.get('timings'), 'Empty output or missing timings'
                t = response['timings']
                run = {'setup': name, 'task': task, 'stage': stage, 'stem': stem,
                       'input_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                       'wall_s': wall, 'finish_reason': choice['finish_reason'], 'timings': t}
                entry['runs'].append(run)
                save(OUT / 'run-manifest.json', manifest)
                print(json.dumps({'request': stem, 'finish_reason': choice['finish_reason'],
                    'tokens': t['predicted_n'], 'decode_tok_s': t['predicted_per_second'],
                    'prompt_read': t['prompt_n'], 'prompt_reused': t['cache_n'],
                    'prefill_tok_s': t['prompt_per_second'], 'wall_s': round(wall, 2)}), flush=True)
                if choice['finish_reason'] == 'length' and stage not in allowed_truncation_stages:
                    manifest['matched_continuation_required'] = {'setup': name, 'task': task, 'stage': stage}
                    save(OUT / 'run-manifest.json', manifest)
                    raise RuntimeError('Apply the saved paired continuation policy before advancing')
                messages.append({'role': 'assistant', 'content': text})
        entry['gpu_snapshot'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,temperature.gpu', '--format=csv'], text=True)
        entry['status'] = 'complete'
    except Exception as exc:
        entry.update(status='failed', error=repr(exc))
        raise
    finally:
        stop(proc)
        console.close()
        entry['elapsed_s_including_stop'] = time.monotonic() - start
        save(OUT / 'run-manifest.json', manifest)



def completed_history(cases, stages):
    messages = [{'role': 'system', 'content': cases['system']}]
    for stage, prompt, _ in stages:
        response = json.loads((OUT / f'return-python-{stage}-response.json').read_text())
        messages.extend([{'role': 'user', 'content': prompt},
                         {'role': 'assistant', 'content': response['choices'][0]['message']['content']}])
    return messages


def main():
    # Retain exactly the inputs/settings and two review continuations from the
    # final helper/four-layer comparison. Each response stays in its own history.
    for name in ('test-inputs.json', 'scope-clarification.json', 'source-snapshot.md',
                 'source-manifest.json', 'rubric.json', 'review-facts.json'):
        shutil.copyfile(SAVED / name, OUT / name)
    cases = json.loads((OUT / 'test-inputs.json').read_text())
    extra = json.loads((OUT / 'scope-clarification.json').read_text())
    validation = json.loads((ROOT / 'bench/results/2026-10-04-return-stage/validation.json').read_text())
    assert validation['status'] == 'complete'
    n = validation['selected_n']
    base = json.loads((ROOT / 'strata-iq4_xs.json').read_text())
    built = ROOT / 'build/strata'
    vocab = Path(base['args'][base['args'].index('--mtp') + 1]) / 'draft_vocab.bin'
    raw = vocab.read_bytes()
    assert struct.unpack('<248320I', raw) == tuple(range(248320))
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert all(int(x) < 512 for x in used.splitlines()), used
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', PORT))
    watched = [ROOT / 'strata-iq4_xs.json', Path(base['exe']), built, vocab, ROOT / 'data/expert-profile.bin']
    watched += [ROOT / p for p in json.loads((OUT / 'source-manifest.json').read_text())]
    before = {str(p): sha(p) for p in dict.fromkeys(watched)}
    cfg = copy.deepcopy(base)
    cfg.update(exe=str(built), gpu=[0, 1], layer_split=f'{n},47', host='127.0.0.1', port=PORT,
               log=str(OUT / 'return-engine.log'), draft_vocab='full')
    cfg['args'] += ['--suffix-draft', '0', '--split-device', '1,0']
    save(OUT / 'return-config.json', cfg)
    manifest = {'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'status': 'running',
                'hashes_before': before, 'sampling': cases['sampling'], 'selected_n': n,
                'configs': {'return': cfg}, 'phases': [],
                'saved_input_hashes': {name: sha(OUT / name) for name in
                    ('test-inputs.json', 'scope-clarification.json', 'source-snapshot.md',
                     'source-manifest.json', 'rubric.json', 'review-facts.json')},
                'method': 'Exact saved marketing and five Python stages; two stage-2 continuations with caps 2048 and 6144, forced to match the final comparison external message sequence. Same four fresh-server phases. No warmup or other benchmark prompts.'}
    save(OUT / 'run-manifest.json', manifest)
    all_stages = [(i, prompt, limit) for i, (prompt, limit) in enumerate(
        zip(cases['python_stages'], cases['token_limits']['python_stages']), 1)]
    try:
        system = [{'role': 'system', 'content': cases['system']}]
        run_phase('return', cfg, 'initial', [
            ('marketing', [(1, cases['marketing'], cases['token_limits']['marketing'])], system),
            ('python', all_stages[:2], system)], manifest, allowed_truncation_stages=(2,))
        stages = all_stages[:2] + [('2-continuation-1', cases['continuation_prompt'], 2048)]
        run_phase('return', cfg, 'stage2-continuation',
                  [('python', stages[-1:], completed_history(cases, stages[:-1]))], manifest,
                  allowed_truncation_stages=('2-continuation-1',))
        stages += [('2-continuation-2', cases['continuation_prompt'], 6144)]
        run_phase('return', cfg, 'stage2-second-continuation-and-final',
                  [('python', stages[-1:] + all_stages[2:], completed_history(cases, stages[:-1]))], manifest)
        stages += all_stages[2:]
        followups = [(i, prompt, limit) for i, (prompt, limit) in enumerate(zip(extra['prompts'], extra['max_tokens']), 4)]
        run_phase('return', cfg, 'followups-final',
                  [('python', followups, completed_history(cases, stages))], manifest)
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        after = {str(p): sha(p) for p in before}
        manifest.update(hashes_after=after, protected_files_unchanged=before == after,
                        finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                        gpu_after=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used', '--format=csv'], text=True))
        save(OUT / 'run-manifest.json', manifest)
        assert before == after, 'Protected files changed during evaluation'
        print('Finished; temporary servers stopped; protected files unchanged', flush=True)


if __name__ == '__main__':
    main()
