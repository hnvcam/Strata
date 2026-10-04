"""Reuse the saved marketing and five-turn Python tasks for two GPU placements."""
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
        if name == 'helper':
            assert 'layer split:' not in startup
            assert 'additional experts' in startup and 'CUDA1:' in startup
        else:
            assert 'layers 0-3 (CUDA0), 4-47 (CUDA1)' in startup
            assert 'expert cache 2048 slots' in startup
            assert 'pre-filled 2048 of 2048 slots' in startup
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


def main():
    for name in ('test-inputs.json', 'scope-clarification.json', 'source-snapshot.md',
                 'source-manifest.json', 'rubric.json', 'review-facts.json'):
        target = OUT / name
        if target.exists():
            assert sha(target) == sha(SAVED / name)
        else:
            shutil.copyfile(SAVED / name, target)
    cases = json.loads((OUT / 'test-inputs.json').read_text())
    extra = json.loads((OUT / 'scope-clarification.json').read_text())
    base = json.loads((ROOT / 'strata-iq4_xs.json').read_text())
    built = ROOT / 'build/strata'
    vocab = Path(base['args'][base['args'].index('--mtp') + 1]) / 'draft_vocab.bin'
    raw = vocab.read_bytes()
    assert len(raw) == 248320 * 4 and struct.unpack('<248320I', raw) == tuple(range(248320))
    memory = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert all(int(x) < 512 for x in memory.splitlines()), memory
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', PORT))
    try:
        production = api('/health', timeout=3, port=1111)
        assert not production.get('loaded'), 'Production model is loaded'
    except urllib.error.URLError:
        production = {'server_reachable': False}
    watched = [ROOT / 'strata-iq4_xs.json', Path(base['exe']), built, vocab, ROOT / 'data/expert-profile.bin']
    watched += [ROOT / p for p in json.loads((OUT / 'source-manifest.json').read_text())]
    before = {str(p): sha(p) for p in dict.fromkeys(watched)}
    configs = {}
    for name in ('helper', 'four-layers'):
        cfg = copy.deepcopy(base)
        cfg.update(exe=str(built), host='127.0.0.1', port=PORT,
                   log=str(OUT / f'{name}-engine.log'), draft_vocab='full')
        cfg['args'] += ['--suffix-draft', '0']
        cfg.pop('layer_split', None)
        if name == 'helper':
            cfg['gpu'] = 0
            cfg['args'] += ['--expert-cache-device1', '2800']
            cfg['env'] = {**cfg.get('env', {}), 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '0,1'}
        else:
            cfg.update(gpu=[1, 0], layer_split=4)
        configs[name] = cfg
        save(OUT / f'{name}-config.json', cfg)
    source_hashes = {name: sha(OUT / name) for name in ('test-inputs.json', 'scope-clarification.json',
                       'source-snapshot.md', 'source-manifest.json', 'rubric.json', 'review-facts.json')}
    manifest = {'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'status': 'running',
                'hashes_before': before, 'saved_input_hashes': source_hashes,
                'sampling': cases['sampling'], 'configs': configs, 'phases': [],
                'production_before': production,
                'method': 'One marketing article and five Python proposal/review turns per placement; exact saved external inputs/settings. Both use current patched engine. Fresh server for stages 4-5, matching original follow-up cold-history procedure. No warmup or extra benchmark prompts.'}
    save(OUT / 'run-manifest.json', manifest)
    try:
        for name, cfg in configs.items():
            system = [{'role': 'system', 'content': cases['system']}]
            initial = [(i, prompt, limit) for i, (prompt, limit) in enumerate(
                       zip(cases['python_stages'], cases['token_limits']['python_stages']), 1)]
            run_phase(name, cfg, 'initial', [
                ('marketing', [(1, cases['marketing'], cases['token_limits']['marketing'])], system),
                ('python', initial, system)], manifest)
            history = copy.deepcopy(system)
            for stage, prompt, _ in initial:
                response = json.loads((OUT / f'{name}-python-{stage}-response.json').read_text())
                history.extend([{'role': 'user', 'content': prompt},
                                {'role': 'assistant', 'content': response['choices'][0]['message']['content']}])
            followups = [(i, prompt, limit) for i, (prompt, limit) in enumerate(
                         zip(extra['prompts'], extra['max_tokens']), 4)]
            run_phase(name, cfg, 'followups', [('python', followups, history)], manifest)
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        after = {str(p): sha(p) for p in before}
        manifest.update(hashes_after=after, protected_files_unchanged=before == after,
                        finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                        gpu_after=subprocess.check_output(
                            ['nvidia-smi', '--query-gpu=index,name,memory.used', '--format=csv'], text=True))
        save(OUT / 'run-manifest.json', manifest)
        assert before == after, 'Protected files changed'
        print('Finished; protected files unchanged; temporary servers stopped', flush=True)


if __name__ == '__main__':
    main()
