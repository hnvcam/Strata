"""Read-only production configs; run the original and rebuilt engines sequentially."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('placement_bench', ROOT / 'bench/results/2026-10-03-iq4-xs-gpu-placement/benchmark.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b.OUT = OUT
BASE = copy.deepcopy(b.BASE)
BUILT = ROOT / 'build/strata'


def smoke(name, exe, gpu, split, profile=None):
    cfg = copy.deepcopy(BASE)
    cfg.update(exe=str(exe), gpu=gpu, host='127.0.0.1', port=b.PORT, log=str(OUT / f'{name}-engine.log'))
    cfg.pop('layer_split', None)
    if split is not None:
        cfg['layer_split'] = split
    cfg['args'] += ['--suffix-draft', '0', '--adapt-every', '0']
    if profile:
        cfg['args'][cfg['args'].index('--expert-profile') + 1] = str(profile)
    path = OUT / f'{name}-config.json'
    b.save(path, cfg)
    log = Path(cfg['log'])
    log.write_text('')
    console = (OUT / f'{name}-server.log').open('w')
    proc = subprocess.Popen([str(ROOT / '.venv/bin/python'), str(ROOT / 'serve/server.py'),
                             '--engine', 'strata', '--config', str(path), '--port', str(b.PORT), '--host', '127.0.0.1'],
                            cwd=ROOT, stdout=console, stderr=subprocess.STDOUT, start_new_session=True)
    result = {'name': name, 'config': cfg, 'runs': []}
    try:
        start = time.monotonic()
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f'Server exited: {proc.returncode}; {console.name}')
            try:
                if b.http('/health', timeout=2).get('loaded'):
                    break
            except Exception:
                pass
            if time.monotonic() - start > 300:
                raise TimeoutError(name)
            time.sleep(1)
        result['load_s'] = time.monotonic() - start
        result['startup_log'] = log.read_text()
        assert 'draft head over 248320 tokens' in result['startup_log']
        print(name, 'ready', round(result['load_s'], 2), flush=True)
        texts = ['Explain how a refrigerator works in three sentences.',
                 'Write a Python function that merges two sorted lists.',
                 'Giải thích sự khác nhau giữa RAM và SSD bằng tiếng Việt.']
        for i, text in enumerate(texts):
            result['runs'].append(b.run_request(str(i), b.request_for(text, 96), log))
        result['gpu_snapshot'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used', '--format=csv'], text=True)
        result['status'] = 'complete'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = repr(exc)
        raise
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        console.close()
        b.save(OUT / f'{name}-results.json', result)
        time.sleep(2)
    return result


def main():
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert all(int(x) < 512 for x in memory.splitlines()), memory
    args = BASE['args']
    vocab = Path(args[args.index('--mtp') + 1]) / 'draft_vocab.bin'
    paths = [ROOT / 'strata-iq4_xs.json', Path(BASE['exe']), vocab]
    before = {str(p): b.digest(p) for p in paths}
    manifest = {'before': before, 'built_engine_sha256': b.digest(BUILT)}
    try:
        # Identical expert residency and disabled adaptation isolate dense-weight correctness.
        pspec = importlib.util.spec_from_file_location('profile', ROOT / 'tools/make_profile.py')
        profile = importlib.util.module_from_spec(pspec)
        pspec.loader.exec_module(profile)
        ranked = profile.read_profile(ROOT / 'data/expert-profile.bin')
        counts = [0] * 48
        pairs = []
        for layer, expert in ranked:
            if counts[layer] < 8:
                pairs.append((layer, expert))
                counts[layer] += 1
        parity_profile = OUT / 'parity-profile.bin'
        profile.write_profile(parity_profile, pairs)
        original = smoke('original-parity-k2', BASE['exe'], [1, 0], 2, parity_profile)
        changed = smoke('changed-parity-k2', BUILT, [1, 0], 2, parity_profile)
        parity = [a['response']['choices'] == c['response']['choices'] for a, c in zip(original['runs'], changed['runs'])]
        manifest['same_output_with_identical_expert_residency'] = parity
        assert all(parity), 'Dense placement changed greedy responses with identical expert residency'
        b.BASE = copy.deepcopy(BASE)
        b.BASE['exe'] = str(BUILT)
        result = b.run_arm('changed-full-cache-k2', [1, 0], 2)
        assert result['status'] == 'complete', result.get('error')
        auto = smoke('changed-auto', BUILT, [1, 0], 'auto')
        manifest['auto_status'] = auto['status']
        single = smoke('changed-single', BUILT, 0, None)
        manifest['single_status'] = single['status']
    finally:
        manifest['after'] = {str(p): b.digest(p) for p in paths}
        manifest['production_unchanged'] = before == manifest['after']
        manifest['gpu_after'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used', '--format=csv'], text=True)
        b.save(OUT / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
