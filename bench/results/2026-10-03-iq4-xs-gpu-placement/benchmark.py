"""Matched local IQ4_XS GPU placement measurements; production files are read only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = json.loads((ROOT / 'strata-iq4_xs.json').read_text())
PORT = 1112
URL = f'http://127.0.0.1:{PORT}'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def http(route, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + route, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode()}') from exc


def request_for(text, tokens=512):
    return {'model': BASE['model_name'], 'messages': [{'role': 'user', 'content': text}],
            'max_tokens': tokens, 'temperature': 0, 'top_p': 1, 'top_k': 0, 'seed': 1234,
            'reasoning_effort': 'none', 'chat_template_kwargs': {'enable_thinking': False},
            'stream': False}


def run_request(label, request, log):
    offset = log.stat().st_size
    t0 = time.monotonic()
    response = http('/v1/chat/completions', request)
    elapsed = time.monotonic() - t0
    if 'error' in response or 'timings' not in response:
        raise RuntimeError(response)
    entry = {'label': label, 'request': request, 'response': response, 'wall_s': elapsed,
             'engine_log': log.read_text()[offset:]}
    timings = response['timings']
    print(json.dumps({'request': label, 'tokens': timings['predicted_n'],
                      'decode_tok_s': timings['predicted_per_second'],
                      'prompt_tokens_read': timings['prompt_n'],
                      'prompt_tok_s': timings['prompt_per_second'], 'wall_s': round(elapsed, 2)}), flush=True)
    return entry


def run_arm(name, gpu, split=None, overrides=None, pilot=False, extra_args=None, extra_env=None):
    cfg = copy.deepcopy(BASE)
    cfg.update(gpu=gpu, host='127.0.0.1', port=PORT, log=str(OUT / f'{name}-engine.log'))
    cfg.pop('layer_split', None)
    if split is not None:
        cfg['layer_split'] = split
    # Disable suffix lookup in every arm to measure the same Q2 MTP mechanism.
    cfg['args'] += ['--suffix-draft', '0']
    cfg['args'] += extra_args or []
    if extra_env:
        cfg['env'] = {**cfg.get('env', {}), **extra_env}
    for key, value in (overrides or {}).items():
        cfg['args'][cfg['args'].index(key) + 1] = str(value)
    config_path = OUT / f'{name}-config.json'
    save(config_path, cfg)
    log = Path(cfg['log'])
    log.write_text('')
    runs = []
    result = {'name': name, 'config': cfg, 'pilot': pilot, 'runs': runs}
    console = (OUT / f'{name}-server.log').open('w')
    proc = subprocess.Popen([str(ROOT / '.venv/bin/python'), str(ROOT / 'serve/server.py'),
                             '--engine', 'strata', '--config', str(config_path), '--port', str(PORT),
                             '--host', '127.0.0.1'], cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        t0 = time.monotonic()
        last = 0
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f'{name} server exited {proc.returncode}; see its logs')
            try:
                health = http('/health', timeout=2)
                if health.get('loaded'):
                    break
            except (urllib.error.URLError, TimeoutError):
                pass
            elapsed = time.monotonic() - t0
            if elapsed > 300:
                raise TimeoutError(f'{name}: startup exceeded 300 seconds')
            if elapsed - last > 10:
                last = elapsed
                print(f'{name}: loading, {elapsed:.0f} seconds', flush=True)
            time.sleep(1)
        result['load_s'] = time.monotonic() - t0
        result['health'] = health
        startup = log.read_text()
        if 'draft head over 248320 tokens' not in startup:
            raise RuntimeError('Full draft vocabulary not confirmed in engine log')
        if extra_args and '--expert-cache-device1' in extra_args:
            if 'layer split' in startup or 'CUDA1:' not in startup or 'additional experts' not in startup:
                raise RuntimeError('Expected one main-model stage on the 5070 Ti and an expert-only CUDA1 helper')
        result['startup_log'] = startup
        print(f'{name}: ready in {result["load_s"]:.1f} seconds', flush=True)
        for line in startup.splitlines():
            if any(s in line for s in ('layer split auto: K=', 'runs layers', 'expert cache ',
                                       'draft head over', 'VRAM free with', 'layer split: layers',
                                       'additional experts', 'context ready')):
                print(line, flush=True)
        result['warmup'] = run_request('warmup', request_for(
            'Explain the difference between RAM and SSD storage in a personal computer.', 128), log)
        old = json.loads((ROOT / 'bench/results/2026-10-03-iq4-xs-mtp/full-vocab/q2-results.json').read_text())
        texts = {language: next(r['request']['messages'][0]['content'] for r in old['runs']
                                if r['language'] == language and r['condition'] == 'current_greedy')
                 for language in ('english', 'code', 'vietnamese')}
        for repeat in range(1 if pilot else 2):
            for language, text in texts.items():
                runs.append(run_request(f'{language}-{repeat + 1}', request_for(text), log))
                save(OUT / f'{name}-results.json', result)
        if not pilot:
            notes = '\n'.join(f'Entry {i:04d}: The maintenance team inspected the water pump, '
                              'checked the pressure gauge, replaced a worn seal, and recorded '
                              'the result in the service log.' for i in range(1, 281))
            text = ('Read the following maintenance notes. After the notes, write a detailed '
                    'general maintenance checklist covering inspection, diagnosis, repair and documentation.\n'
                    + notes + '\nNow write the detailed checklist. Do not list individual entry numbers.')
            runs.append(run_request('long-prompt', request_for(text, 256), log))
        result['gpu_snapshot'] = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,pcie.link.gen.current,pcie.link.width.current,temperature.gpu',
             '--format=csv'], text=True)
        result['status'] = 'complete'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = repr(exc)
        print(f'{name}: FAILED: {exc}', flush=True)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        console.close()
        time.sleep(2)
        save(OUT / f'{name}-results.json', result)
    return result


def main():
    args = BASE['args']
    vocab = Path(args[args.index('--mtp') + 1]) / 'draft_vocab.bin'
    paths = [ROOT / 'strata-iq4_xs.json', Path(BASE['exe']), vocab]
    before = {str(p): digest(p) for p in paths}
    raw = vocab.read_bytes()
    if len(raw) != 248320 * 4:
        raise RuntimeError('Unexpected draft vocabulary size')
    # Both cards must be free before allocating the 55-GiB expert arena.
    gpu_memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    if any(int(n) > 512 for n in gpu_memory.splitlines()):
        raise RuntimeError('A GPU is still occupied; not starting another model')
    manifest = {'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'hashes_before': before,
                'draft_vocab_count': len(raw) // 4, 'base_config': BASE,
                'hardware': subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,pcie.link.gen.max,pcie.link.width.current', '--format=csv'], text=True),
                'method': 'Q2 MTP, full vocabulary, 131072 context, int8 KV, spec=4, cutoff=.70, greedy; suffix drafts off in all arms; one warmup; English/code/Vietnamese twice at 512 output tokens plus one cold long prompt at 256.'}
    save(OUT / 'manifest.json', manifest)
    variants = {
        'single': ('5070-alone', 0, None, None, False),
        'single-check': ('5070-alone-check', 0, None, None, False),
        'dual': ('5070-4060-auto', [0, 1], 'auto', None, False),
        '4060-first': ('4060-first-5070-head-mtp', [1, 0], 'auto', None, False),
        '4060-first-four': ('4060-first-four-layers', [1, 0], 4, None, True),
        'more-layers': ('5070-4060-k44', [0, 1], 44, None, True),
        'more-experts': ('5070-4060-tuned', [0, 1], 'auto',
                         {'--prefill': 256, '--vram-reserve-mib': 1024}, False),
        'more-experts-safe': ('5070-4060-tuned-safe', [0, 1], 'auto',
                              {'--prefill': 256, '--vram-reserve-mib': 1152}, False),
        # gpu=0 prevents the server from adding --layer-split. Both physical devices remain
        # visible to the engine, and only the explicit helper cache uses CUDA1.
        'helper': ('5070-head-mtp-4060-experts', 0, None, None, False,
                   ['--expert-cache-device1', '2800'],
                   {'CUDA_DEVICE_ORDER': 'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES': '0,1'}),
    }
    selected = sys.argv[1:] or ['single', 'dual']
    try:
        for key in selected:
            run_arm(*variants[key])
    finally:
        manifest['hashes_after'] = {str(p): digest(p) for p in paths}
        manifest['production_files_unchanged'] = before == manifest['hashes_after']
        manifest['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        manifest['gpu_after'] = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used', '--format=csv'], text=True)
        save(OUT / 'manifest.json', manifest)
        print('Finished; production files unchanged:', manifest['production_files_unchanged'], flush=True)


if __name__ == '__main__':
    main()
