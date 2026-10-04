"""Run the saved placement benchmark with layers 46-47/head/MTP on the 4060."""
import copy
import importlib.util
import json
from pathlib import Path
import socket
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'placement_benchmark', ROOT / 'bench/results/2026-10-03-iq4-xs-gpu-placement/benchmark.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b.OUT = OUT


def main():
    cfg = copy.deepcopy(b.BASE)
    args = cfg['args']
    vocab = Path(args[args.index('--mtp') + 1]) / 'draft_vocab.bin'
    built = ROOT / 'build/strata'
    paths = [ROOT / 'strata-iq4_xs.json', Path(cfg['exe']), vocab, built,
             ROOT / 'data/expert-profile.bin', Path(__file__), Path(spec.origin)]
    before = {str(p): b.digest(p) for p in paths}
    raw = vocab.read_bytes()
    assert len(raw) == 248320 * 4
    assert struct.unpack('<248320I', raw) == tuple(range(248320)), 'Expected full draft vocabulary'
    memory = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,name,memory.used', '--format=csv'], text=True)
    used = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert all(int(x) < 512 for x in used.splitlines()), memory
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', b.PORT))
    saved = json.loads((ROOT / 'bench/results/2026-10-03-dense-layer-placement/fresh-manifest.json').read_text())
    assert before[str(built)] == saved['engine_hashes']['changed'], 'Patched engine differs from saved benchmark'
    b.BASE['exe'] = str(built)
    manifest = {
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'before': before,
        'gpu_before': memory, 'draft_vocab_count': len(raw) // 4,
        'requested_placement': {'5070_ti': 'layers 0-45', '4060': 'layers 46-47, output head, Q2 MTP'},
        'method': 'Original placement runner: 128-token warmup, English/Python/Vietnamese twice at up to 512 tokens, cold 9851-token prompt at up to 256 tokens. Greedy, thinking off, suffix drafts off, adaptive caching and MTP policy enabled.'}
    b.save(OUT / 'manifest.json', manifest)
    try:
        result = b.run_arm('patched-k46', [0, 1], 46)
        assert result['status'] == 'complete', result.get('error')
        assert 'layers 0-45 (CUDA0), 46-47 (CUDA1)' in result['startup_log'], 'Unexpected placement'
        baseline = json.loads((ROOT / 'bench/results/2026-10-03-dense-layer-placement/changed-full-cache-k4-results.json').read_text())
        assert result['warmup']['request'] == baseline['warmup']['request']
        assert [r['request'] for r in result['runs']] == [r['request'] for r in baseline['runs']], 'Requests differ from saved evaluation'
        long = result['runs'][-1]['response']['timings']
        assert long['prompt_n'] == 9851 and long['cache_n'] == 0, 'Long prompt was not cold/matched'
        manifest.update(status='complete', identical_requests_to_saved_k4=True,
                        confirmed_layer_split=True, cold_prompt_tokens=long['prompt_n'])
    except Exception as exc:
        manifest.update(status='failed', error=repr(exc))
        raise
    finally:
        after = {str(p): b.digest(p) for p in paths}
        manifest.update(after=after, files_unchanged=before == after,
                        finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                        gpu_after=subprocess.check_output(
                            ['nvidia-smi', '--query-gpu=index,name,memory.used', '--format=csv'], text=True))
        b.save(OUT / 'manifest.json', manifest)
        print('Finished; protected files unchanged:', manifest['files_unchanged'], flush=True)


if __name__ == '__main__':
    main()
