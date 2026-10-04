"""Test all experts of the first four layers using per-stage dense weights."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('validation', OUT / 'validate.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
b = v.b


def main():
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert all(int(x) < 512 for x in memory.splitlines()), memory
    cfg = json.loads((ROOT / 'strata-iq4_xs.json').read_text())
    args = cfg['args']
    vocab = Path(args[args.index('--mtp') + 1]) / 'draft_vocab.bin'
    paths = [ROOT / 'strata-iq4_xs.json', Path(cfg['exe']), vocab, v.BUILT]
    before = {str(p): b.digest(p) for p in paths}
    sizes = {}
    for line in Path('/home/hnvcam/Workplace/Strata-data/packs/ud-iq4_xs/native_experts.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            fields = line.split()
            sizes[int(fields[0])] = int(fields[4]) * 512
    manifest = {'before': before, 'expert_bytes_first_four': sum(sizes[i] for i in range(4)),
                'expert_bytes_first_five': sum(sizes[i] for i in range(5)), 'per_layer_expert_bytes': sizes}
    print('All experts of layers 0-3:', manifest['expert_bytes_first_four'] / 2**30, 'GiB', flush=True)
    try:
        b.BASE = copy.deepcopy(cfg)
        b.BASE['exe'] = str(v.BUILT)
        result = b.run_arm('changed-full-cache-k4', [1, 0], 4)
        assert result['status'] == 'complete', result.get('error')
        manifest['complete_cache_confirmed'] = 'expert cache 2048 slots' in result['startup_log']
        manifest['status'] = result['status']
        assert manifest['complete_cache_confirmed'], 'The first-four-layer cache was not complete'
    finally:
        after = {str(p): b.digest(p) for p in paths}
        manifest.update(after=after, production_unchanged=before == after,
                        gpu_after=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used', '--format=csv'], text=True))
        b.save(OUT / 'four-layer-manifest.json', manifest)


if __name__ == '__main__':
    main()
