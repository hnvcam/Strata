"""Fresh before/after timings after compilation, plus normal GPU-order validation."""
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('validate', OUT / 'validate.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
b = v.b


def main():
    memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert all(int(x) < 512 for x in memory.splitlines()), memory
    cfg = json.loads((ROOT / 'strata-iq4_xs.json').read_text())
    args = cfg['args']
    paths = [ROOT / 'strata-iq4_xs.json', Path(cfg['exe']), Path(args[args.index('--mtp') + 1]) / 'draft_vocab.bin']
    before = {str(p): b.digest(p) for p in paths}
    hashes = {'original': b.digest(cfg['exe']), 'changed': b.digest(v.BUILT)}
    try:
        for name, exe in [('original-fresh-k2', cfg['exe']), ('changed-fresh-k2', str(v.BUILT))]:
            b.BASE['exe'] = exe
            result = b.run_arm(name, [1, 0], 2)
            assert result['status'] == 'complete', result.get('error')
        v.smoke('changed-default-order', v.BUILT, [0, 1], 'auto')
    finally:
        after = {str(p): b.digest(p) for p in paths}
        b.save(OUT / 'fresh-manifest.json', {'before': before, 'after': after, 'production_unchanged': before == after,
                                          'engine_hashes': hashes,
                                          'gpu_after': subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used', '--format=csv'], text=True)})


if __name__ == '__main__':
    main()
