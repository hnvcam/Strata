"""Matched-residency checks and full-cache capacity verification for separated head placement."""
import copy
import importlib.util
import json
from pathlib import Path
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
source = ROOT / 'bench/results/2026-10-03-dense-layer-placement/validate.py'
spec = importlib.util.spec_from_file_location('existing_validation', source)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
v.OUT = OUT
v.b.OUT = OUT


def main():
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert all(int(n) < 512 for n in used.splitlines()), used
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', v.b.PORT))
    base = copy.deepcopy(v.BASE)
    vocab = Path(base['args'][base['args'].index('--mtp') + 1]) / 'draft_vocab.bin'
    watched = [ROOT / 'strata-iq4_xs.json', Path(base['exe']), vocab, ROOT / 'data/expert-profile.bin']
    hashes = {str(p): v.b.digest(p) for p in watched}
    old = ROOT / 'bench/results/2026-10-04-return-stage/before-strata'
    old.chmod(0o755)
    spec_p = importlib.util.spec_from_file_location('profile_tools', ROOT / 'tools/make_profile.py')
    profile = importlib.util.module_from_spec(spec_p)
    spec_p.loader.exec_module(profile)
    counts = [0] * 48
    pairs = []
    for layer, expert in profile.read_profile(ROOT / 'data/expert-profile.bin'):
        if counts[layer] < 8:
            pairs.append((layer, expert))
            counts[layer] += 1
    parity = OUT / 'parity-profile.bin'
    profile.write_profile(parity, pairs)
    result = {'protected_before': hashes, 'new_engine_sha256': v.b.digest(ROOT / 'build/strata')}
    try:
        v.BASE = copy.deepcopy(base)
        a = v.smoke('before-parity-k44', old, [0, 1], 44, parity)
        b = v.smoke('after-parity-k44', ROOT / 'build/strata', [0, 1], 44, parity)
        assert [r['response']['choices'] for r in a['runs']] == [r['response']['choices'] for r in b['runs']], 'Existing two-stage output changed'
        result['two_stage_regression_identical'] = True
        v.BASE['args'] += ['--head-device', '0']
        c = v.smoke('return-parity-k44', ROOT / 'build/strata', [0, 1], 44, parity)
        assert '0-43 (CUDA0), 44-47 (CUDA1), head/MTP only (CUDA0)' in c['startup_log']
        result['return_vs_contiguous_choices_identical'] = [x['response']['choices'] == y['response']['choices'] for x, y in zip(b['runs'], c['runs'])]
        result['return_vs_contiguous_text_identical'] = [x['response']['choices'][0]['message']['content'] == y['response']['choices'][0]['message']['content'] for x, y in zip(b['runs'], c['runs'])]
        # Check five layers first: a partial cache is explicitly not a valid fit.
        capacity = v.smoke('head-capacity-k43', ROOT / 'build/strata', [0, 1], 43)
        result['five_layer_full_cache'] = 'runs layers 43-47, expert cache 2560 slots' in capacity['startup_log']
        full = v.smoke('head-full-k44', ROOT / 'build/strata', [0, 1], 44)
        assert 'runs layers 44-47, expert cache 2048 slots' in full['startup_log'], 'Four-layer cache incomplete'
        assert '2048 of its 2048 profiled pairs' in full['startup_log']
        assert 'CUDA0 runs head/MTP only, no transformer expert cache' in full['startup_log']
        result.update(status='complete', selected_n=43 if result['five_layer_full_cache'] else 44,
                      four_layer_full_cache=True, head_only_stage_confirmed=True)
    finally:
        result['protected_after'] = {str(p): v.b.digest(p) for p in watched}
        result['protected_files_unchanged'] = result['protected_after'] == hashes
        v.b.save(OUT / 'validation.json', result)
        assert result['protected_files_unchanged']


if __name__ == '__main__':
    main()
