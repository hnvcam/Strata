"""Compare continuation and branch restoration with identical cache residency."""
import copy
import json
import socket
import subprocess
from validate import ROOT, OUT, v


def main():
    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
    assert all(int(x) < 512 for x in used.splitlines()), used
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', v.b.PORT))
    base = copy.deepcopy(v.BASE)
    raw_request = v.b.run_request
    first = {}

    def branched_request(label, request, log):
        if label != '0':
            request['messages'] = copy.deepcopy(first['messages']) + [
                {'role': 'assistant', 'content': first['answer']},
                request['messages'][-1]]
        run = raw_request(label, request, log)
        if label == '0':
            first.update(messages=copy.deepcopy(request['messages']),
                         answer=run['response']['choices'][0]['message']['content'])
        return run

    v.b.run_request = branched_request
    v.BASE = copy.deepcopy(base)
    # The next requests continue from turn one, then branch back to it.
    a = v.smoke('checkpoint-contiguous', ROOT / 'bench/results/2026-10-04-return-stage/before-strata', [0, 1], 44, OUT / 'parity-profile.bin')
    first.clear()
    v.BASE['args'] += ['--head-device', '0']
    b = v.smoke('checkpoint-head-primary', ROOT / 'build/strata', [0, 1], 44, OUT / 'parity-profile.bin')
    identical = [x['response']['choices'] == y['response']['choices'] for x, y in zip(a['runs'], b['runs'])]
    reused = [r['response']['timings']['cache_n'] for r in b['runs']]
    result = {'identical_choices': identical, 'return_prompt_reused_tokens': reused,
              'method': 'Fixed eight experts per layer; adaptation disabled. Same English request, Python continuation, then Vietnamese branch from the English turn.'}
    v.b.save(OUT / 'checkpoint-validation.json', result)
    assert all(identical), 'Returning-stage continuation or branch output diverged'
    assert all(n > 0 for n in reused[1:]), 'Continuation/branch did not reuse any state'


if __name__ == '__main__':
    main()
