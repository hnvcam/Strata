"""Replay the saved original writing/review tasks against the user-loaded endpoint.

Does not start, stop, configure or unload any model server. No benchmark suite imports.
"""
from pathlib import Path
import hashlib
import json
import time
import urllib.error
import urllib.request

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SAVED = ROOT / 'evaluations/2026-10-03-iq3-s-vs-iq4-xs'
BASE = 'http://127.0.0.1:1234'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def api(path, body=None):
    req = urllib.request.Request(BASE + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: {e.read().decode(errors="replace")}') from e


def main():
    old = json.loads((SAVED / 'run-manifest.json').read_text())
    watched = list(old['production_hashes_before']) + [str(ROOT / 'engine/strata')]
    before = {f: sha(f) for f in watched}
    manifest = {'endpoint': BASE + '/v1', 'runs': [], 'watched_hashes_before': before,
        'reference_directory': str(SAVED), 'input_files': {},
        'note': 'Identical saved system/user prompts and requested sampling, with explicit min_p=0 matching the prior Strata default; user-loaded server settings differ from Strata.'}
    for name in ['test-inputs.json', 'scope-clarification.json', 'source-snapshot.md',
                 'source-manifest.json', 'rubric.json']:
        (OUT / name).write_bytes((SAVED / name).read_bytes())
        manifest['input_files'][name] = sha(OUT / name)
    cases = json.loads((OUT / 'test-inputs.json').read_text())
    extra = json.loads((OUT / 'scope-clarification.json').read_text())
    metadata = api('/v1/models')
    (OUT / 'endpoint-models.json').write_text(json.dumps(metadata, indent=2) + '\n')
    models = metadata['data']
    assert len(models) == 1, 'More than one model; select the requested 27B explicitly.'
    model = models[0]['id']
    assert '27B' in model and 'Q4_K_M' in model, model
    manifest['model'] = model
    manifest['model_metadata'] = models[0]
    try:
        props = api('/props')
        (OUT / 'endpoint-props.json').write_text(json.dumps(props, indent=2) + '\n')
    except Exception as e:
        manifest['props_error'] = str(e)
    try:
        for task, prompts, limits in [
            ('marketing', [cases['marketing']], [cases['token_limits']['marketing']]),
            ('python', cases['python_stages'] + extra['prompts'],
             cases['token_limits']['python_stages'] + extra['max_tokens'])]:
            messages = [{'role': 'system', 'content': cases['system']}]
            for stage, (prompt, limit) in enumerate(zip(prompts, limits), 1):
                messages.append({'role': 'user', 'content': prompt})
                continuation = 0
                while True:
                    payload = {**cases['sampling'], 'min_p': 0.0, 'model': model, 'messages': list(messages),
                               'max_tokens': limit if continuation == 0 else cases['token_limits']['continuation'],
                               'stream': False}
                    suffix = '' if continuation == 0 else f'-continuation-{continuation}'
                    stem = f'qwen27b-{task}-{stage}{suffix}'
                    (OUT / f'{stem}-request.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
                    print(f'{task} stage {stage}{suffix}: started', flush=True)
                    started = time.monotonic()
                    response = api('/v1/chat/completions', payload)
                    wall = time.monotonic() - started
                    (OUT / f'{stem}-response.json').write_text(json.dumps(response, ensure_ascii=False, indent=2) + '\n')
                    choice = response['choices'][0]
                    text = choice['message'].get('content') or ''
                    reasoning = choice['message'].get('reasoning_content') or ''
                    (OUT / f'{stem}.md').write_text(text + '\n')
                    if reasoning:
                        (OUT / f'{stem}-reasoning.md').write_text(reasoning + '\n')
                    assert text.strip() or reasoning.strip(), 'Empty response'
                    manifest['runs'].append({'task': task, 'stage': stage,
                        'continuation': continuation, 'wall_s': round(wall, 3),
                        'finish_reason': choice['finish_reason'], 'timings': response.get('timings'),
                        'usage': response.get('usage'), 'reasoning_chars': len(reasoning),
                        'input_sha256': hashlib.sha256(messages[-1]['content'].encode()).hexdigest()})
                    (OUT / 'run-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
                    print(f'{task} stage {stage}{suffix}: {choice["finish_reason"]}, {len(text)} chars, {wall:.2f}s', flush=True)
                    messages.append({'role': 'assistant', 'content': text})
                    if choice['finish_reason'] != 'length':
                        assert text.strip(), 'No final content despite completed generation'
                        break
                    continuation += 1
                    messages.append({'role': 'user', 'content': cases['continuation_prompt']})
    finally:
        after = {f: sha(f) for f in watched}
        manifest['watched_hashes_after'] = after
        manifest['production_code_config_engine_vocab_unchanged'] = before == after
        (OUT / 'run-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        assert before == after
    print('Completed saved tasks; server left as loaded; production files unchanged.', flush=True)


if __name__ == '__main__':
    main()
