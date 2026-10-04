"""Write the final report from verified timings and evidence-based reviews."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def load(name):
    return json.loads((OUT / name).read_text())


def main():
    summary = load('summary.json')
    manifest = load('run-manifest.json')
    verification = load('comparison-verification.json')
    marketing = load('marketing-review.json')
    technical = {name: load(f'{name}-python-review.json') for name in ('helper', 'four-layers')}
    rubric = load('rubric.json')
    scores = {'note': 'Subjective reviewer scores for the completed article and regenerated stage-5 proposal, using the saved rubric. No combined task score.',
              'marketing': {name: marketing[name]['scores'] for name in technical},
              'python': {name: technical[name]['scores'] for name in technical}}
    for task in ('marketing', 'python'):
        for name, categories in scores[task].items():
            assert set(categories) == set(rubric[task])
            assert all(0 <= value <= rubric[task][key] for key, value in categories.items())
            expected = marketing[name]['total'] if task == 'marketing' else technical[name]['total']
            assert sum(categories.values()) == expected
    (OUT / 'scores.json').write_text(json.dumps(scores, indent=2) + '\n')
    lines = ['# Final comparison: 4060 expert helper versus fully cached layers 0–3', '',
        'The helper produced the better marketing article in this sample and decoded the Python sequence 2.3% faster. '
        'The four-layer split read the identical initial source prompt 12.8% faster and finished the matched Python '
        'sequence 16.2% sooner, while producing 16.5% fewer output tokens. Its final proposal handled activation '
        'and scratch buffers better, but both proposals contain blocking omissions. These results do not establish '
        'one placement as the best for every workload.', '',
        'Measured 2026-10-04 on the Intel i5-13500, 64 GB RAM, RTX 5070 Ti 16 GB and RTX 4060 8 GB, Linux. '
        'Only the two requested placements were run. Both used the current patched build/strata executable, '
        'SHA-256 ' + manifest['hashes_before'][manifest['configs']['helper']['exe']] + '.', '',
        '## Confirmed placements', '',
        '| Setup | RTX 5070 Ti | RTX 4060 |', '|---|---|---|',
        '| Expert helper | All 48 layers, output head, Q2 MTP; 2513 cached experts (5.66 GiB) | Computes 2800 additional experts (6.31 GiB), returns results through pinned host rows |',
        '| Four-layer split | Layers 4–47, output head, Q2 MTP; 2939 cached experts (6.58 GiB) | Layers 0–3; all 2048 experts cached (4.99 GiB) |', '',
        'Startup logs confirmed these placements and full 248320-token draft heads on every server start. '
        'Main-model experts absent from the GPU caches use the CPU/RAM path. Prompt buffers can temporarily borrow '
        'stage cache slots, which the engine refills before decode.', '',
        '## Saved tasks and matched settings', '',
        'The exact LoopDay marketing brief, five Python proposal/review prompts, source snapshot and rubric were copied '
        'from ../2026-10-03-iq3-s-vs-iq4-xs without changing their bytes. The same material was also used in the saved 27B comparison. '
        'The Python task is a Q4_K_M MTP import/runtime proposal based on supplied source; it is not an executable unit-test suite. '
        'Only the final corrected proposal is scored. No proposed implementation was applied.', '',
        'IQ4_XS, installed Q2 MTP with all 248320 draft IDs, 131072 context capacity, int8 KV, prefill=512, '
        'spec=4, cutoff=.70, 1248 MiB reserve, PCIe fraction=.20, suffix drafting disabled. '
        'Sampling: temperature=.2, top_p=.95, top_k=20, seed=1234; thinking disabled. '
        'No additional warmup or benchmark prompts. The system/user messages and named settings match between placements; '
        'each conversation retains its own assistant answers.', '',
        'The helper stage-2 review reached 4096 tokens. The saved paired-continuation rule required an identical '
        'continuation for both placements; the four-layer first continuation then reached 2048 tokens. Both received a '
        'second identical continuation with a 6144-token budget. This second budget is the only output-limit change. '
        'Both continuation turns read cold histories. The helper stages 3–5 produced before the second continuation '
        'were archived in superseded-one-continuation and regenerated with the matched external sequence. '
        'All final deliverables completed. The 16 matched requests include both continuations; the three superseded '
        'helper requests are excluded from task rates and times. [Full adjustment record](test-adjustments.json).', '',
        '## Quality results', '',
        '| Task | Expert helper | Four-layer split |', '|---|---:|---:|',
        f'| Marketing article | {marketing["helper"]["total"]}/100; {marketing["helper"]["word_count"]} words | {marketing["four-layers"]["total"]}/100; {marketing["four-layers"]["word_count"]} words |',
        f'| Corrected Python proposal | {technical["helper"]["total"]}/100 | {technical["four-layers"]["total"]}/100 |', '',
        'These are subjective rubric scores for one sampled article and one technical conversation per placement. '
        'They are not general quality rankings. The concrete errors below carry more weight than a small score difference.', '',
        '### Marketing', '',
        'The helper article follows the brief more closely: 771 words meets the 600–800 requirement, and it includes '
        'the account/internet requirements for optional sync. The four-layer article is 974 words, omits those sync '
        'requirements, and adds unprovided notification/weekly-review behavior. Both correctly state the prices, Free '
        'project/task restrictions, platforms, calendar/AI limitations and download CTA. Both use familiar productivity '
        'phrases and feature-oriented sections. [Category scores and evidence](marketing-review.json).', '',
        '### Corrected technical proposals', '']
    for name, label in (('helper', 'Expert helper'), ('four-layers', 'Four-layer split')):
        review = technical[name]
        lines += [f'**{label}: {review["total"]}/100.**', '', *[f'- {s}' for s in review['strengths']], '', 'Remaining issues:', '']
        for finding in review['blocking_findings']:
            lines.append(f'- **{finding["issue"]}:** {finding["response_evidence"]} {finding["reference_fact"]}')
        lines += ['', review['review_history'], '']
    lines += ['Both proposals need corrections before implementation. Exact reference facts: Q8_1 is 36 bytes per 32 values; '
              'Q8_0 is 34. Native expert stride is 3584000 bytes, with 1835008000 bytes for 512 experts (1750 MiB), '
              '1075 MiB above legacy Q2 expert storage. Decoded eh_proj is 2560 output rows by 5120 input columns, split '
              'into embedding and hidden input columns per row. The full draft head, KV/work buffers, target cache and '
              'reserve require additional memory. The Q2 draft layer in these runs logged 949 MiB plus a separate '
              '497.3 MiB full draft head. [Saved reference facts](review-facts.json), [source snapshot](source-snapshot.md).', '',
              '## Observed performance', '',
              '| Setup / task | Output tokens | Decode tok/s | Prefill seconds | Request wall seconds | MTP acceptance |',
              '|---|---:|---:|---:|---:|---:|']
    for name in technical:
        for task in ('marketing', 'python'):
            t = summary['setups'][name][task]
            lines.append(f'| {name} / {task} | {t["output_tokens"]} | {t["decode_tok_s"]:.2f} | '
                         f'{t["prefill_s"]:.2f} | {t["wall_s"]:.2f} | {t["draft_acceptance"] * 100:.2f}% |')
    lines += ['', 'The first Python request uses identical cold input in both placements:', '',
              '| Setup | Input tokens | Prefill tok/s | Prefill seconds | Decode tok/s | Request wall seconds |',
              '|---|---:|---:|---:|---:|---:|']
    for name in technical:
        r = next(r for p in manifest['phases'] for r in p['runs'] if r['setup'] == name and r['task'] == 'python' and r['stage'] == 1)
        t = r['timings']
        lines.append(f'| {name} | {t["prompt_n"]} | {t["prompt_per_second"]:.1f} | '
                     f'{t["prompt_ms"] / 1000:.2f} | {t["predicted_per_second"]:.1f} | {r["wall_s"]:.2f} |')
    lines += ['', 'Decode rates divide total output tokens by summed generation time; warmup is absent. '
              'Prefill rates count only tokens actually read, and reused tokens are reported separately. '
              'Request wall times include prefill and generation, excluding startup. Different answer lengths and '
              'assistant histories make task wall times observed outcomes rather than equal-output speed comparisons. '
              'The continuation restarts and original cold scope-review restart also add source-reading work. '
              'This is one run per placement, not a full 128K test or a significance claim. '
              '[Every request, prefill rate, reused token count and startup/elapsed time](timing.md).', '',
              '## State and artifacts', '',
              'Production config, original and patched engines, expert profile, full draft vocabulary and watched source '
              'files were unchanged by hash. The temporary servers stopped and production remains unloaded.', '',
              '```text', verification['gpu_after'].strip(), '```', '',
              '[Verification](comparison-verification.json), [timings and counters](summary.json), '
              '[full run manifest including superseded work](run-manifest.json), [rubric scores](scores.json). '
              'Completed articles: [helper](helper-marketing-1.md), [four layers](four-layers-marketing-1.md). '
              'Scored final proposals: [helper](helper-python-5.md), [four layers](four-layers-python-5.md). '
              'Exact request and response JSON files are saved beside each output.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('Wrote REPORT.md and scores.json')


if __name__ == '__main__':
    main()
