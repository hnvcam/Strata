"""Isolated local validation; production must already be stopped. No server/config writes.

Run with .venv/bin/python .../validate.py quality|integration|bench.
"""
import argparse
import json
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path[:0] = [str(ROOT / "tools/qwen35"), str(ROOT / "tools")]
from q36_harness import Eng, load_prompts
from strata_tokenizer import Tokenizer

MODEL = "/media/hnvcam/AI/LLAMA_Models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
installation = OUT / "installation.json"
original = Path(json.loads(installation.read_text())["backup"]) / "strata" if installation.exists() else ROOT / "engine/strata"
BASE = Path(os.environ.get("STRATA_BASELINE_EXE", str(original)))
EXE = ROOT / "build/strata"
PROMPTS = load_prompts(ROOT / "tools/qwen35/parity_prompts.ids")
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="0")


def tokenizer():
    d = ROOT / "packs/qwen36-ud-q4_k_xl/tokenizer"
    vocab = json.loads((d / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    return Tokenizer(tokens, (d / "merges.txt").read_text().split("\n"), json.loads((d / "token_type.json").read_text()))


def chat(tok, text):
    return tok.encode("<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)


def args(q8=True, chunk=2048, mtp=2):
    return ["--native", MODEL, "--max-context", "131072", "--prefill", str(chunk), "--mtp", str(mtp), "--pool-workers", "9"] + (["--kv", "int8"] if q8 else [])


def save(name, result):
    rendered = json.dumps(result, indent=2)
    rendered = re.sub(r"\[\s*(-?\d+(?:,\s*-?\d+)*)\s*\]",
                      lambda m: "[" + ", ".join(re.findall(r"-?\d+", m[1])) + "]", rendered)
    (OUT / (name + ".json")).write_text(rendered + "\n")
    summary = dict(result)
    if "ids" in summary:
        summary["ids"] = [f"{len(ids)} tokens" for ids in summary["ids"]]
    if "tokens" in summary:
        summary["tokens"] = [f"{len(ids)} tokens" for ids in summary["tokens"]]
    print(name, json.dumps(summary), flush=True)


def close(e):
    # EOF shuts down only this script's isolated engine, without signalling any other process.
    e.p.stdin.close()
    if e.p.wait(timeout=120) != 0:
        raise RuntimeError("test engine exited unsuccessfully")
    if hasattr(e.logf, "close"):
        e.logf.close()


def quality():
    # Fixed expert placement and native prefill isolate KV precision from MMQ and routing/cache effects.
    baseline = json.loads((OUT / "baseline-fp32.json").read_text())["tokens"]
    env = dict(ENV, STRATA_Q36_NO_MMQ="1", STRATA_Q36_FIXED_SLOTS="1")
    tok = tokenizer()
    text = "Explain briefly why a Python dictionary is useful. Then translate 'The train leaves at seven' into Vietnamese."
    prompt = chat(tok, text)
    # A shared, teacher-forced continuation means both engines predict the same histories.
    forced = tok.encode("A dictionary maps keys to values and supports fast lookup. Tiếng Việt: Tàu khởi hành lúc bảy giờ.")[:48]
    logits = {}
    for tag, q8 in (("quality-fp32", False), ("quality-q8", True)):
        dest = OUT / (tag + ".full")
        cmd = [str(EXE), *args(q8, mtp=0), "--expert-slots", "23", "--tokens", ",".join(map(str, prompt)),
               "--force", ",".join(map(str, forced)), "--dump-full-logits", str(dest)]
        with (OUT / (tag + ".log")).open("w") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=log, check=True, timeout=600)
        a = np.fromfile(dest, dtype=np.float32).reshape(len(forced) + 1, -1).astype(np.float64)
        assert np.isfinite(a).all()
        a -= a.max(axis=1, keepdims=True)
        a -= np.log(np.exp(a).sum(axis=1, keepdims=True))
        logits[tag] = a
    p, q = logits.values()
    kl = (np.exp(p) * (p - q)).sum(axis=1)
    save("quality", {"rows": len(p), "mean_kl": float(kl.mean()), "max_kl": float(kl.max()),
                     "top1_agreement": float((p.argmax(axis=1) == q.argmax(axis=1)).mean()), "kl_by_row": kl.tolist()})
    assert kl.mean() < .05 and kl.max() < .5, "unexpectedly large Q8 perturbation"
    # Confirm the opt-out path still returns the installed build's exact token IDs.
    e = Eng(str(EXE), args(False, mtp=0) + ["--expert-slots", "23"], env=env, log=str(OUT / "candidate-fp32.log"))
    try:
        actual = [e.solo(ids, 64)[0] for ids in PROMPTS]
        save("candidate-fp32", {"tokens": actual, "identical_prompts": sum(x == y for x, y in zip(actual, baseline))})
        assert actual == baseline, "FP32 output changed"
    finally:
        close(e)


def integration():
    tok = tokenizer()
    env = dict(ENV, STRATA_Q36_FIXED_SLOTS="1")
    prompt = chat(tok, "Return only the result of 17 * 23 + 144 / 12.")
    if (OUT / "arithmetic-baseline.json").exists():
        baseline_answer = json.loads((OUT / "arithmetic-baseline.json").read_text())["answer"]
    else:
        baseline = Eng(str(BASE), args(False) + ["--expert-slots", "23"], env=env, log=str(OUT / "arithmetic-baseline.log"))
        try:
            baseline_answer = tok.decode(baseline.solo(prompt, 64)[0])
            save("arithmetic-baseline", {"answer": baseline_answer, "expected": "403"})
        finally:
            close(baseline)
    # Batch tables, speculative verification and copying the solo session into another slot.
    e = Eng(str(EXE), args(True) + ["--batch", "2", "--batch-mtp", "--expert-slots", "23"], env=env, log=str(OUT / "integration.log"))
    try:
        solo = [e.solo(ids, 64)[0] for ids in PROMPTS[:3]]
        batch = e.batch([{"ids": ids, "n": 64, "slot": i} for i, ids in enumerate(PROMPTS[:2])])
        assert batch["tokens"] == solo[:2], "batch differs from Q8 solo/MTP"
        # Repeated prompts, a copied held prefix and a continuation through a different batch slot.
        prefix = PROMPTS[2] + solo[2]
        continuation = prefix + [13]
        expected = e.solo(continuation, 32)[0]
        e.solo(prefix, 1)  # leave exactly the prefix held, without speculative lookahead
        seed_slot = int(re.findall(r"strata serve: session (\d+):", (OUT / "integration.log").read_text())[-1])
        target = 1 - seed_slot
        copied = e.batch([{"ids": continuation, "n": 32, "slot": target}])["tokens"][0]
        assert copied == expected, "copied Q8 conversation differs"
        assert re.search(rf"strata batch: slot {target} takes a prompt of {len(continuation)} tokens = {len(prefix)} reused", (OUT / "integration.log").read_text()), "test did not exercise session copying"
        answer = tok.decode(e.solo(prompt, 64)[0])
        assert answer == baseline_answer, (baseline_answer, answer)
        simple = tok.decode(e.solo(chat(tok, "Return only the result of 2 + 2."), 32)[0])
        assert simple.strip().removesuffix("<|im_end|>").strip() == "4", simple
        filler = "The following is ordinary reference text about trains, weather, gardens and software. " * 1200
        long_prompt = chat(tok, "Remember this code: STRATA-73921.\n" + filler + "\nWhat code was given at the beginning? Return only that code.")
        answer_long = tok.decode(e.solo(long_prompt, 64)[0])
        assert "STRATA-73921" in answer_long, answer_long
        save("integration", {"batch_matches_solo": True, "copied_conversation_matches": True,
                             "arithmetic_baseline": baseline_answer, "arithmetic_q8": answer,
                             "arithmetic_expected": "403", "simple_arithmetic": simple,
                             "retrieval_prompt_tokens": len(long_prompt), "retrieval": answer_long})
    finally:
        close(e)


def bench():
    tok = tokenizer()
    # Same saved prompts and sampling for all cases; no prefix reuse in timed reads.
    documents = []
    for n in (4096, 16384):
        body = tok.encode("Reference note: stable cache storage lets a language model retain earlier attention keys and values.\n" * 2000)
        tail = chat(tok, "Reply with OK.")
        documents.append(body[:n - len(tail)] + tail)
    save("bench-prompts", {"ids": documents})
    cases = [("installed-fp32-2k", BASE, False, 2048), ("q8-mmq-2k", EXE, True, 2048), ("q8-mmq-4k", EXE, True, 4096)]
    results = {}
    for tag, exe, q8, chunk in cases:
        e = Eng(str(exe), args(q8, chunk), env=ENV, log=str(OUT / (tag + ".log")))
        try:
            e.solo(PROMPTS[1], 64, "temperature=0.7 seed=42")
            rates = []
            for r in range(2):
                # Distinct first token invalidates prefix reuse without changing document length.
                for ids in documents:
                    if r and len(ids) == 16384:
                        continue  # one long-read smoke timing; two 4K reads
                    inp = [1000 + r] + ids[1:]
                    _, done, _, _ = e.solo(inp, 1)
                    f = done.split()
                    assert int(f[8]) == 0
                    rates.append({"tokens": len(inp), "prefill_tps": len(inp) * 1000 / float(f[3])})
                    print(tag, rates[-1], flush=True)
            decode = []
            for r in range(3):
                out, done, _, _ = e.solo(PROMPTS[1], 256, "temperature=0.7 seed=" + str(100 + r))
                f = done.split()
                decode.append({"tokens": len(out), "decode_tps": len(out) * 1000 / float(f[4]), "done": done})
                print(tag, decode[-1], flush=True)
            results[tag] = {"prefill": rates, "decode": decode,
                            "prefill_median": {str(n): statistics.median(x["prefill_tps"] for x in rates if x["tokens"] == n) for n in (4096, 16384)},
                            "decode_median": statistics.median(x["decode_tps"] for x in decode)}
            save("bench", results)
        finally:
            close(e)


def fresh_decode():
    # Separate from the long, unrelated prefill document: it can dominate the adaptive expert scores.
    results = {}
    for tag, exe, q8, chunk in [("installed-fp32-2k", BASE, False, 2048),
                                ("q8-mmq-2k", EXE, True, 2048), ("q8-mmq-4k", EXE, True, 4096)]:
        e = Eng(str(exe), args(q8, chunk), env=ENV, log=str(OUT / (tag + "-fresh-decode.log")))
        try:
            e.solo(PROMPTS[1], 128, "temperature=0.7 seed=42")
            rates = []
            for r in range(3):
                out, done, _, _ = e.solo(PROMPTS[1], 256, "temperature=0.7 seed=" + str(100 + r))
                f = done.split()
                rates.append({"tokens": len(out), "decode_tps": len(out) * 1000 / float(f[4]), "done": done})
            results[tag] = {"runs": rates, "median": statistics.median(x["decode_tps"] for x in rates)}
            save("fresh-decode", results)
        finally:
            close(e)


def candidate_smoke():
    tok = tokenizer()
    cfg = json.loads((ROOT / "strata-qwen36-ud-q4_k_xl.json").read_text())
    exe = str((ROOT / cfg["exe"]).resolve())
    e = Eng(exe, cfg["args"], env=ENV, log=str(OUT / "candidate-smoke.log"))
    try:
        filler = "The following is ordinary reference text about trains, weather, gardens and software. " * 1200
        prompt = chat(tok, "Remember this code: STRATA-73921.\n" + filler + "\nWhat code was given at the beginning? Return only that code.")
        ids, done, _, _ = e.solo(prompt, 64)
        answer = tok.decode(ids)
        assert "STRATA-73921" in answer, answer
        save("candidate-smoke", {"exe": cfg["exe"], "args": cfg["args"], "prompt_tokens": len(prompt), "answer": answer, "done": done})
    finally:
        close(e)


if __name__ == "__main__":
    mode = argparse.ArgumentParser()
    mode.add_argument("mode", choices=["quality", "integration", "bench", "fresh_decode", "candidate_smoke"])
    selected = mode.parse_args().mode
    # Refuse to compete with an engine somebody started meanwhile.
    live = subprocess.run(["pgrep", "-x", "strata"], capture_output=True, text=True)
    if live.returncode == 0:
        raise SystemExit("An engine is already running; stop it yourself before isolated validation.")
    globals()[selected]()
