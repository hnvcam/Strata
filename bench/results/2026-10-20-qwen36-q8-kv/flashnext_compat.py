"""Compare installed/candidate Flash-Next on the saved dual-GPU IQ4_XS config, without profile/cache writes."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path[:0] = [str(ROOT / "tools/qwen35"), str(ROOT / "tools")]
from q36_harness import Eng
from strata_tokenizer import Tokenizer


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if subprocess.run(["pgrep", "-x", "strata"], capture_output=True).returncode == 0:
        raise SystemExit("An engine is already running; leave it in the user's control.")
    cfg = json.loads((ROOT / "strata-iq4_xs.json").read_text())
    args = list(cfg["args"])
    for flag in ("--expert-profile-save", "--expert-profile-save-every"):
        i = args.index(flag)
        del args[i:i + 2]
    args += ["--layer-split", str(cfg["layer_split"]), "--prompt-cache", "0", "--conversation-cache-mib", "0"]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_", "GGML_"))}
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(map(str, cfg["gpu"])),
               LD_LIBRARY_PATH=":".join(cfg["lib_dirs"] + [env.get("LD_LIBRARY_PATH", "")]))
    d = Path(cfg["tokenizer"])
    vocab = json.loads((d / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    tok = Tokenizer(tokens, (d / "merges.txt").read_text().split("\n"), json.loads((d / "token_type.json").read_text()))
    texts = ["Reply with the word ready.", "Explain how a refrigerator works in two sentences.",
             "Write a Python function that returns the median of a nonempty list, followed by two assert tests."]
    cases = [{"name": f"short-{i}", "n": 64, "ids": tok.encode("<|im_start|>user\n" + s +
              "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)} for i, s in enumerate(texts)]
    retrieval = json.loads((ROOT / "bench/results/2026-10-19-iq4-xs-prefill-chunks/prompt-4k.json").read_text())
    cases.append({"name": "retrieval-4k", "n": 24, "ids": retrieval["ids"], "expected": retrieval["expected"]})
    protected = [ROOT / "strata-iq4_xs.json", ROOT / "run-iq4_xs.sh",
                 Path(args[args.index("--expert-profile") + 1]), Path(args[args.index("--mtp") + 1]) / "draft_vocab.bin"]
    before = {str(p): digest(p) for p in protected}
    recorded_cases = [dict(c) for c in cases]
    recorded_cases[-1].pop("ids")
    recorded_cases[-1].update(fixture="bench/results/2026-10-19-iq4-xs-prefill-chunks/prompt-4k.json",
                              fixture_sha256=digest(ROOT / "bench/results/2026-10-19-iq4-xs-prefill-chunks/prompt-4k.json"))
    result = {"args": args, "gpu_order": cfg["gpu"], "protected_before": before, "cases": recorded_cases, "engines": {}}
    installation = OUT / "installation.json"
    original = Path(json.loads(installation.read_text())["backup"]) / "strata" if installation.exists() else ROOT / "engine/strata"
    baseline = Path(os.environ.get("STRATA_BASELINE_EXE", str(original)))
    for name, exe in (("baseline", baseline), ("candidate", ROOT / "engine/candidates/qwen36-q8-kv/strata")):
        print(name, "loading", flush=True)
        e = Eng(str(exe), args, env=env, log=str(OUT / ("flashnext-" + name + ".log")))
        entry = {"exe": str(exe), "sha256": digest(exe), "ready": e.ready, "info": e.info, "requests": []}
        try:
            print(name, e.ready, "slots", e.info.get("expert_slots"), flush=True)
            for case in cases:
                ids, done, _, _ = e.solo(case["ids"], case["n"], "ckpt=0")
                text = tok.decode(ids)
                if "expected" in case:
                    assert case["expected"] in text, text
                entry["requests"].append({"name": case["name"], "tokens": ids, "text": text, "done": done})
                print(name, case["name"], len(ids), "tokens", done, flush=True)
        finally:
            e.p.stdin.close()
            assert e.p.wait(timeout=120) == 0
            e.logf.close()
        result["engines"][name] = entry
        (OUT / "flashnext-compat.json").write_text(json.dumps(result, indent=2) + "\n")
    a, b = result["engines"].values()
    for key in ("context", "kv", "kv_resident", "expert_slots", "expert_slots_primary", "spec", "mtp_max", "prefill_cap"):
        assert a["info"][key] == b["info"][key], (key, a["info"][key], b["info"][key])
    result["identical_requests"] = [x["tokens"] == y["tokens"] for x, y in zip(a["requests"], b["requests"])]
    result["protected_after"] = {str(p): digest(p) for p in protected}
    assert before == result["protected_after"], "saved Flash-Next files changed"
    assert all(result["identical_requests"]), "Flash-Next token parity failed"
    result["passed"] = True
    (OUT / "flashnext-compat.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Flash-Next compatibility PASS: all 4 requests identical; saved files unchanged", flush=True)


if __name__ == "__main__":
    main()
