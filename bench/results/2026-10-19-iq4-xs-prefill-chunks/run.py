#!/usr/bin/env python3
"""Isolated prefill measurements; engines exit normally when their input reaches EOF.

Does not start/stop the user's server or edit its config, engine, profiles, or models.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gpu():
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.used,memory.free,temperature.gpu,power.draw",
        "--format=csv"], text=True).strip()


def memory():
    data = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree"):
            data[key + "_kib"] = int(value.split()[0])
    return data


def fixtures(cfg):
    sys.path.insert(0, str(ROOT / "tools"))
    from strata_tokenizer import Tokenizer
    folder = Path(cfg["tokenizer"])
    vocab = json.loads((folder / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, i in vocab.items():
        tokens[i] = token
    tok = Tokenizer(tokens, (folder / "merges.txt").read_text().split("\n"),
                    json.loads((folder / "token_type.json").read_text()))
    enc = lambda s: tok.encode(s, parse_special=True)
    samples = []
    for name in ("tools/strata_bench.py", "serve/server.py", "src/prefill/prefill.cpp"):
        samples.extend(enc(f"\nRepository excerpt: {name}\n" + (ROOT / name).read_text()))
    cases = []
    for n in (4096, 16384):
        marker = "cobalt-714" if n == 4096 else "amber-926"
        head = enc("<|im_start|>user\nRead these repository excerpts. A record inside them specifies a passphrase. "
                   "After reading, return only that passphrase.\n\n")
        needle = enc(f"\n\nIMPORTANT RECORD: The passphrase is {marker}.\n\n")
        tail = enc("\n\nWhat is the passphrase in the IMPORTANT RECORD? Return only the passphrase, with no explanation."
                   "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        body_n = n - len(head) - len(needle) - len(tail)
        left = body_n // 2
        ids = head + samples[:left] + needle + samples[left:body_n] + tail
        assert len(ids) == n
        case = {"tag": f"{n // 1024}k", "ids": ids, "expected": marker, "max_new": 24}
        save(OUT / f"prompt-{n // 1024}k.json", case)
        cases.append(case)
    warm = {"tag": "warmup", "ids": enc("<|im_start|>user\nReply with the word ready.<|im_end|>\n"
                                            "<|im_start|>assistant\n<think>\n\n</think>\n\n"), "max_new": 8}
    return tok, warm, cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--ring", type=int)
    ap.add_argument("--reserve")
    ap.add_argument("--ram-budget", type=float)
    ap.add_argument("--warm-prefill", action="store_true")
    ap.add_argument("--decode-check", action="store_true")
    ap.add_argument("--decode-runs", type=int, default=0)
    ap.add_argument("--decode-max", type=int, default=512)
    ap.add_argument("--sizes", default="4k,16k")
    ap.add_argument("--require-chunk", action="store_true",
                    help="record fallback and exit normally if the requested chunk does not fit")
    a = ap.parse_args()
    active = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if (entry / "comm").read_text().strip() == "strata":
                active.append(int(entry.name))
        except (OSError, ProcessLookupError):
            pass
    if active:
        raise SystemExit(f"A Strata engine is already running (PIDs {active}). Leave it in the user's control; "
                         "run this benchmark only after the user has stopped it.")
    cfg = json.loads((ROOT / "strata-iq4_xs.json").read_text())
    tok, warm, cases = fixtures(cfg)
    cases = [c for c in cases if c["tag"] in a.sizes.split(",")]
    args = list(cfg["args"])
    for key in ("--expert-profile-save", "--expert-profile-save-every"):
        if key in args:
            i = args.index(key)
            del args[i:i + 2]
    args[args.index("--prefill") + 1] = str(a.chunk)
    if a.reserve:
        args[args.index("--vram-reserve-mib") + 1] = a.reserve
    if a.ram_budget:
        if "--resident-budget-gib" in args:
            args[args.index("--resident-budget-gib") + 1] = str(a.ram_budget)
        else:
            args += ["--resident-budget-gib", str(a.ram_budget)]
    args += ["--layer-split", str(cfg["layer_split"]), "--prompt-cache", "0", "--conversation-cache-mib", "0"]
    env = dict(os.environ)
    # Do not inherit unrelated experimental tuning from the invoking shell.
    inherited = {k: v for k, v in env.items() if k.startswith(("STRATA_", "GGML_"))}
    for key in inherited:
        env.pop(key)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg["gpu"]))
    env["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"] + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    if a.ring is not None:
        env["STRATA_SPLIT_RING"] = str(a.ring)
    protected = [ROOT / "strata-iq4_xs.json", Path(cfg["exe"]),
                 Path(args[args.index("--expert-profile") + 1]),
                 Path(args[args.index("--mtp") + 1]) / "draft_vocab.bin"]
    before = {str(p): digest(p) for p in protected}
    result = {"tag": a.tag, "asked_chunk": a.chunk, "args": args,
              "gpu_order": cfg["gpu"], "ring": a.ring, "ram_budget_gib": a.ram_budget, "gpu_before": gpu(),
              "protected_before": before, "inherited_tuning_removed": sorted(inherited),
              "cache_policy": "no disk cache; prompt-cache=0; conversation-cache=0; ckpt=0",
              "started": time.strftime("%Y-%m-%d %H:%M:%S %z"), "requests": []}
    result["memory_before"] = memory()
    save(OUT / f"{a.tag}.json", result)
    q = queue.Queue()
    log = open(OUT / f"{a.tag}.engine.log", "w")
    proc = subprocess.Popen([cfg["exe"], "--serve", *args], cwd=ROOT, env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                            text=True, bufsize=1)
    result["pid"] = proc.pid
    print(f"{a.tag}: isolated engine PID {proc.pid}, chunk {a.chunk}, ring {a.ring}", flush=True)

    def pump():
        with open(OUT / f"{a.tag}.protocol.log", "w") as protocol:
            for line in proc.stdout:
                protocol.write(line)
                protocol.flush()
                q.put(line.strip())
        q.put(None)

    threading.Thread(target=pump, daemon=True).start()
    last_report = time.monotonic()

    def line(timeout=1800):
        nonlocal last_report
        start = time.monotonic()
        while True:
            try:
                value = q.get(timeout=2)
                if value is None:
                    raise RuntimeError(f"engine exited: {proc.poll()}; see {a.tag}.engine.log")
                return value
            except queue.Empty:
                if time.monotonic() - last_report >= 30:
                    last_report = time.monotonic()
                    tail = (OUT / f"{a.tag}.engine.log").read_text(errors="replace").splitlines()
                    print(f"{a.tag}: waiting; {tail[-1] if tail else 'loading'}", flush=True)
                if time.monotonic() - start > timeout:
                    raise TimeoutError("engine produced no protocol line before deadline; no signal sent")

    try:
        start = time.monotonic()
        while True:
            value = line()
            if value.startswith("INFO "):
                result["info"] = dict(x.split("=", 1) for x in value.split()[1:] if "=" in x)
            if value.startswith("READY"):
                result["ready"] = value
                result["load_s"] = time.monotonic() - start
                result["gpu_ready"] = gpu()
                result["memory_ready"] = memory()
                print(f"{a.tag}: READY in {result['load_s']:.1f}s; info={result.get('info', {})}", flush=True)
                save(OUT / f"{a.tag}.json", result)
                break
        actual_chunk = int(result.get("info", {}).get("prefill_cap", 0))
        if a.require_chunk and actual_chunk != a.chunk:
            result["status"] = "fallback"
            result["reason"] = f"requested {a.chunk}, engine resolved {actual_chunk}"
            print(f"{a.tag}: {result['reason']}; no timing attributed to the requested size", flush=True)
            return
        prefill_warm = [dict(cases[0], tag="warmup-prefill")] if a.warm_prefill and cases else []
        check = []
        if a.decode_check:
            check = [{"tag": "decode-check", "max_new": 128,
                      "ids": tok.encode("<|im_start|>user\nWrite a Python function that returns the median of "
                                        "a nonempty list of numbers. Explain how it handles an even length."
                                        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                                        parse_special=True)}]
        decode_case = {"tag": "decode", "max_new": a.decode_max,
                       "ids": tok.encode("<|im_start|>user\nWrite a Python function that returns the median of "
                                         "a nonempty list of numbers, followed by two assert tests. Explain in "
                                         "100 words how it handles an even length."
                                         "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                                         parse_special=True)}
        decode_warm = [dict(decode_case, tag="warmup-decode", max_new=128)] if a.decode_runs else []
        for case in [warm] + prefill_warm + cases * a.runs + check + decode_warm + [decode_case] * a.decode_runs:
            start = time.monotonic()
            proc.stdin.write(f"GEN {case['max_new']} ckpt=0 " + ",".join(map(str, case["ids"])) + "\n")
            proc.stdin.flush()
            tokens, progress = [], []
            first_s = None
            while True:
                value = line()
                if value.startswith("T "):
                    if first_s is None:
                        first_s = time.monotonic() - start
                    tokens.append(int(value.split()[1]))
                elif value.startswith("PP "):
                    progress.append(value)
                elif value.startswith("ERR") or value.startswith("FATAL"):
                    raise RuntimeError(value)
                elif value.startswith("DONE "):
                    f = value.split()
                    row = {"case": case["tag"], "n_input": len(case["ids"]), "done": value,
                           "generated": int(f[1]), "prompt_tokens": int(f[2]), "prompt_ms": float(f[3]),
                           "decode_ms": float(f[4]), "finish": f[5], "reused": int(f[8]),
                           "prompt_read": int(f[14]), "wall_s": time.monotonic() - start,
                           "first_s": first_s, "text": tok.decode(tokens),
                           "tokens": tokens, "progress": progress}
                    row["memory_after"] = memory()
                    row.update({k: int(v) for x in f[16:] if "=" in x for k, v in [x.split("=", 1)]
                                if k in ("chunk", "experts_streamed")})
                    row["prefill_tps"] = row["prompt_read"] * 1000 / row["prompt_ms"] if row["prompt_ms"] else None
                    row["decode_tps"] = row["generated"] * 1000 / row["decode_ms"] if row["decode_ms"] else None
                    if "expected" in case:
                        row["expected"] = case["expected"]
                        row["needle_ok"] = case["expected"] in row["text"]
                        assert row["reused"] == 0, "unexpected prompt reuse"
                    result["requests"].append(row)
                    save(OUT / f"{a.tag}.json", result)
                    print(f"{a.tag}: {case['tag']} {row['prefill_tps']:.1f} tok/s; "
                          f"decode={row['decode_tps']:.1f} tok/s ({row['generated']} tokens); "
                          f"chunk={row.get('chunk')} first={first_s:.2f}s needle={row.get('needle_ok')} "
                          f"text={row['text']!r}", flush=True)
                    break
        result["status"] = "complete"
    except BaseException as e:
        result["status"] = "error"
        result["error"] = repr(e)
        raise
    finally:
        # End only this test process's finite input stream. No terminate/kill/restart.
        proc.stdin.close()
        try:
            result["exit_code"] = proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            result["exit_pending_pid"] = proc.pid
        log.close()
        result["gpu_after"] = gpu()
        result["memory_after"] = memory()
        result["protected_after"] = {str(p): digest(p) for p in protected}
        result["protected_unchanged"] = result["protected_after"] == before
        save(OUT / f"{a.tag}.json", result)
        print(f"{a.tag}: exit={result.get('exit_code')}; protected unchanged={result['protected_unchanged']}", flush=True)


if __name__ == "__main__":
    main()
