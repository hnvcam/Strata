"""Run the upstream ten greedy parity prompts after the user stops production.

No running process is stopped, no binary is replaced, and no live config is edited.
Uses the saved Qwen3.6 runtime config, overriding only context, workers and KV format.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "tools/qwen35"))
from q36_harness import Eng, load_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--workers", type=int, default=13,
                        help="pool workers; the host adds one CPU thread")
    parser.add_argument("--kv", choices=("fp32", "int8"), default="int8")
    args = parser.parse_args()
    if args.context < 256 or args.workers < 1:
        parser.error("context must be >= 256 and workers >= 1")
    active = subprocess.run(["pgrep", "-a", "-x", "strata"], capture_output=True, text=True)
    if active.returncode == 0:
        raise SystemExit("Existing engine is running; ask the user to stop it first.\n" + active.stdout)
    if active.returncode != 1:
        raise SystemExit("Could not check running engines; refusing to start.")
    cfg = json.loads((ROOT / "strata-qwen36-ud-q4_k_xl.json").read_text())
    exe = (ROOT / cfg["exe"]).resolve()
    engine_args = list(cfg["args"])
    engine_args[engine_args.index("--max-context") + 1] = str(args.context)
    engine_args[engine_args.index("--pool-workers") + 1] = str(args.workers)
    if "--kv" in engine_args:
        position = engine_args.index("--kv")
        del engine_args[position:position + 2]
    engine_args += ["--kv", args.kv]
    prompts_file = ROOT / "tools/qwen35/parity_prompts.ids"
    prompts = load_prompts(prompts_file)
    tag = f"ctx{args.context}-{args.kv}-workers{args.workers}-{time.time_ns()}"
    destination = OUT / f"{tag}.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    result = {
        "exe": str(exe), "exe_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
        "args": engine_args,
        "prompts_sha256": hashlib.sha256(prompts_file.read_bytes()).hexdigest(),
        "method": "Ten upstream prompt-ID lists, once each, greedy, up to 128 output tokens; normal engine warmup.",
        "context_note": "Context is allocated capacity; these are short upstream prompts, not a filled history.",
        "requests": [],
    }
    engine = Eng(str(exe), engine_args, env=env, log=str(OUT / f"{tag}.log"))
    result["ready"] = engine.ready
    result["info"] = engine.info
    try:
        for index, ids in enumerate(prompts):
            tokens, done, ttft, wall = engine.solo(ids, 128)
            fields = done.split()
            count, prompt_ms, decode_ms = int(fields[1]), float(fields[3]), float(fields[4])
            row = {"prompt": index, "output_tokens": count, "prompt_ms": prompt_ms,
                   "decode_ms": decode_ms, "decode_tok_s": 1000 * count / decode_ms if decode_ms else 0,
                   "drafts_accepted": int(fields[6]), "drafts_offered": int(fields[7]),
                   "ttft_s": ttft, "wall_s": wall, "done": done, "tokens": tokens}
            result["requests"].append(row)
            destination.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k != "tokens"}), flush=True)
    finally:
        # EOF exits only the benchmark process this script created. Never signal other engines.
        engine.p.stdin.close()
        exit_code = engine.p.wait(timeout=60)
        engine.logf.close()
    if exit_code:
        raise SystemExit(f"Benchmark engine exited with {exit_code}; see its log.")
    rows = result["requests"]
    offered = sum(r["drafts_offered"] for r in rows)
    result["summary"] = {
        "requests": len(rows), "tokens": sum(r["output_tokens"] for r in rows),
        "median_request_tok_s": statistics.median(r["decode_tok_s"] for r in rows),
        "aggregate_decode_tok_s": 1000 * sum(r["output_tokens"] for r in rows) / sum(r["decode_ms"] for r in rows),
        "drafts_accepted": sum(r["drafts_accepted"] for r in rows), "drafts_offered": offered,
    }
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(str(destination), json.dumps(result["summary"]), flush=True)
    if not offered:
        raise SystemExit("Unexpected: greedy MTP benchmark produced no drafts.")


if __name__ == "__main__":
    main()
