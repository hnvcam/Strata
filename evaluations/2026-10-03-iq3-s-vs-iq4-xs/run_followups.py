"""Same two extra proposal/review turns for both models; no benchmark suites."""
import hashlib
import json
import subprocess
import time

from run_tasks import ROOT, OUT, api, sha, wait_ready, stop


def main():
    cases = json.loads((OUT / "test-inputs.json").read_text())
    extra = json.loads((OUT / "scope-clarification.json").read_text())
    original_manifest = json.loads((OUT / "run-manifest.json").read_text())
    watched = original_manifest["production_hashes_before"]
    assert all(sha(p) == h for p, h in watched.items())
    assert not api(1111, "/health")["loaded"]
    manifest = {"clarification_sha256": sha(OUT / "scope-clarification.json"), "runs": []}
    proc = None
    try:
        for name, cfg in original_manifest["configs"].items():
            messages = [{"role": "system", "content": cases["system"]}]
            for stage, prompt in enumerate(cases["python_stages"], 1):
                response = json.loads((OUT / f"{name}-python-{stage}-response.json").read_text())
                messages.extend([{"role": "user", "content": prompt},
                    {"role": "assistant", "content": response["choices"][0]["message"]["content"]}])
            with (OUT / f"{name}-followup-server.log").open("w") as log:
                proc = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"),
                    "--engine", "strata", "--config", str(OUT / f"{name}-test-config.json"),
                    "--host", "127.0.0.1", "--port", "1112"], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                wait_ready(proc)
                for stage, (prompt, limit) in enumerate(zip(extra["prompts"], extra["max_tokens"]), 4):
                    messages.append({"role": "user", "content": prompt})
                    payload = {**cases["sampling"], "model": cfg["model_name"],
                               "messages": list(messages), "max_tokens": limit, "stream": False}
                    stem = f"{name}-python-{stage}"
                    (OUT / f"{stem}-request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                    print(f"{name}: Python stage {stage} started", flush=True)
                    start = time.monotonic()
                    response = api(1112, "/v1/chat/completions", payload)
                    (OUT / f"{stem}-response.json").write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n")
                    c = response["choices"][0]
                    text = c["message"]["content"]
                    (OUT / f"{stem}.md").write_text(text + "\n")
                    manifest["runs"].append({"model": name, "stage": stage,
                        "input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "finish_reason": c["finish_reason"], "wall_s": round(time.monotonic() - start, 3),
                        "timings": response.get("timings")})
                    (OUT / "followup-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                    print(f"{name}: Python stage {stage} finished ({c['finish_reason']}, {len(text)} chars)", flush=True)
                    assert c["finish_reason"] != "length", "truncated answer needs matched continuation"
                    messages.append({"role": "assistant", "content": text})
                stop(proc)
                proc = None
    finally:
        try:
            if proc is not None:
                stop(proc)
        finally:
            manifest.update(production_unchanged=all(sha(p) == h for p, h in watched.items()),
                            production_health_after=api(1111, "/health"))
            (OUT / "followup-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            assert manifest["production_unchanged"]
            print("Follow-ups finished; project code/config unchanged; production left unloaded", flush=True)


if __name__ == "__main__":
    main()
