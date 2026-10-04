"""Run only the two original cases in test-inputs.json; no benchmark suite imports."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def api(port, path, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}" + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as response:
        return json.load(response)


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def wait_ready(proc):
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"test server exited with {proc.returncode}")
        try:
            if api(1112, "/health").get("loaded"):
                return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError("test server startup")


def stop(proc):
    try:
        if proc.poll() is None:
            try:
                api(1112, "/v1/unload", {})
            except OSError:
                pass
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)


def main():
    cases = json.loads((OUT / "test-inputs.json").read_text())
    configs = {name: json.loads((ROOT / f"strata-{name}.json").read_text())
               for name in ("iq3_s", "iq4_xs")}
    cfg = configs["iq4_xs"]
    rt = Path(cfg["args"][cfg["args"].index("--mtp") + 1])
    watched = [ROOT / "strata-iq3_s.json", ROOT / "strata-iq4_xs.json", rt / "draft_vocab.bin"]
    source_files = json.loads((OUT / "source-manifest.json").read_text())
    watched += [ROOT / n for n in source_files]
    before = {str(p): sha(p) for p in watched}
    manifest = {"test_inputs_sha256": sha(OUT / "test-inputs.json"),
                "source_snapshot_sha256": sha(OUT / "source-snapshot.md"),
                "production_hashes_before": before, "engine_sha256": sha(cfg["exe"]),
                "configs": {}, "runs": [], "suite": "original tasks only; no existing suite"}
    if api(1111, "/health").get("loaded"):
        if api(1111, "/metrics")["live"]["state"] != "idle":
            raise RuntimeError("production is busy")
        api(1111, "/v1/unload", {})
    proc = None
    try:
        for name, original in configs.items():
            test = dict(cfg)
            test.update(gpu=[1, 0], layer_split=2, host="127.0.0.1", port=1112,
                        model_name=original["model_name"], tokenizer=original["tokenizer"],
                        log=str(OUT / f"{name}-engine.log"))
            test["args"] = list(cfg["args"])
            for flag in ("--pack", "--native", "--ple-gguf"):
                test["args"][test["args"].index(flag) + 1] = original["args"][original["args"].index(flag) + 1]
            test["args"] += ["--suffix-draft", "0"]
            config = OUT / f"{name}-test-config.json"
            config.write_text(json.dumps(test, indent=1) + "\n")
            manifest["configs"][name] = test
            with (OUT / f"{name}-server.log").open("w") as log:
                proc = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"),
                    "--engine", "strata", "--config", str(config), "--host", "127.0.0.1", "--port", "1112"],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                wait_ready(proc)
                assert "draft head over 248320 tokens" in Path(test["log"]).read_text()
                print(name + ": ready with full vocabulary", flush=True)
                for task, prompts, limits in [
                    ("marketing", [cases["marketing"]], [cases["token_limits"]["marketing"]]),
                    ("python", cases["python_stages"], cases["token_limits"]["python_stages"]),
                ]:
                    messages = [{"role": "system", "content": cases["system"]}]
                    for stage, (prompt, limit) in enumerate(zip(prompts, limits), 1):
                        messages.append({"role": "user", "content": prompt})
                        payload = {**cases["sampling"], "model": test["model_name"],
                                   "messages": list(messages), "max_tokens": limit, "stream": False}
                        stem = f"{name}-{task}-{stage}"
                        (OUT / f"{stem}-request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                        print(f"{name}: {task} stage {stage} started", flush=True)
                        started = time.monotonic()
                        response = api(1112, "/v1/chat/completions", payload)
                        (OUT / f"{stem}-response.json").write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n")
                        choice = response["choices"][0]
                        text = choice["message"].get("content") or ""
                        (OUT / f"{stem}.md").write_text(text + "\n")
                        assert text.strip(), "empty answer"
                        entry = {"model": name, "task": task, "stage": stage,
                                 "input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                 "wall_s": round(time.monotonic() - started, 3),
                                 "finish_reason": choice["finish_reason"], "timings": response.get("timings")}
                        manifest["runs"].append(entry)
                        (OUT / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                        print(f"{name}: {task} stage {stage} finished ({choice['finish_reason']}, {len(text)} chars)", flush=True)
                        if choice["finish_reason"] == "length":
                            raise RuntimeError("truncated deliverable: requires the predeclared matched continuation procedure")
                        messages.append({"role": "assistant", "content": text})
                stop(proc)
                proc = None
    finally:
        try:
            if proc is not None:
                stop(proc)
        finally:
            after = {str(p): sha(p) for p in watched}
            manifest.update(production_hashes_after=after, production_unchanged=before == after,
                            production_health_after=api(1111, "/health"))
            (OUT / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            assert before == after
            print("Temporary server stopped; production code/config/vocabulary unchanged; production left unloaded", flush=True)


if __name__ == "__main__":
    main()
