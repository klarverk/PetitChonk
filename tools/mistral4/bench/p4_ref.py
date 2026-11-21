"""PCHONK-PUB-001 P4: the llama.cpp side of B2's and B3's scoring: ref_logprobs.exe with T4's variant settings.

Runs ref_logprobs (scripts\\pub\\p4, built against llama.cpp c7518dbe) on JSON-lines sequences {"id","ids","from"}
with one of T4's variants (scripts\\fti\\t4\\t4_runs.py VARIANTS):
  gpu  --gpu 1 --experts-cpu 1 --fa auto --ubatch 1024   (the reference B2 and B3 are scored against)
  cpu  --gpu 0 --fa auto --ubatch 1024                   (llama.cpp's own GPU-vs-CPU spread, for scale)
and --threads 10, --ctx and --batch = the longest sequence rounded up to a multiple of 256 (ref_logprobs needs
n <= both), T4's environment (no STRATA_*, GGML_*, LLAMA_* variable; CUDA 13.0 and llama.cpp's bin on PATH).
Writes <out>/logprobs.jsonl (ref_logprobs's format) and <out>/run.json (argv, hashes, times, exit).

usage: <python> -I p4_ref.py [--execute] --variant gpu|cpu --model M --in SEQ.jsonl --out DIR [--rows ID]...
The out dir must not exist. Without --execute it prints the argv and runs nothing.
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(os.environ.get("PETITCHONK_WORK", "."))   # the work folder (tools/mistral4/README.md)
TOOL = Path(__file__).resolve().parent / "ref_logprobs.exe"
CUDA = os.environ.get("PETITCHONK_CUDA", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")   # 13.0: never 13.2.x on RTX 50
REF_BIN = ROOT / r"ref\llama.cpp-c7518dbe\build\bin"
VARIANTS = {"gpu": ["--gpu", "1", "--experts-cpu", "1", "--fa", "auto", "--ubatch", "1024"],
            "cpu": ["--gpu", "0", "--fa", "auto", "--ubatch", "1024"]}


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rows", action="append", default=[])
    ap.add_argument("--ctx", type=int, help="--ctx and --batch (default: the longest sequence, rounded up to 256)")
    a = ap.parse_args()
    longest = max(len(json.loads(l)["ids"]) for l in open(a.inp, encoding="utf-8") if l.strip())
    ctx = a.ctx or -(-longest // 256) * 256
    if ctx < longest:
        raise SystemExit(f"--ctx {ctx} < the longest sequence {longest}")
    argv = [str(TOOL), "--model", a.model, "--in", a.inp, "--out", str(a.out)] + VARIANTS[a.variant] + \
        ["--ctx", str(ctx), "--batch", str(ctx), "--threads", "10"] + [x for r in a.rows for x in ("--rows", r)]
    print(json.dumps({"argv": argv}, indent=1))
    if not a.execute:
        return
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("STRATA_", "GGML_", "LLAMA_"))}
    env["PATH"] = rf"{CUDA}\bin\x64;{REF_BIN};" + env["PATH"]
    run = {"argv": argv, "variant": a.variant, "tool_sha256": sha_file(TOOL), "input_sha256": sha_file(a.inp),
           "start": now()}
    a.out.mkdir(parents=True, exist_ok=False)
    with open(a.out / "stdout.txt", "wb") as o, open(a.out / "stderr.txt", "wb") as e:
        r = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, stdout=o, stderr=e)
    run.update({"end": now(), "exit": r.returncode})
    if (a.out / "logprobs.jsonl").exists():
        run["logprobs_sha256"] = sha_file(a.out / "logprobs.jsonl")
    (a.out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    if r.returncode:
        raise SystemExit(f"ref_logprobs exit {r.returncode}")


if __name__ == "__main__":
    main()
