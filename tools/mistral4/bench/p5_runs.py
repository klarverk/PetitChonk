"""PCHONK-PUB-001 P5: the accuracy runs B2 and B3 under P4's frozen part A, then their scores.

Runs one model process at a time, in this order, each into its own folder under <out root> (which must not exist):
  b2_cand      p4_cand serve, b2_windows.jsonl, default placement, no freeze, --batch-rows 160000 (2 batches)
  b2_ref_gpu   p4_ref --variant gpu, b2_windows.jsonl (--ctx 4096)
  b2_ref_cpu   p4_ref --variant cpu, the first 8 windows (b2_first8.jsonl, written here): the scale reference
  b3_cand      p4_cand serve, b3_sequences.jsonl, default placement, no freeze, --batch-rows 150000 (2 batches)
  b3_ref_gpu   p4_ref --variant gpu, b3_sequences.jsonl (--ctx 1024)
  score        p4_score --b2 ... --b2-scale ... --b3 ... --out scores.json (reads only)
Before the first run, every file in pub\\p4\\FROZEN_P4A_v1.sha256 is checked against its hash; any mismatch stops
everything. A non-zero exit stops the sequence there; runs.json records each step's argv, start, end and exit.
The model is used throughout; nothing else may use it meanwhile (Klardon is told before and after, by hand).

Resume (added 20 Nov after run 1 was stopped mid-step by Claude Code's memory reaper; P5_accuracy.md §4):
  --resume STEP continues an existing <out root> from STEP. Every earlier step must be recorded in runs.json with
  exit 0, and its logprobs.jsonl must match the logprobs_sha256 in its own run.json (score has no folder). The
  manifest is checked again; b2_first8.jsonl must equal the first 8 windows. STEP's partial folder and log are
  moved aside, kept as evidence, to <STEP>.interrupted<k>. The steps from STEP on then run as above, and
  runs.json gains a {"resume": STEP, "at": ...} entry before them.

usage: <venv-tools python> -I p5_runs.py [--execute] [--resume STEP] <out root>
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("PETITCHONK_WORK", "."))   # the work folder (tools/mistral4/README.md)
PY = sys.executable
P4 = Path(__file__).resolve().parent
FREEZE = ROOT / r"pub\p4\freeze_a"
MANIFEST = ROOT / r"pub\p4\FROZEN_P4A_v1.sha256"
MANIFEST_SHA = os.environ.get("PETITCHONK_P4A_MANIFEST_SHA256",
                              "c78895d11cd27595aa6ec786567b1d85dfc84c6d4b28e13a8d3a800f8380ca2d")   # Petit Chonk 1.0.0's
BINARY = Path(os.environ.get("PETITCHONK_BINARY") or ROOT / r"release\build-p2\strata-poly.exe")   # the candidate
MODEL = ROOT / (r"models\unsloth-Mistral-Small-4-119B-2603-GGUF@bd93c721\UD-Q4_K_M"
                r"\Mistral-Small-4-119B-2603-UD-Q4_K_M-00001-of-00003.gguf")
SCALE_WINDOWS = 8                                              # P4 §2 and §4: the CPU scale reference


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def check_manifest():
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != MANIFEST_SHA:
        raise SystemExit(f"{MANIFEST} is not the recorded manifest")
    bad = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        h, name = line.split(maxsplit=1)
        p = ROOT / name.lstrip("*")
        d = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                d.update(chunk)
        if d.hexdigest() != h:
            bad.append(name)
    if bad:
        raise SystemExit(f"part A manifest mismatch: {bad}")


def steps(out):
    cand = [PY, "-I", str(P4 / "p4_cand.py"), "--execute", "--binary", str(BINARY), "--model", str(MODEL)]
    ref = [PY, "-I", str(P4 / "p4_ref.py"), "--execute", "--model", str(MODEL)]
    w, s, it = FREEZE / "b2_windows.jsonl", FREEZE / "b3_sequences.jsonl", FREEZE / "b3_items.jsonl"
    first8 = out / "b2_first8.jsonl"
    return [
        ("b2_cand", cand + ["--in", str(w), "--out", str(out / "b2_cand"), "--batch-rows", "160000"]),
        ("b2_ref_gpu", ref + ["--variant", "gpu", "--in", str(w), "--out", str(out / "b2_ref_gpu")]),
        ("b2_ref_cpu", ref + ["--variant", "cpu", "--in", str(first8), "--out", str(out / "b2_ref_cpu")]),
        ("b3_cand", cand + ["--in", str(s), "--out", str(out / "b3_cand"), "--batch-rows", "150000"]),
        ("b3_ref_gpu", ref + ["--variant", "gpu", "--in", str(s), "--out", str(out / "b3_ref_gpu")]),
        ("score", [PY, "-I", str(P4 / "p4_score.py"),
                   "--b2", str(w), str(out / "b2_cand" / "logprobs.jsonl"), str(out / "b2_ref_gpu" / "logprobs.jsonl"),
                   "--b2-scale", str(out / "b2_ref_cpu" / "logprobs.jsonl"),
                   "--b3", str(it), str(out / "b3_cand" / "logprobs.jsonl"), str(out / "b3_ref_gpu" / "logprobs.jsonl"),
                   "--out", str(out / "scores.json")]),
    ]


def sha_file(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            d.update(chunk)
    return d.hexdigest()


def prepare_resume(out, plan, step):
    """Check the steps before `step` and set `step`'s partial output aside. -> the runs.json log so far."""
    names = [n for n, _ in plan]
    if step not in names:
        raise SystemExit(f"--resume {step}: not a step ({', '.join(names)})")
    log = json.loads((out / "runs.json").read_text(encoding="utf-8"))
    done = {r["step"]: r for r in log if "step" in r}
    for name in names[:names.index(step)]:
        if done.get(name, {}).get("exit") != 0:
            raise SystemExit(f"--resume {step}: {name} is not recorded as complete with exit 0")
        run = json.loads((out / name / "run.json").read_text(encoding="utf-8"))
        if sha_file(out / name / "logprobs.jsonl") != run.get("logprobs_sha256"):
            raise SystemExit(f"--resume {step}: {name}'s logprobs.jsonl does not match its run.json")
        print(f"{now()} {name}: complete, logprobs {run['logprobs_sha256'][:16]}... checked", flush=True)
    lines = (FREEZE / "b2_windows.jsonl").read_bytes().split(b"\n")
    if (out / "b2_first8.jsonl").read_bytes() != b"\n".join(lines[:SCALE_WINDOWS]) + b"\n":
        raise SystemExit("--resume: b2_first8.jsonl is not the first 8 windows")
    k = 1
    while (out / f"{step}.interrupted{k}").exists() or (out / f"{step}.interrupted{k}.log").exists():
        k += 1
    for src, dst in ((out / step, out / f"{step}.interrupted{k}"), (out / f"{step}.log", out / f"{step}.interrupted{k}.log")):
        if src.exists():
            src.rename(dst)
            print(f"{now()} set aside: {src.name} -> {dst.name}", flush=True)
    log.append({"resume": step, "at": now()})
    (out / "runs.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return log


def main(argv):
    execute = "--execute" in argv
    args = [a for a in argv if a != "--execute"]
    resume = None
    if "--resume" in args:
        i = args.index("--resume")
        resume = args[i + 1]
        del args[i:i + 2]
    if len(args) != 1:
        raise SystemExit(__doc__)
    out = Path(args[0])
    plan = steps(out)
    if resume:
        plan_run = plan[[n for n, _ in plan].index(resume):] if resume in [n for n, _ in plan] else plan
    else:
        plan_run = plan
    if not execute:
        for name, a in plan_run:
            print(name, " ".join(a[2:]) if a[0] == PY else " ".join(a))
        return
    check_manifest()
    print(f"{now()} part A manifest checked", flush=True)
    if resume:
        log = prepare_resume(out, plan, resume)
    else:
        out.mkdir(parents=True, exist_ok=False)
        lines = (FREEZE / "b2_windows.jsonl").read_bytes().split(b"\n")
        (out / "b2_first8.jsonl").write_bytes(b"\n".join(lines[:SCALE_WINDOWS]) + b"\n")
        log = []
    for name, a in plan_run:
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("STRATA_", "GGML_", "LLAMA_"))}
        rec = {"step": name, "argv": a, "start": now()}
        print(f"{rec['start']} {name}", flush=True)
        with open(out / f"{name}.log", "wb") as f:
            r = subprocess.run(a, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
        rec.update({"end": now(), "exit": r.returncode})
        log.append(rec)
        (out / "runs.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        if r.returncode:
            raise SystemExit(f"{name}: exit {r.returncode} (see {name}.log); P5 stops here")
    print(f"{now()} done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
