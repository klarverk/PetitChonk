"""PCHONK-PUB-001 P4: run the harness qualification's eight runs in order, then judge them (p4_qual.py).

The runs and pass rules are p4_qual.py's (P4_benchmark_design.md §3.2). Uses the model; nothing else may use it
meanwhile (Klardon is told before and after). Inputs: <freeze dir>/qual_all.jsonl; output: <qual root> (must not
exist), with runs.json recording each run's argv, start, end and exit.

usage: <venv-tools python> -I p4_qual_runs.py <freeze dir> <qual root>
"""
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("PETITCHONK_WORK", "."))   # the work folder (tools/mistral4/README.md)
PY = sys.executable
P4 = Path(__file__).resolve().parent
BINARY = Path(os.environ.get("PETITCHONK_BINARY") or ROOT / r"release\build-p2\strata-poly.exe")   # the candidate
MODEL = ROOT / (r"models\unsloth-Mistral-Small-4-119B-2603-GGUF@bd93c721\UD-Q4_K_M"
                r"\Mistral-Small-4-119B-2603-UD-Q4_K_M-00001-of-00003.gguf")
REF_LOGITS = Path(__file__).resolve().parent.parent / "fti" / "ref_logits.exe"
CUDA = os.environ.get("PETITCHONK_CUDA", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")   # 13.0: never 13.2.x on RTX 50
REF_BIN = ROOT / r"ref\llama.cpp-c7518dbe\build\bin"
X1, X2 = "arc_challenge:701:2", "arc_challenge:594:1"
CTX = "2560"


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def main(freeze, root):
    freeze, root = Path(freeze), Path(root)
    qual = freeze / "qual_all.jsonl"
    root.mkdir(parents=True, exist_ok=False)
    seqs = {s["id"]: s for s in (json.loads(l) for l in open(qual, encoding="utf-8"))}
    (root / "qual_x1.jsonl").write_text(json.dumps(seqs[X1], separators=(",", ":")) + "\n", encoding="utf-8")
    (root / "x1_ids.txt").write_text(",".join(map(str, seqs[X1]["ids"])), encoding="utf-8")
    cand = [PY, "-I", str(P4 / "p4_cand.py"), "--execute", "--binary", str(BINARY), "--model", str(MODEL),
            "--in", str(qual)]
    ref = [PY, "-I", str(P4 / "p4_ref.py"), "--execute", "--variant", "gpu", "--model", str(MODEL), "--ctx", CTX]
    runs = [
        ("ref_all", ref + ["--in", str(qual), "--out", str(root / "ref_all"), "--rows", "tail", "--rows", X1,
                           "--rows", X2]),
        ("ref_x", ref + ["--in", str(root / "qual_x1.jsonl"), "--out", str(root / "ref_x"), "--rows", X1]),
        ("ref_logits_x", [str(REF_LOGITS), "--model", str(MODEL), "--prompt-file", str(root / "x1_ids.txt"),
                          "--gpu", "1", "--experts-cpu", "1", "--fa", "auto", "--ubatch", "1024", "--ctx", CTX,
                          "--batch", CTX, "--threads", "10", "--out", str(root / "ref_logits_x")]),
        ("cand_p5cfg", cand + ["--out", str(root / "cand_p5cfg"), "--keep-rows", "tail", "--keep-rows", X1,
                               "--keep-rows", X2]),
        ("cand_first_serve", cand + ["--out", str(root / "cand_first_serve"), "--freeze-experts", "--only", "tail",
                                     "--keep-rows", "tail"]),
        ("cand_first_oneshot", cand + ["--out", str(root / "cand_first_oneshot"), "--mode", "one-shot",
                                       "--freeze-experts", "--only", "tail", "--keep-rows", "tail"]),
        ("cand_cpu_serve", cand + ["--out", str(root / "cand_cpu_serve"), "--placement", "cpu", "--keep-rows", X1,
                                   "--keep-rows", X2]),
        ("cand_cpu_oneshot", cand + ["--out", str(root / "cand_cpu_oneshot"), "--mode", "one-shot", "--placement",
                                     "cpu", "--only", X1, "--only", X2, "--keep-rows", X1, "--keep-rows", X2]),
    ]
    log = []
    for name, argv in runs:
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("STRATA_", "GGML_", "LLAMA_"))}
        if name == "ref_logits_x":
            env["PATH"] = rf"{CUDA}\bin\x64;{REF_BIN};" + env["PATH"]
            (root / name).mkdir()
        rec = {"run": name, "argv": argv, "start": now()}
        print(f"{rec['start']} {name}", flush=True)
        with open(root / f"{name}.log", "wb") as f:
            r = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
        rec.update({"end": now(), "exit": r.returncode})
        log.append(rec)
        (root / "runs.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        if r.returncode:
            raise SystemExit(f"{name}: exit {r.returncode} (see {name}.log); the qualification stops here")
    r = subprocess.run([PY, "-I", str(P4 / "p4_qual.py"), str(root), str(freeze)])
    print(f"{now()} judged, exit {r.returncode}", flush=True)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
