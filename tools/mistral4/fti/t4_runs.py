"""T4 reference runs (PCHONK-FTI-001): llama.cpp c7518dbe through ref_logits.exe on T3's frozen sequences, in six
variants, for T6's trusted pairs. Every run: ctx 16384, batch 16384, 10 threads; the logits of every position.

  gpu      GPU, blk.1-35's experts on the CPU (as S6b), FA auto, ubatch 1024   (the primary reference)
  chunk    as gpu, ubatch 128                                                  (smaller prefill chunks)
  fa_off   as gpu, flash attention off
  unfused  as gpu, with a callback on every node (no fusion); records each layer's amax factor (amax.jsonl)
  cpu      CPU only, FA auto, ubatch 1024
  decode   as gpu, but only BOS as the prompt and every other token one per call (the decode path)

After each run: rows = the sequence's ids, every logit finite (read in chunks), SHA-256 of logits.f32, the time and
the placement lines of stderr. A finished run (run.json with "ok") is skipped, so the script can be restarted.

usage: <venv-tools python> -I t4_runs.py [variant ...]   (default: all, in the order above)
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PETITCHONK_WORK", "."))   # the work folder (tools/mistral4/README.md)
TOOL = Path(__file__).resolve().parent / "ref_logits.exe"
MODEL = ROOT / (r"models\unsloth-Mistral-Small-4-119B-2603-GGUF@bd93c721\UD-Q4_K_M"
                r"\Mistral-Small-4-119B-2603-UD-Q4_K_M-00001-of-00003.gguf")
SEQ = ROOT / r"fti\sequences"
OUT = ROOT / r"fti\ref"
CUDA = os.environ.get("PETITCHONK_CUDA", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")   # 13.0: never 13.2.x on RTX 50
REF_BIN = ROOT / r"ref\llama.cpp-c7518dbe\build\bin"
V = 131072
VARIANTS = {
    "gpu": ["--gpu", "1", "--experts-cpu", "1", "--fa", "auto", "--ubatch", "1024"],
    "chunk": ["--gpu", "1", "--experts-cpu", "1", "--fa", "auto", "--ubatch", "128"],
    "fa_off": ["--gpu", "1", "--experts-cpu", "1", "--fa", "off", "--ubatch", "1024"],
    "unfused": ["--gpu", "1", "--experts-cpu", "1", "--fa", "auto", "--ubatch", "1024", "--unfused", "1"],
    "cpu": ["--gpu", "0", "--fa", "auto", "--ubatch", "1024"],
    "decode": ["--gpu", "1", "--experts-cpu", "1", "--fa", "auto", "--ubatch", "1024"],
}


def env():
    e = {k: v for k, v in os.environ.items() if not k.upper().startswith(("STRATA_", "GGML_", "LLAMA_"))}
    e["PATH"] = rf"{CUDA}\bin\x64;{REF_BIN};" + e["PATH"]
    return e


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def check(p, rows):
    m = np.memmap(p, dtype=np.float32, mode="r")
    if m.size != rows * V:
        return {"rows_ok": False, "rows": m.size / V}
    bad, most = 0, 0.0
    for r0 in range(0, rows, 256):
        blk = np.asarray(m[r0 * V:min(rows, r0 + 256) * V])
        fin = np.isfinite(blk)
        bad += int((~fin).sum())
        if fin.any():
            most = max(most, float(np.abs(blk[fin]).max()))
    del m
    return {"rows_ok": True, "rows": rows, "nonfinite": bad, "max_abs_logit": most}


def main(names):
    manifest = json.loads((SEQ / "sequences.json").read_text(encoding="utf-8"))
    seqs = manifest["sequences"]
    ids_dir = OUT / "ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    for n, s in seqs.items():
        (ids_dir / f"{n}.txt").write_text(",".join(map(str, s["ids"])), encoding="utf-8")
        (ids_dir / f"{n}.bos.txt").write_text(str(s["ids"][0]), encoding="utf-8")
        (ids_dir / f"{n}.rest.txt").write_text(",".join(map(str, s["ids"][1:])), encoding="utf-8")
    for var in names:
        for n, s in seqs.items():
            d = OUT / var / n
            rj = d / "run.json"
            if rj.exists() and json.loads(rj.read_text()).get("ok"):
                print(f"{var}/{n}: done before, skipped", flush=True)
                continue
            d.mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                f.unlink()
            if var == "decode":
                inp = ["--prompt-file", str(ids_dir / f"{n}.bos.txt"), "--follow-file", str(ids_dir / f"{n}.rest.txt")]
            else:
                inp = ["--prompt-file", str(ids_dir / f"{n}.txt")]
            argv = [str(TOOL), "--model", str(MODEL)] + inp + VARIANTS[var] + [
                "--ctx", "16384", "--batch", "16384", "--threads", "10", "--out", str(d)]
            t0 = time.time()
            r = subprocess.run(argv, env=env(), stdin=subprocess.DEVNULL, capture_output=True, timeout=4 * 3600)
            (d / "stdout.txt").write_bytes(r.stdout)
            (d / "stderr.txt").write_bytes(r.stderr)
            secs = round(time.time() - t0, 1)
            err = r.stderr.decode("utf-8", "replace").splitlines()
            info = {"variant": var, "sequence": n, "argv": argv[1:], "exit": r.returncode, "seconds": secs,
                    "n_ids": s["n_ids"], "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                    "stdout": r.stdout.decode("utf-8", "replace").strip(),
                    "placement": [l.strip()[:200] for l in err if "buffer size" in l or "offloaded" in l
                                  or "REPACK" in l][:12]}
            if r.returncode == 0 and (d / "logits.f32").exists():
                info.update(check(d / "logits.f32", s["n_ids"]))
                info["logits_sha256"] = sha(d / "logits.f32")
                if var == "unfused":
                    lines = [json.loads(l) for l in (d / "amax.jsonl").read_text().splitlines()]
                    info["amax"] = {"lines": len(lines), "max": max(l["max"] for l in lines),
                                    "not_one": sum(l["not_one"] for l in lines),
                                    "nonfinite": sum(l["nonfinite"] for l in lines),
                                    "layers": sorted({l["layer"] for l in lines if l["not_one"]})}
            info["ok"] = r.returncode == 0 and info.get("rows_ok") and info.get("nonfinite") == 0
            rj.write_text(json.dumps(info, indent=1), encoding="utf-8")
            print(f"{var}/{n}: exit {r.returncode}, {secs} s, rows {info.get('rows')}, nonfinite "
                  f"{info.get('nonfinite')}, max|logit| {info.get('max_abs_logit')}"
                  + (f", amax {info['amax']}" if 'amax' in info else ""), flush=True)
    print("all done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(VARIANTS))
