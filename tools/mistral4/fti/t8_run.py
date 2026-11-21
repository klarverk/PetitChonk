"""PCHONK-FTI-001 T8: the candidate's runs, as fixed by the T6 contract (§3). Frozen with the contract; T8 runs it
unchanged after T7's gate opens. It refuses to run without --execute.

Run sets (D-FTI2, and the D-FTI6 review's B1 and S1): placement x path over every T3 sequence, plus default_split.
  placement  default     the engine's default path: CUDA graphs on, partial expert cache in VRAM, the rest on the CPU
             graphs_off  as default, with --no-graphs; STRATA_GUARD_LOG on (the guard log needs graphs off, T5)
             cpu         --no-gpu-experts: every routed expert on the CPU
  path       whole       ids[:n-TAIL] as the prompt (--tokens-file), --prompt-batch 1024, STRATA_LOGITS_ALL_ROWS=1;
                         ids[n-TAIL:] followed (--follow-file), --max-new TAIL+1. Rows 0..n-TAIL-2 come from the
                         prompt read (calls of 1024); rows n-TAIL-1..n-1 from single-token steps after it (B1: the
                         hand-over from a long read to decoding, where the expert cache takes back its VRAM)
             chunk       as whole, --prompt-batch 128
             decode      BOS as the prompt, the rest followed, --max-new n: every row from a single-token step
             split       (default placement only) as whole, with --stage-ahead 16384: every read goes call by call,
                         and each call leaves a share of the uncached experts to the CPU, the path a prompt under
                         1,024 tokens takes (S1)
Every run: --max-context 16384 --lookup 0 --window 1 --eos-ids 0, STRATA_LOGITS_DIR=<run>/dump, no other STRATA_*
variable but those named above, CUDA 13.0 on PATH. --eos-ids 0: chat_en holds </s> (id 2) at 779, 916 and 1125;
a followed token counts as picked and a picked EOS ends the answer; <unk> (0) occurs in no T3 sequence, and with
--eos-ids given the engine does not add the GGUF's EOS (T5 re-entry R-002, check 3).

The model (the re-review's RB1): every shard of the model file is hashed once, at the start, into <out>/model.json
and every run.json (model_sha256); the judge's X4 compares them with T7's.

Attempts (S2, the re-review's RS5): a run whose run.json says exit 0 is kept and skipped (a resume), but only when
its argv, binary SHA-256 and model SHA-256 equal this invocation's; otherwise the script stops (a new binary or model
needs a new --out). Any other earlier attempt (a non-zero exit, a timeout, or no run.json at all) stops the script,
unless that run is named with --new-attempt <run set>/<sequence> <re-entry id>: then the old attempt's folder is
renamed <sequence>.attempt<k> and kept, the run starts again, and its attempt.json names the re-entry and every
earlier attempt. The contract (§5) says when a new attempt is lawful; the judge fails any earlier attempt that is not
named this way. Output: <out>/<run set>/<sequence>/ with run.json (argv, the STRATA_* environment, exit or
"timeout", seconds, the binary's and the model's SHA-256, the SHA-256 of the ids passed, n), stdout.txt,
stderr.txt, dump/, guard.jsonl, attempt.json (new attempts only).

usage: <venv-tools python> -I t8_run.py --execute --binary <strata-poly.exe> --out <dir> [--model <gguf>]
                                        [--new-attempt <run set>/<sequence> <re-entry id>] ... [run set ...]
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("PETITCHONK_WORK", "."))   # the work folder (tools/mistral4/README.md)
MODEL = str(ROOT / (r"models\unsloth-Mistral-Small-4-119B-2603-GGUF@bd93c721\UD-Q4_K_M"
                     r"\Mistral-Small-4-119B-2603-UD-Q4_K_M-00001-of-00003.gguf"))
SEQ = ROOT / "fti" / "sequences" / "sequences.json"
CUDA = os.environ.get("PETITCHONK_CUDA", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")   # 13.0: never 13.2.x on RTX 50
TAIL = 256
TIMEOUT_S = 6 * 3600
PLACEMENTS = {"default": [], "graphs_off": ["--no-graphs"], "cpu": ["--no-gpu-experts"]}
RUN_SETS = [(p, q) for p in PLACEMENTS for q in ("whole", "chunk", "decode")] + [("default", "split")]
PROMPT_BATCH = {"whole": 1024, "chunk": 128, "split": 1024}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def model_files(model):
    """{shard file name: SHA-256} for every shard of a split GGUF (<stem>-0000k-of-0000N.gguf), or the one file."""
    p = Path(model)
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", p.name)
    names = [f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf" for i in range(1, int(m.group(3)) + 1)] if m else [p.name]
    return {nm: sha(p.parent / nm) for nm in names}


def command(binary, model, placement, path, d, ids, write=True):
    n = len(ids)
    if path == "decode":
        prompt, follow, max_new = ids[:1], ids[1:], n
        extra = []
    else:
        prompt, follow, max_new = ids[:n - TAIL], ids[n - TAIL:], TAIL + 1
        extra = ["--prompt-batch", str(PROMPT_BATCH[path])] + (["--stage-ahead", "16384"] if path == "split" else [])
    if write:
        (d / "tokens.txt").write_text(",".join(map(str, prompt)), encoding="utf-8")
        (d / "follow.txt").write_text(",".join(map(str, follow)), encoding="utf-8")
    argv = [binary, "--model", model, "--tokens-file", str(d / "tokens.txt"), "--follow-file", str(d / "follow.txt"),
            "--max-new", str(max_new)] + extra + \
        ["--max-context", "16384", "--lookup", "0", "--window", "1", "--eos-ids", "0"] + PLACEMENTS[placement]
    strata = {"STRATA_LOGITS_DIR": str(d / "dump")}
    if path != "decode":
        strata["STRATA_LOGITS_ALL_ROWS"] = "1"
    if placement == "graphs_off":
        strata["STRATA_GUARD_LOG"] = str(d / "guard.jsonl")
    if not write:
        return argv, strata, None
    passed = (d / "tokens.txt").read_text(encoding="utf-8") + "," + (d / "follow.txt").read_text(encoding="utf-8")
    return argv, strata, hashlib.sha256(passed.encode()).hexdigest()


def run_one(binary, bin_sha, model, model_sha, out, placement, path, name, ids, new_attempts):
    """new_attempts: {"<run set>/<sequence>": re-entry id}, from --new-attempt."""
    rs = f"{placement}_{path}"
    d = out / rs / name
    attempt = None
    if d.exists():
        rj = d / "run.json"
        info = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
        if info.get("exit") == 0:
            want = command(binary, model, placement, path, d, ids, write=False)[0][1:]
            if info.get("argv") == want and info.get("binary_sha256") == bin_sha and \
                    info.get("model_sha256") == model_sha:
                return
            sys.exit(f"stopped: {rs}/{name} exited 0 with another argv, binary or model. A new binary or model "
                     f"needs a new --out (contract §5, §10).")
        if f"{rs}/{name}" not in new_attempts:
            sys.exit(f"stopped: an earlier attempt of {rs}/{name} did not exit 0 (or left no run.json). The contract "
                     f"(§5) decides whether a new attempt is lawful; if it is, rerun with "
                     f"--new-attempt {rs}/{name} <re-entry id>.")
        k = 1
        while (d.parent / f"{name}.attempt{k}").exists():
            k += 1
        d.rename(d.parent / f"{name}.attempt{k}")
        attempt = {"reentry": new_attempts[f"{rs}/{name}"],
                   "previous": sorted(p.name for p in d.parent.glob(f"{name}.attempt*"))}
    d.mkdir(parents=True)
    if attempt:
        (d / "attempt.json").write_text(json.dumps(attempt, indent=1), encoding="utf-8")
    argv, strata, ids_sha = command(binary, model, placement, path, d, ids)
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("STRATA_")}
    env["PATH"] = rf"{CUDA}\bin\x64;{CUDA}\bin;" + env["PATH"]
    env.update(strata)
    t0 = time.time()
    try:
        r = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, capture_output=True, timeout=TIMEOUT_S)
        code, so, se = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        code, so, se = "timeout", e.stdout or b"", e.stderr or b""
    (d / "stdout.txt").write_bytes(so)
    (d / "stderr.txt").write_bytes(se)
    info = {"placement": placement, "path": path, "sequence": name, "argv": argv[1:], "env_strata": strata,
            "exit": code, "seconds": round(time.time() - t0, 1), "binary_sha256": bin_sha, "model_sha256": model_sha,
            "ids_sha256": ids_sha, "n_ids": len(ids), "tail": 0 if path == "decode" else TAIL}
    (d / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"{placement}_{path}/{name}: exit {code}, {info['seconds']} s", flush=True)


def main(argv):
    if "--execute" not in argv:
        sys.exit("refused: T8 runs only with --execute, after T7's gate is open")
    binary = argv[argv.index("--binary") + 1]
    out = Path(argv[argv.index("--out") + 1])
    model = argv[argv.index("--model") + 1] if "--model" in argv else MODEL
    names = {f"{p}_{q}" for p, q in RUN_SETS}
    new_attempts = {argv[i + 1]: argv[i + 2] for i, a in enumerate(argv) if a == "--new-attempt"}
    if any(not v.strip() or k.split("/")[0] not in names for k, v in new_attempts.items()):
        sys.exit("--new-attempt takes <run set>/<sequence> <re-entry id>")
    wanted = [a for i, a in enumerate(argv) if a in names and (i == 0 or argv[i - 1] != "--new-attempt")]
    bin_sha = sha(binary)
    model_sha = model_files(model)
    out.mkdir(parents=True, exist_ok=True)
    mj = out / "model.json"
    if mj.exists() and json.loads(mj.read_text(encoding="utf-8")) != model_sha:
        sys.exit(f"stopped: {mj} names another model; a new model needs a new --out")
    mj.write_text(json.dumps(model_sha, indent=1), encoding="utf-8")
    print(f"binary {bin_sha}; model {model_sha}", flush=True)
    seqs = json.loads(SEQ.read_text(encoding="utf-8"))["sequences"]
    for placement, path in RUN_SETS:
        if wanted and f"{placement}_{path}" not in wanted:
            continue
        for name, s in seqs.items():
            run_one(binary, bin_sha, model, model_sha, out, placement, path, name, s["ids"], new_attempts)
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
