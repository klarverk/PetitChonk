"""PCHONK-PUB-001 P4: the candidate side of B2's and B3's scoring (option (a): no engine change; the user, 20 Nov).

Runs the release binary (`strata-poly`, mistral4) on JSON-lines sequences {"id","ids","from"} (freeze_a.py's
format) and writes, per sequence, the record ref_logprobs writes for llama.cpp:
  {"id","n","from","logp","argmax","nonfinite"}
logp[j - from] = log p(ids[j] | ids[:j]) from logits row j-1, in double precision over the full vocabulary:
m = max, lse = m + log(sum(exp(r - m))), logp = r[target] - lse; argmax = the lowest id holding the maximum. A row
with a non-finite logit gives logp null and argmax -1, and "nonfinite" counts the non-finite logits of the scored
rows. (ref_logprobs.cpp's header states the same rule.)

Modes:
  serve (default)  one `strata-poly --serve` process per batch. Each sequence is one `GEN 1 <ids>` request, longest
                   first, so no request extends the live sequence; --prompt-cache 0, so there are no checkpoints.
                   Every reply must report RESUME 0, and its DONE line reused 0 (field 8) and prompt tokens
                   read = n (field 14).
  one-shot         one process per sequence: --tokens-file <ids> --max-new 1 (T8's whole path without follow ids).
Both: STRATA_LOGITS_DIR=<batch>/dump and STRATA_LOGITS_ALL_ROWS=1 (every prompt row dumped); --max-context C
--lookup 0 --window 1 --eos-ids 0 --prompt-batch 1024, as T8's whole path, with the default placement; no other
STRATA_* variable except STRATA_FREEZE_EXPERTS=1 with --freeze-experts; CUDA 13.0 on PATH.
A request's rows in logits.jsonl must be exactly n, in order, with pos 0..n-1 and token == ids[pos], and the rows of
a batch must be exactly its requests'. Anything else stops the run.

Option (a): each batch's dump is hashed (SHA-256 of logits.f32 and logits.jsonl, into batch.json), scored, and then
deleted, so at most one batch's rows are on disk. --keep-rows ID keeps that sequence's rows (rows_<ID>.f32, ':' in
the id written as '_') for the qualification. Without --execute it prints the plan and runs nothing.

usage: <venv-tools python> -I p4_cand.py [--execute] --binary B --model M --in SEQ.jsonl --out DIR
           [--mode serve|one-shot] [--batch-rows N] [--max-context C] [--freeze-experts] [--keep-rows ID]...
The out dir must not exist.
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CUDA = os.environ.get("PETITCHONK_CUDA", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")   # 13.0: never 13.2.x on RTX 50
V = 131072
ROW_BYTES = 4 * V
MARGIN = 32 << 30              # free space kept on top of a batch's dump
FLAGS = ["--lookup", "0", "--window", "1", "--eos-ids", "0", "--prompt-batch", "1024"]
PLACEMENTS = {"default": [], "cpu": ["--no-gpu-experts"]}   # T8's; cpu only for the qualification (no cache)


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def read_seqs(path):
    seqs, seen = [], set()
    for k, line in enumerate(open(path, encoding="utf-8")):
        if not line.strip():
            continue
        s = json.loads(line)
        if set(s) != {"id", "ids", "from"}:
            raise SystemExit(f"line {k + 1}: keys {sorted(s)}")
        n, f = len(s["ids"]), s["from"]
        if s["id"] in seen or not (1 <= f <= n - 1) or not all(isinstance(t, int) and 0 <= t < V for t in s["ids"]):
            raise SystemExit(f"line {k + 1}: bad sequence {s['id']}")
        seen.add(s["id"])
        seqs.append(s)
    return seqs


def plan(seqs, mode, batch_rows):
    order = sorted(range(len(seqs)), key=lambda i: (-len(seqs[i]["ids"]), i))   # longest first, then input order
    if mode == "one-shot":
        return [[i] for i in order]
    batches, cur, rows = [], [], 0
    for i in order:
        n = len(seqs[i]["ids"])
        if cur and rows + n > batch_rows:
            batches.append(cur)
            cur, rows = [], 0
        cur.append(i)
        rows += n
    return batches + ([cur] if cur else [])


def environment(dump, freeze):
    e = {k: v for k, v in os.environ.items() if not k.upper().startswith("STRATA_")}
    e["PATH"] = rf"{CUDA}\bin\x64;{CUDA}\bin;" + e["PATH"]
    strata = {"STRATA_LOGITS_DIR": str(dump), "STRATA_LOGITS_ALL_ROWS": "1"}
    if freeze:
        strata["STRATA_FREEZE_EXPERTS"] = "1"
    e.update(strata)
    return e, strata


def run_serve(a, seqs, batch, d):
    dump = d / "dump"
    env, strata = environment(dump, a.freeze_experts)
    argv = [a.binary, "--model", a.model, "--serve", "--max-context", str(a.max_context)] + FLAGS + \
        ["--prompt-cache", "0"] + PLACEMENTS[a.placement]
    rec = {"argv": argv, "strata_env": strata, "start": now(), "requests": []}
    with open(d / "stderr.txt", "w", encoding="utf-8", errors="replace") as err, \
            open(d / "stdout.txt", "w", encoding="utf-8", newline="\n") as out:
        p = subprocess.Popen(argv, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)

        def line():
            s = p.stdout.readline()
            if not s:
                raise SystemExit(f"{d.name}: the engine closed its output (exit {p.wait()})")
            out.write(s)
            return s.rstrip("\n")

        while not (l := line()).startswith("READY"):
            pass
        rec["ready"] = l
        for i in batch:
            s = seqs[i]
            p.stdin.write("GEN 1 " + ",".join(map(str, s["ids"])) + "\n")
            p.stdin.flush()
            r = {"id": s["id"], "n": len(s["ids"]), "resume": None, "tokens": []}
            while True:
                l = line()
                if l.startswith("ERR"):
                    raise SystemExit(f"{d.name}: {s['id']}: {l}")
                if l.startswith("RESUME "):
                    r["resume"] = int(l.split()[1])
                elif l.startswith("T "):
                    r["tokens"].append(int(l.split()[1]))
                elif l.startswith("DONE "):
                    f = l.split()
                    r["done"] = l
                    r["reused"], r["read"] = int(f[8]), int(f[14])
                    break
            if r["resume"] != 0 or r["reused"] != 0 or r["read"] != r["n"]:
                raise SystemExit(f"{d.name}: {s['id']}: resume {r['resume']}, reused {r['reused']}, "
                                 f"read {r['read']} of {r['n']}")
            rec["requests"].append(r)
        p.stdin.write("QUIT\n")
        p.stdin.close()
        rec["exit"] = p.wait(timeout=600)
    rec["end"] = now()
    if rec["exit"] != 0:
        raise SystemExit(f"{d.name}: exit {rec['exit']}")
    return rec


def run_one_shot(a, seqs, batch, d):
    (i,) = batch
    dump = d / "dump"
    env, strata = environment(dump, a.freeze_experts)
    tf = d / "tokens.txt"
    tf.write_text(",".join(map(str, seqs[i]["ids"])), encoding="utf-8")
    argv = [a.binary, "--model", a.model, "--tokens-file", str(tf), "--max-new", "1",
            "--max-context", str(a.max_context)] + FLAGS + PLACEMENTS[a.placement]
    rec = {"argv": argv, "strata_env": strata, "start": now()}
    r = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, capture_output=True)
    (d / "stdout.txt").write_bytes(r.stdout)
    (d / "stderr.txt").write_bytes(r.stderr)
    rec.update({"end": now(), "exit": r.returncode, "requests": [{"id": seqs[i]["id"], "n": len(seqs[i]["ids"])}]})
    if r.returncode != 0:
        raise SystemExit(f"{d.name}: exit {r.returncode}")
    return rec


def score(rows, ids, frm):
    """rows: the sequence's n rows (float32, n x V). The rule of the module docstring."""
    logp, argmax, nonfinite = [], [], 0
    for a in range(frm - 1, len(ids) - 1, 256):
        b = min(a + 256, len(ids) - 1)
        r = np.asarray(rows[a:b], dtype=np.float64)
        fin = np.isfinite(r)
        for k in range(b - a):
            j = a + k + 1
            if not fin[k].all():
                nonfinite += int((~fin[k]).sum())
                logp.append(None)
                argmax.append(-1)
                continue
            m = r[k].max()
            lse = m + np.log(np.exp(r[k] - m).sum())
            logp.append(float(r[k][ids[j]] - lse))
            argmax.append(int(np.argmax(r[k])))
    return logp, argmax, nonfinite


def take_batch(a, seqs, batch, d, rec, results, keep):
    dump = d / "dump"
    ixp, fp = dump / "logits.jsonl", dump / "logits.f32"
    ix = [json.loads(l) for l in open(ixp, encoding="utf-8")]
    if fp.stat().st_size != len(ix) * ROW_BYTES:
        raise SystemExit(f"{d.name}: logits.f32 holds {fp.stat().st_size} bytes for {len(ix)} rows")
    rec["dump"] = {"rows": len(ix), "bytes": fp.stat().st_size, "logits.f32": sha_file(fp),
                   "logits.jsonl": sha_file(ixp)}
    rows = np.memmap(fp, dtype=np.float32, mode="r").reshape(-1, V)
    at = 0
    for i in batch:
        s = seqs[i]
        ids, n = s["ids"], len(s["ids"])
        part = ix[at:at + n]
        if len(part) != n or any(x["row"] != at + j or x["pos"] != j or x["token"] != ids[j]
                                 for j, x in enumerate(part)):
            raise SystemExit(f"{d.name}: {s['id']}: its rows are not pos 0..{n - 1} with its tokens")
        lp, am, nf = score(rows[at:at + n], ids, s["from"])
        results[s["id"]] = {"id": s["id"], "n": n, "from": s["from"], "logp": lp, "argmax": am, "nonfinite": nf}
        if s["id"] in keep:
            name = "rows_" + s["id"].replace(":", "_") + ".f32"
            np.asarray(rows[at:at + n]).tofile(a.out / name)
            rec.setdefault("kept_rows", {})[s["id"]] = name
        at += n
    if at != len(ix):
        raise SystemExit(f"{d.name}: {len(ix) - at} rows beyond its requests'")
    del rows
    with open(d / "logprobs_part.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for i in batch:
            f.write(json.dumps(results[seqs[i]["id"]], separators=(",", ":")) + "\n")
    shutil.rmtree(dump)
    rec["dump_deleted"] = now()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=["serve", "one-shot"], default="serve")
    ap.add_argument("--batch-rows", type=int, default=150000)
    ap.add_argument("--max-context", type=int, default=16384)
    ap.add_argument("--freeze-experts", action="store_true")
    ap.add_argument("--placement", choices=sorted(PLACEMENTS), default="default")
    ap.add_argument("--keep-rows", action="append", default=[])
    ap.add_argument("--only", action="append", default=[], help="run only these ids (qualification)")
    a = ap.parse_args()
    seqs = read_seqs(a.inp)
    if a.only:
        missing = set(a.only) - {s["id"] for s in seqs}
        if missing:
            raise SystemExit(f"--only: no such id {sorted(missing)}")
        seqs = [s for s in seqs if s["id"] in set(a.only)]
    batches = plan(seqs, a.mode, a.batch_rows)
    if max(len(s["ids"]) for s in seqs) + 1 > a.max_context:
        raise SystemExit("a sequence does not fit --max-context")
    unknown = set(a.keep_rows) - {s["id"] for s in seqs}
    if unknown:
        raise SystemExit(f"--keep-rows: no such id {sorted(unknown)}")
    need = max(sum(len(seqs[i]["ids"]) for i in b) for b in batches) * ROW_BYTES
    print(json.dumps({"sequences": len(seqs), "batches": len(batches), "rows": sum(len(s["ids"]) for s in seqs),
                      "largest_batch_GiB": round(need / 2**30, 1), "mode": a.mode}, indent=1))
    if not a.execute:
        return
    a.out.mkdir(parents=True, exist_ok=False)
    run = {"input": a.inp, "input_sha256": sha_file(a.inp), "binary": a.binary, "binary_sha256": sha_file(a.binary),
           "model": a.model, "mode": a.mode, "batch_rows": a.batch_rows, "max_context": a.max_context,
           "freeze_experts": a.freeze_experts, "placement": a.placement, "only": a.only, "start": now(),
           "batches": len(batches)}
    (a.out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    results, keep = {}, set(a.keep_rows)
    for k, batch in enumerate(batches):
        d = a.out / f"batch_{k:04d}"
        d.mkdir()
        rows = sum(len(seqs[i]["ids"]) for i in batch)
        free = shutil.disk_usage(a.out).free
        if free < rows * ROW_BYTES + MARGIN:
            raise SystemExit(f"{d.name}: {free / 2**30:.0f} GiB free, {rows * ROW_BYTES / 2**30:.0f} GiB needed + margin")
        t0 = time.time()
        rec = (run_serve if a.mode == "serve" else run_one_shot)(a, seqs, batch, d)
        take_batch(a, seqs, batch, d, rec, results, keep)
        rec["seconds"] = round(time.time() - t0, 1)
        (d / "batch.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{d.name}: {len(batch)} sequences, {rows} rows, {rec['seconds']} s", flush=True)
    with open(a.out / "logprobs.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for s in seqs:
            f.write(json.dumps(results[s["id"]], separators=(",", ":")) + "\n")
    run["end"] = now()
    run["logprobs_sha256"] = sha_file(a.out / "logprobs.jsonl")
    (a.out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
