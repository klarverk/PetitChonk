"""Logic test of p4_cand.py against stub_engine.py (no model). Prints each check and exits 1 on any failure.

usage: <venv-tools python> -I test_p4_cand.py <empty scratch dir>
"""
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE.parent / "p4_cand.py"
sys.path.insert(0, str(HERE))
import stub_engine  # noqa: E402

PY = sys.executable
SEQS = [{"id": f"t:{k}:0", "ids": [1] + [(37 * k + 11 * j) % 5000 + 3 for j in range(n - 1)], "from": f}
        for k, (n, f) in enumerate([(5, 1), (12, 3), (8, 7), (9, 2), (6, 4)])]


def independent(ids, frm):
    """logp by math.fsum over the stub's rows, argmax as the lowest maximal id: a second implementation."""
    lp, am = [], []
    for j in range(frm, len(ids)):
        r = stub_engine.row(ids[:j])
        m = max(r)
        lse = m + math.log(math.fsum(math.exp(v - m) for v in r))
        lp.append(r[ids[j]] - lse)
        am.append(r.index(m))
    return lp, am


def cand(work, name, *extra, fault=None):
    out = work / name
    env = None
    if fault:
        import os
        env = dict(os.environ, STUB_FAULT=fault)
    r = subprocess.run([PY, "-I", str(CAND), "--execute", "--binary", str(work / "stub.bat"), "--model", "stub.gguf",
                        "--in", str(work / "seqs.jsonl"), "--out", str(out)] + list(extra),
                       capture_output=True, text=True, env=env)
    return r, out


def main(work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=False)
    (work / "stub.bat").write_text(f'@"{PY}" -I "{HERE / "stub_engine.py"}" %*\r\n', encoding="ascii")
    with open(work / "seqs.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for s in SEQS:
            f.write(json.dumps(s) + "\n")
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail and not ok else ""))

    r, out = cand(work, "serve", "--keep-rows", "t:1:0")
    check("serve run exits 0", r.returncode == 0, r.stderr[-400:])
    res = [json.loads(l) for l in open(out / "logprobs.jsonl", encoding="utf-8")]
    check("output in input order", [x["id"] for x in res] == [s["id"] for s in SEQS])
    worst, am_ok = 0.0, True
    for s, x in zip(SEQS, res):
        lp, am = independent(s["ids"], s["from"])
        worst = max([worst] + [abs(a - b) for a, b in zip(lp, x["logp"])])
        am_ok &= am == x["argmax"] and len(lp) == len(x["logp"]) and x["nonfinite"] == 0
    check("logp equals an independent computation (|d| <= 1e-9)", worst <= 1e-9, worst)
    check("argmax and lengths equal", am_ok)
    kept = (out / "rows_t_1_0.f32").read_bytes()
    want = b"".join(stub_engine.row(SEQS[1]["ids"][:j + 1]).tobytes() for j in range(len(SEQS[1]["ids"])))
    check("kept rows are the sequence's rows", kept == want)
    check("dumps deleted", not list(out.glob("batch_*/dump")))
    bj = json.loads((out / "batch_0000" / "batch.json").read_text(encoding="utf-8"))
    check("run order is longest first", [q["n"] for q in bj["requests"]] == sorted((len(s["ids"]) for s in SEQS),
                                                                                    reverse=True))

    r2, out2 = cand(work, "one_shot", "--mode", "one-shot")
    check("one-shot run exits 0", r2.returncode == 0, r2.stderr[-400:])
    check("one-shot logprobs.jsonl == serve's", (out2 / "logprobs.jsonl").read_bytes() ==
          (out / "logprobs.jsonl").read_bytes())

    r3, out3 = cand(work, "batched", "--batch-rows", "15")
    check("batched run exits 0, several batches", r3.returncode == 0 and len(list(out3.glob("batch_*"))) > 1,
          r3.stderr[-400:])
    check("batched logprobs.jsonl == serve's", (out3 / "logprobs.jsonl").read_bytes() ==
          (out / "logprobs.jsonl").read_bytes())

    r4, _ = cand(work, "f_reused", fault="reused")
    check("a reused prefix stops the run", r4.returncode != 0 and "reused 1" in r4.stderr, r4.stderr[-300:])
    r5, _ = cand(work, "f_shift", fault="shift")
    check("misplaced rows stop the run", r5.returncode != 0 and "not pos" in r5.stderr, r5.stderr[-300:])
    r6, out6 = cand(work, "f_nonfinite", fault="nonfinite")
    res6 = [json.loads(l) for l in open(out6 / "logprobs.jsonl", encoding="utf-8")] if r6.returncode == 0 else []
    ok6 = r6.returncode == 0 and all(
        (x["nonfinite"] == 1 and x["logp"][2 - s["from"]] is None and x["argmax"][2 - s["from"]] == -1)
        if s["from"] <= 2 else x["nonfinite"] == 0 for s, x in zip(SEQS, res6))
    check("a non-finite row gives null, -1 and a count", ok6, r6.stderr[-300:])
    r7 = subprocess.run([PY, "-I", str(CAND), "--binary", str(work / "stub.bat"), "--model", "m", "--in",
                         str(work / "seqs.jsonl"), "--out", str(work / "plan_only")], capture_output=True, text=True)
    check("without --execute nothing runs", r7.returncode == 0 and not (work / "plan_only").exists())
    print(f"{sum(checks)} of {len(checks)} checks pass")
    sys.exit(0 if all(checks) else 1)


if __name__ == "__main__":
    main(sys.argv[1])
