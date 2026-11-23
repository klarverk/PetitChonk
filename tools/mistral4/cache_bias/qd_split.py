"""Exploration: split the smoke test's quality ratios into engine error and the quantization's own loss.

Reads only: the smoke runs' per-token NLL (pub\\explore\\qd1\\quality), llama.cpp's reference for UD-Q4_K_M from P5
(pub\\p5\\run1\\b2_ref_gpu) and the new llama.cpp references per quant (pub\\explore\\qd1_ref\\<quant>, P5's runner,
variant gpu). Tokens: B2 windows w00 w24 w48, targets 65..575 (logp entries 64..574). Ratios exp(mean(NLL_x -
NLL_y)) with 95% intervals from 10,000 bootstrap resamples (seed 20261122) of 24 blocks of 64 tokens.

  1. repeatability: llama.cpp UD-Q4_K_M now against P5's
  2. engine error:  our engine at lambda 0 against llama.cpp on the same quant ("no engine error seen" if the
                    interval contains 1.0, else "flag: follow up")
  3. quant loss:    llama.cpp on the quant against llama.cpp UD-Q4_K_M (P5's)
  4. delta loss:    our lambda 0.25 and 0.5 against llama.cpp on the same quant

usage: <venv-tools python> -I qd_split.py
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PETITCHONK_WORK", "."))   # the work folder (tools/mistral4/README.md)
QD = ROOT / r"pub\explore\qd1"
REFS = ROOT / r"pub\explore\qd1_ref"
P5 = ROOT / r"pub\p5\run1\b2_ref_gpu"
WINDOWS = ("w00", "w24", "w48")
LO, HI = 64, 575
QUANTS = ("Q4KM", "Q3KM", "Q3KXL", "Q2KXL")


def ref(path):
    out = {}
    for line in (path / "logprobs.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["id"] in WINDOWS:
            if r["from"] != 1 or r["n"] != 4096 or r.get("nonfinite"):
                raise SystemExit(f"{path} {r['id']}: from {r['from']}, n {r['n']}, nonfinite {r.get('nonfinite')}")
            lp = r["logp"][LO:HI]
            if any(v is None for v in lp):
                raise SystemExit(f"{path} {r['id']}: a null logp")
            out[r["id"]] = -np.array(lp, dtype=np.float64)
    if set(out) != set(WINDOWS):
        raise SystemExit(f"{path}: windows {sorted(out)}")
    return out


def ours(q, arm):
    return {w: np.array(json.loads((QD / "quality" / q / w / arm / "per_token.json").read_text(encoding="utf-8"))
                        ["nll"]) for w in WINDOWS}


def ratio(x, y, rng):
    bd, bn = [], []
    for w in WINDOWS:
        d = x[w] - y[w]
        for s in range(0, len(d), 64):
            bd.append(d[s:s + 64].sum())
            bn.append(len(d[s:s + 64]))
    bd, bn = np.array(bd), np.array(bn)
    pick = rng.integers(0, len(bd), size=(10000, len(bd)))
    boot = np.exp(bd[pick].sum(1) / bn[pick].sum(1))
    return [round(float(np.exp(bd.sum() / bn.sum())), 4), round(float(np.quantile(boot, 0.025)), 4),
            round(float(np.quantile(boot, 0.975)), 4)]


def main():
    rng = np.random.default_rng(20261122)
    p5 = ref(P5)
    lc = {q: ref(REFS / q) for q in QUANTS}
    res = {"repeatability": ratio(lc["Q4KM"], p5, rng), "quants": {}}
    print("1. llama.cpp UD-Q4_K_M now vs P5's: %.4f (%.4f-%.4f)" % tuple(res["repeatability"]))
    for q in QUANTS:
        lam0 = [("A", ours(q, "A")), ("A2", ours(q, "A2"))] if q == "Q4KM" else [("l0", ours(q, "l0"))]
        r = {"engine_error": {}, "quant_loss": ratio(lc[q], p5, rng), "delta_loss": {}}
        for name, x in lam0:
            e = ratio(x, lc[q], rng)
            r["engine_error"][name] = e + ["no engine error seen" if e[1] <= 1.0 <= e[2] else "flag: follow up"]
        for arm in ("l0.25", "l0.5"):
            r["delta_loss"][arm] = ratio(ours(q, arm), lc[q], rng)
        r["llamacpp_ppl"] = round(float(np.exp(np.concatenate([lc[q][w] for w in WINDOWS]).mean())), 3)
        res["quants"][q] = r
        print(f"{q}: llama.cpp perplexity {r['llamacpp_ppl']}")
        for name, e in r["engine_error"].items():
            print("   2. engine error (ours %s vs llama.cpp same quant): %.4f (%.4f-%.4f)  %s" % (name, *e))
        print("   3. quant loss (llama.cpp %s vs llama.cpp Q4):        %.4f (%.4f-%.4f)" % (q, *r["quant_loss"]))
        for arm, d in r["delta_loss"].items():
            print("   4. delta loss (ours %s vs llama.cpp same quant):  %.4f (%.4f-%.4f)" % (arm, *d))
    (REFS / "split.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
