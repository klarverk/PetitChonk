"""Exploration: the quant x delta smoke test (pub\\explore\\qd1) against P5's llama.cpp reference for UD-Q4_K_M.

Reads only. For the tokens the smoke test scored (B2 windows w00 w24 w48, target positions 65..575: logp entries
64..574, as logp[j - 1] = log p(ids[j] | ids[:j]) with from = 1), the per-token NLL of every smoke run
(quality\\<quant>\\<window>\\<arm>\\per_token.json) is paired with llama.cpp's (pub\\p5\\run1\\b2_ref_gpu), and the
perplexity ratio exp(mean(NLL_run - NLL_ref)) gets a 95% interval from 10,000 bootstrap resamples (seed 20261122) of
the same 24 blocks of 64 tokens. Yardsticks over the same tokens: llama.cpp CPU (b2_ref_cpu; it holds w00 only of the three) and P5's candidate
(b2_cand, the unsteered engine reading each window as a prompt batch) against llama.cpp GPU. The alignment is checked
first: the correlation of our UD-Q4_K_M lambda 0 per-token NLL with llama.cpp's.

usage: <venv-tools python> -I qd_vs_ref.py
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PETITCHONK_WORK", "."))   # the work folder (tools/mistral4/README.md)
QD = ROOT / r"pub\explore\qd1"
P5 = ROOT / r"pub\p5\run1"
WINDOWS = ("w00", "w24", "w48")
LO, HI = 64, 575            # logp entries 64..574: targets 65..575
QUANTS = ("Q4KM", "Q3KM", "Q3KXL", "Q2KXL")


def ref(name):
    out = {}
    for line in (P5 / name / "logprobs.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["id"] in WINDOWS:
            if r["from"] != 1 or r["n"] != 4096:
                raise SystemExit(f"{name} {r['id']}: from {r['from']}, n {r['n']}")
            lp = r["logp"][LO:HI]
            if any(v is None for v in lp):
                raise SystemExit(f"{name} {r['id']}: a null logp in the range")
            out[r["id"]] = -np.array(lp, dtype=np.float64)
    return out


def ratio(nll, base, rng):
    bd, bn = [], []
    for w in [w for w in WINDOWS if w in nll]:
        d = nll[w] - base[w]
        for s in range(0, len(d), 64):
            bd.append(d[s:s + 64].sum())
            bn.append(len(d[s:s + 64]))
    bd, bn = np.array(bd), np.array(bn)
    pick = rng.integers(0, len(bd), size=(10000, len(bd)))
    boot = np.exp(bd[pick].sum(1) / bn[pick].sum(1))
    return (round(float(np.exp(bd.sum() / bn.sum())), 4),
            round(float(np.quantile(boot, 0.025)), 4), round(float(np.quantile(boot, 0.975)), 4))


def main():
    gpu, cpu, cand = ref("b2_ref_gpu"), ref("b2_ref_cpu"), ref("b2_cand")
    runs = {}
    for q in QUANTS:
        arms = ("A", "A2", "l0.25", "l0.5") if q == "Q4KM" else ("l0", "l0.25", "l0.5")
        for arm in arms:
            runs[(q, arm)] = {w: np.array(json.loads((QD / "quality" / q / w / arm / "per_token.json")
                                                     .read_text(encoding="utf-8"))["nll"]) for w in WINDOWS}
    a = np.concatenate([runs[("Q4KM", "A")][w] for w in WINDOWS])
    g = np.concatenate([gpu[w] for w in WINDOWS])
    print(f"alignment: corr(our Q4KM lambda 0 NLL, llama.cpp GPU NLL) = {np.corrcoef(a, g)[0, 1]:.4f}; "
          f"mean |difference| {np.mean(np.abs(a - g)):.4f} nats over {len(a)} tokens")
    rng = np.random.default_rng(20261122)
    print(f"llama.cpp GPU perplexity on these tokens: {np.exp(g.mean()):.3f}")
    rows = [("llama.cpp CPU (yardstick)", cpu), ("P5 candidate, unsteered batch (yardstick)", cand)]
    rows += [(f"{q} {arm}", runs[(q, arm)]) for (q, arm) in runs]
    res = {}
    for name, nll in rows:
        r = ratio(nll, gpu, rng)
        have = [w for w in WINDOWS if w in nll]
        ppl = float(np.exp(np.concatenate([nll[w] for w in have]).mean()))
        res[name] = {"windows": have, "ppl": round(ppl, 3), "ratio_vs_llamacpp_gpu": r[0], "ci95": [r[1], r[2]]}
        note = "" if len(have) == len(WINDOWS) else f"   (only {', '.join(have)})"
        print(f"{name:45s} perplexity {ppl:7.3f}   ratio vs llama.cpp GPU {r[0]:.4f} ({r[1]:.4f}-{r[2]:.4f}){note}")
    (QD / "vs_llamacpp_ref.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
