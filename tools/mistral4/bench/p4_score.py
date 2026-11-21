"""PCHONK-PUB-001 P4: B2's and B3's measures from the two engines' logprobs.jsonl (P4_benchmark_design.md §2).

Reads only. Both engines' files have ref_logprobs's format ({"id","n","from","logp","argmax","nonfinite"}); a null
logp (a non-finite row) stops the scoring.

B2 (perplexity parity), per engine and paired:
  NLL(w) = -mean(logp) over a window's scored tokens; perplexity = exp(sum of -logp / scored tokens), pooled.
  Paired: d(w) = NLL_cand(w) - NLL_ref(w); its mean, and a 95% percentile bootstrap interval over windows (10,000
  resamples, numpy PCG64 seed 20261121); exp(mean d) is the perplexity ratio. The per-token logp differences'
  quantiles. Secondary: the same on the second half of each window (tokens at positions >= 2048, as llama.cpp's
  perplexity tool scores). With --b2-scale, llama.cpp's own GPU-vs-CPU difference on the windows both cover.
B3 (task parity), per task and pooled:
  acc: the choice with the largest summed logp (np.argmax: the first on ties) is the gold. ARC also acc_norm:
  summed logp / the choice's length in characters. Per-item agreement of the two engines' picks; McNemar's exact
  two-sided test on the items where exactly one engine is right (b, c): p = min(1, 2 * P[Binom(b+c, 1/2) <=
  min(b, c)]). The per-choice summed-logp differences' quantiles.

usage: python -I p4_score.py [--b2 WINDOWS CAND REF [--b2-scale REF_CPU]] [--b3 ITEMS CAND REF] --out OUT.json
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

BOOT, SEED, HALF = 10000, 20261121, 2048
Q = (0.5, 0.9, 0.99, 1.0)


def load(p):
    d = {x["id"]: x for x in (json.loads(l) for l in open(p, encoding="utf-8"))}
    bad = [k for k, x in d.items() if any(v is None for v in x["logp"])]
    if bad:
        raise SystemExit(f"{p}: non-finite rows in {bad[:5]}")
    return d


def quantiles(x):
    x = np.abs(np.asarray(x, dtype=np.float64))
    return {str(q): float(np.quantile(x, q)) for q in Q} if x.size else {}


def paired(ids, a, b, first_pos=None):
    """a, b: {id: record}; first_pos: score only tokens at positions >= first_pos."""
    na, nb, toks, d_tok = [], [], 0, []
    for w in ids:
        la, lb = np.array(a[w]["logp"]), np.array(b[w]["logp"])
        if first_pos is not None:
            k = max(0, first_pos - a[w]["from"])
            la, lb = la[k:], lb[k:]
        na.append(-la.mean()); nb.append(-lb.mean())
        toks += la.size
        d_tok.extend((la - lb).tolist())
    na, nb = np.array(na), np.array(nb)
    sum_a = sum(-np.sum(np.array(a[w]["logp"])[max(0, (first_pos or 0) - a[w]["from"]):]) for w in ids)
    sum_b = sum(-np.sum(np.array(b[w]["logp"])[max(0, (first_pos or 0) - b[w]["from"]):]) for w in ids)
    d = na - nb
    rng = np.random.Generator(np.random.PCG64(SEED))
    boot = d[rng.integers(0, d.size, size=(BOOT, d.size))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"windows": len(ids), "scored_tokens": toks, "ppl_a": math.exp(sum_a / toks), "ppl_b": math.exp(sum_b / toks),
            "mean_d_nll": float(d.mean()), "ci95_d_nll": [float(lo), float(hi)],
            "ppl_ratio": math.exp(d.mean()), "ci95_ppl_ratio": [math.exp(lo), math.exp(hi)],
            "abs_d_logp_token_quantiles": quantiles(d_tok), "mean_d_logp_token": float(np.mean(d_tok))}


def b2(windows, cand, ref, scale):
    ws = [json.loads(l)["id"] for l in open(windows, encoding="utf-8")]
    c, r = load(cand), load(ref)
    out = {"candidate_vs_reference": paired(ws, c, r),
           "candidate_vs_reference_second_half": paired(ws, c, r, first_pos=HALF)}
    if scale:
        s = load(scale)
        both = [w for w in ws if w in s]
        out["reference_gpu_vs_cpu"] = paired(both, r, s)
        out["candidate_vs_reference_on_scale_windows"] = paired(both, c, r)
    return out


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def b3(items_path, cand, ref):
    items = [json.loads(l) for l in open(items_path, encoding="utf-8")]
    c, r = load(cand), load(ref)
    out, pooled = {}, {"n": 0, "acc_c": 0, "acc_r": 0, "agree": 0, "b": 0, "c": 0, "d": []}
    for task in sorted({i["task"] for i in items}):
        t = {"n": 0, "acc_c": 0, "acc_r": 0, "agree": 0, "b": 0, "c": 0, "d": [],
             "accn_c": 0, "accn_r": 0, "agree_n": 0}
        for it in (i for i in items if i["task"] == task):
            key = f"{task}:{it['row']}"
            sc = np.array([sum(c[f"{key}:{k}"]["logp"]) for k in range(len(it["choices"]))])
            sr = np.array([sum(r[f"{key}:{k}"]["logp"]) for k in range(len(it["choices"]))])
            pc, pr = int(np.argmax(sc)), int(np.argmax(sr))
            okc, okr = pc == it["gold"], pr == it["gold"]
            t["n"] += 1; t["acc_c"] += okc; t["acc_r"] += okr; t["agree"] += pc == pr
            t["b"] += okc and not okr; t["c"] += okr and not okc
            t["d"].extend((sc - sr).tolist())
            if task == "arc_challenge":
                ln = np.array([len(ch) for ch in it["choices"]], dtype=np.float64)
                qc, qr = int(np.argmax(sc / ln)), int(np.argmax(sr / ln))
                t["accn_c"] += qc == it["gold"]; t["accn_r"] += qr == it["gold"]; t["agree_n"] += qc == qr
        res = {"items": t["n"], "acc_candidate": t["acc_c"] / t["n"], "acc_reference": t["acc_r"] / t["n"],
               "agreement": t["agree"] / t["n"], "only_candidate_right": t["b"], "only_reference_right": t["c"],
               "mcnemar_p": mcnemar(t["b"], t["c"]), "abs_d_choice_logp_quantiles": quantiles(t["d"])}
        if task == "arc_challenge":
            res.update({"acc_norm_candidate": t["accn_c"] / t["n"], "acc_norm_reference": t["accn_r"] / t["n"],
                        "agreement_norm": t["agree_n"] / t["n"]})
        out[task] = res
        for k in ("n", "acc_c", "acc_r", "agree", "b", "c"):
            pooled[k] += t[k]
        pooled["d"].extend(t["d"])
    out["pooled"] = {"items": pooled["n"], "acc_candidate": pooled["acc_c"] / pooled["n"],
                     "acc_reference": pooled["acc_r"] / pooled["n"], "agreement": pooled["agree"] / pooled["n"],
                     "only_candidate_right": pooled["b"], "only_reference_right": pooled["c"],
                     "mcnemar_p": mcnemar(pooled["b"], pooled["c"]),
                     "abs_d_choice_logp_quantiles": quantiles(pooled["d"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b2", nargs=3, metavar=("WINDOWS", "CAND", "REF"))
    ap.add_argument("--b2-scale")
    ap.add_argument("--b3", nargs=3, metavar=("ITEMS", "CAND", "REF"))
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    res = {}
    if a.b2:
        res["B2"] = b2(*a.b2, a.b2_scale)
    if a.b3:
        res["B3"] = b3(*a.b3)
    a.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
