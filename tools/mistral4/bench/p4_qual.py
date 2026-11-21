"""PCHONK-PUB-001 P4: judge the harness qualification (P4_benchmark_design.md §3.2). Reads only; runs no model.

The qualification runs (on qual_all.jsonl, never on frozen items or FTI's sequences), each in its own folder under
<qual root>:
  ref_all            p4_ref gpu, --ctx 2560, --rows tail, X1, X2
  ref_x              p4_ref gpu, --ctx 2560, on X1 alone (a fresh process, X1 its first sequence), --rows X1
  ref_logits_x       T4's ref_logits on X1's ids, T4's gpu options, --ctx 2560 --batch 2560
  cand_p5cfg         p4_cand serve, default placement, no freeze (P5's configuration), --keep-rows tail, X1, X2
  cand_first_serve   p4_cand serve, STRATA_FREEZE_EXPERTS=1, --only tail (tail the first request), --keep-rows tail
  cand_first_oneshot p4_cand one-shot, STRATA_FREEZE_EXPERTS=1, --only tail, --keep-rows tail
  cand_cpu_serve     p4_cand serve, --placement cpu, all of qual_all, --keep-rows X1, X2
  cand_cpu_oneshot   p4_cand one-shot, --placement cpu, --only X1, X2, --keep-rows X1, X2
X1 = arc_challenge:701:2 and X2 = arc_challenge:594:1: the last two in the serve run order (longest first).

Pass rules (fixed before the runs):
  QR1 ref scoring: from ref_all's rows of tail, X1 and X2, logp recomputed a second way (float64 logaddexp.reduce)
      equals ref_logprobs's within 1e-9 absolute, and argmax (the lowest maximal id) is equal.
  QR2 ref after a clear = fresh: ref_all's rows of X1 (after the 7 sequences before it in the input, the 2,542-token
      tail among them) == ref_x's rows of X1, bytes.
  QR3 ref_logprobs = T4's ref_logits: ref_x's rows of X1 == ref_logits_x's logits.f32, bytes.
  QR4 ref finite: ref_all has no non-finite logit in any scored row.
  QC1 serve path = one-shot path: cand_first_serve's rows of tail == cand_first_oneshot's, bytes.
  QC2 nothing carried between requests: cand_cpu_serve's rows of X1 and X2 (the 40th and 39th requests) ==
      cand_cpu_oneshot's, bytes (the cpu placement has no expert cache, so they must agree exactly).
  QC3 candidate scoring: from cand_p5cfg's kept rows, logp recomputed the second way equals p4_cand's within 1e-9,
      and argmax is equal.
  QC4 candidate run: cand_p5cfg completed (so every request reported RESUME 0, reused 0 and read n, and every row
      mapped), and no scored row has a non-finite logit.
Report only (no pass rule): the candidate's against the reference's per-token logp on qual_all, and the 9 items'
answers by each engine.

usage: <venv-tools python> -I p4_qual.py <qual root> <freeze dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

V = 131072
X1, X2 = "arc_challenge:701:2", "arc_challenge:594:1"


def lp_file(d):
    return {x["id"]: x for x in (json.loads(l) for l in open(d / "logprobs.jsonl", encoding="utf-8"))}


def second_way(rows, ids, frm):
    lp, am = [], []
    for j in range(frm, len(ids)):
        r = np.asarray(rows[j - 1], dtype=np.float64)
        lp.append(float(r[ids[j]] - np.logaddexp.reduce(r)))
        am.append(int(np.flatnonzero(r == r.max())[0]))
    return lp, am


def compare(rec, rows, seq):
    lp, am = second_way(rows, seq["ids"], seq["from"])
    d = max(abs(a - b) for a, b in zip(lp, rec["logp"]))
    return d <= 1e-9 and am == rec["argmax"] and len(lp) == len(rec["logp"]), d


def rows_of(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, V)


def ref_rows(d, sid):
    return d / lp_file(d)[sid]["rows"]


def cand_rows(d, sid):
    return d / ("rows_" + sid.replace(":", "_") + ".f32")


def main(root, freeze):
    root, freeze = Path(root), Path(freeze)
    seqs = {s["id"]: s for s in (json.loads(l) for l in open(freeze / "qual_all.jsonl", encoding="utf-8"))}
    v = {}
    ra = lp_file(root / "ref_all")
    res = [compare(ra[s], rows_of(ref_rows(root / "ref_all", s)), seqs[s]) for s in ("tail", X1, X2)]
    v["QR1"] = {"pass": all(ok for ok, _ in res), "max_abs_diff": max(d for _, d in res)}
    a = ref_rows(root / "ref_all", X1).read_bytes()
    v["QR2"] = {"pass": a == ref_rows(root / "ref_x", X1).read_bytes()}
    v["QR3"] = {"pass": ref_rows(root / "ref_x", X1).read_bytes() ==
                (root / "ref_logits_x" / "logits.f32").read_bytes()}
    v["QR4"] = {"pass": all(x["nonfinite"] == 0 for x in ra.values()) and len(ra) == len(seqs)}
    v["QC1"] = {"pass": cand_rows(root / "cand_first_serve", "tail").read_bytes() ==
                cand_rows(root / "cand_first_oneshot", "tail").read_bytes()}
    v["QC2"] = {"pass": all(cand_rows(root / "cand_cpu_serve", s).read_bytes() ==
                            cand_rows(root / "cand_cpu_oneshot", s).read_bytes() for s in (X1, X2))}
    cp = lp_file(root / "cand_p5cfg")
    res = [compare(cp[s], rows_of(cand_rows(root / "cand_p5cfg", s)), seqs[s]) for s in ("tail", X1, X2)]
    v["QC3"] = {"pass": all(ok for ok, _ in res), "max_abs_diff": max(d for _, d in res)}
    run = json.loads((root / "cand_p5cfg" / "run.json").read_text(encoding="utf-8"))
    v["QC4"] = {"pass": "logprobs_sha256" in run and len(cp) == len(seqs) and
                all(x["nonfinite"] == 0 for x in cp.values())}
    # report only: candidate (P5 configuration) against the reference (gpu)
    d = np.array([c - r for s in seqs for c, r in zip(cp[s]["logp"], ra[s]["logp"])])
    agree = np.mean([c == r for s in seqs for c, r in zip(cp[s]["argmax"], ra[s]["argmax"])])
    items = [json.loads(l) for l in open(freeze / "qual_b3_items.jsonl", encoding="utf-8")]
    answers = []
    for it in items:
        key = f"{it['task']}:{it['row']}"
        pick = {}
        for eng, lp in (("cand", cp), ("ref", ra)):
            sums = [sum(lp[f"{key}:{k}"]["logp"]) for k in range(len(it["choices"]))]
            pick[eng] = int(np.argmax(sums))
        answers.append({"item": key, "gold": it["gold"], **pick})
    v["report"] = {"scored_tokens": int(d.size), "abs_dlogp_quantiles": {q: float(np.quantile(np.abs(d), q / 100))
                                                                         for q in (50, 90, 99, 100)},
                   "mean_dlogp": float(d.mean()), "argmax_agreement": float(agree), "items": answers}
    v["all_pass"] = all(v[k]["pass"] for k in ("QR1", "QR2", "QR3", "QR4", "QC1", "QC2", "QC3", "QC4"))
    print(json.dumps(v, indent=1))
    (root / "qual_verdict.json").write_text(json.dumps(v, indent=1), encoding="utf-8")
    sys.exit(0 if v["all_pass"] else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
