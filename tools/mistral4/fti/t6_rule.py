"""PCHONK-FTI-001 T6, contract §11: re-run the frozen rule for another model file, with no edit to any frozen script
(the D-FTI6 re-review's RS3). Reference data only.

  calibrate <ref root> <out dir>
      <ref root>/<variant>/<seq>/logits.f32 for the 6 variants (T4-style runs of ref_logits on the new file and T3's
      sequences). Writes <out>/pairs/<x>__<seq>.npz (each trusted variant against gpu, t6_measure.measure()) and
      <out>/thresholds.json (t6_contract_eval.thresholds(), with its leave-one-out).
  gate <ref root> <out dir> <wrong-weights dir>
      The control gate (§11 step 3), against <out>/thresholds.json. Each must come out as for UD-Q4_K_M:
        leave one out: all 5 trusted variants PASS (from thresholds.json)
        the pairs without gpu (cpu-fa_off, cpu-decode, fa_off-decode; the first is the reference side): PASS
        wrong weights: <wrong-weights dir>/<seq>/logits.f32 (llama.cpp, gpu settings, on another quant) against
          gpu: FAIL
        on gpu's rows: badly wrong rows at 1% of positions (seeded as t6_review_controls.py), scale x0.95, x1.07: FAIL
      Also reports, per sequence, the positions with gpu margin > MARGIN and which thresholds a floor sets (N11).
      Writes <out>/gate.json with "gate": "PASS" or "STOP" (STOP: the rule is not reused for that file; a new T6
      revision with its own derivation is needed).

Self-test (20 November 2026): calibrate on UD-Q4_K_M's own reference must reproduce thresholds_UD-Q4_K_M.json, and
gate with the UD-Q3_K_XL runs as wrong weights must give PASS.

usage: <venv-tools python> -I t6_rule.py calibrate <ref root> <out dir>
       <venv-tools python> -I t6_rule.py gate <ref root> <out dir> <wrong-weights dir>
"""
import json
import sys
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t6_measure import measure  # noqa: E402
import t6_contract_eval as ce  # noqa: E402

SEQ = Path(os.environ.get("PETITCHONK_WORK", ".")) / "fti" / "sequences" / "sequences.json"
V = 131072
PAIRS = [("cpu", "fa_off"), ("cpu", "decode"), ("fa_off", "decode")]


def seqs():
    return json.loads(SEQ.read_text(encoding="utf-8"))["sequences"]


def calibrate(ref, out):
    ref, out = Path(ref), Path(out)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    for name, s in seqs().items():
        for x in ce.TRUSTED:
            np.savez_compressed(out / "pairs" / f"{x}__{name}.npz",
                                **measure(ref / "gpu" / name / "logits.f32", ref / x / name / "logits.f32", s["ids"]))
        print(f"calibrated {name}", flush=True)
    ce.thresholds(out, out / "thresholds.json")


def gate(ref, out, wrong):
    ref, out, wrong = Path(ref), Path(out), Path(wrong)
    th = json.loads((out / "thresholds.json").read_text(encoding="utf-8"))
    g = out / "gate"
    g.mkdir(parents=True, exist_ok=True)
    sq = seqs()
    res = {"leave_one_out": {x: {"worst": v["worst_ratio_to_threshold"], "ok": v["passes"]}
                             for x, v in th["leave_one_out"].items()}}

    def judged(label, ms, expect):
        r = ce.judge(th, ms)
        res[label] = {"verdict": r["verdict"], "expected": expect, "ok": r["verdict"] == expect,
                      "largest_ratios": r["largest_ratios"][:3]}
        print(label, r["verdict"], "expected", expect, flush=True)

    for a, b in PAIRS:
        judged(f"pair_{a}__{b}", {n: measure(ref / a / n / "logits.f32", ref / b / n / "logits.f32", s["ids"])
                                  for n, s in sq.items()}, "PASS")
    judged("wrong_weights", {n: measure(ref / "gpu" / n / "logits.f32", wrong / n / "logits.f32", s["ids"])
                             for n, s in sq.items()}, "FAIL")
    shift = {}
    for si, (n, s) in enumerate(sorted(sq.items())):
        k = len(s["ids"])
        rng = np.random.default_rng(20261121 + si)
        mask = np.zeros(k, dtype=bool)
        mask[rng.choice(k, size=max(1, round(0.01 * k)), replace=False)] = True

        def t(z, r0, r1, a, mask=mask):
            sel = np.nonzero(mask[r0:r1])[0]
            if sel.size:
                z[sel] = np.asarray(a[np.minimum(r0 + sel + 1, a.shape[0] - 1)], dtype=np.float64)
            return z
        shift[n] = measure(ref / "gpu" / n / "logits.f32", None, s["ids"], transform=t)
    judged("local_shift_0.01", shift, "FAIL")
    for f in (0.95, 1.07):
        judged(f"scale_{f}", {n: measure(ref / "gpu" / n / "logits.f32", None, s["ids"],
                                         transform=lambda z, r0, r1, a, f=f: z * f) for n, s in sq.items()}, "FAIL")
    floors = {}
    for n in th["sequences"]:
        p = np.load(out / "pairs" / f"chunk__{n}.npz")
        floors[n] = {"positions_margin_gt": int((p["margin"] > ce.MARGIN).sum()),
                     "set_by_floor": [st for st in ce.PER_SEQ
                                      if th["per_sequence"][n][st] == ce.FLOOR[st]]}
    res["floors"] = floors
    res["pooled_set_by_floor"] = [st for st in ce.POOLED if th["pooled"][st] == ce.FLOOR[st]]
    ok = all(v["ok"] for v in res["leave_one_out"].values()) and \
        all(v["ok"] for k, v in res.items() if isinstance(v, dict) and "expected" in v)
    res["gate"] = "PASS" if ok else "STOP"
    (out / "gate.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("gate", res["gate"], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "calibrate":
        calibrate(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "gate":
        gate(sys.argv[2], sys.argv[3], sys.argv[4])
