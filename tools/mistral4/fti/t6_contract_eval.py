"""PCHONK-FTI-001 T6/T8: the contract's rule, as code. Thresholds come from a model file's reference runs; a run is
judged against them. Frozen with the contract (ML/S8 practice: the rule is the code plus its parameters).

THRESHOLDS (re-runnable for any model file): from the per-position measures of every trusted variant x in TRUSTED
against the primary gpu (t6_measure.measure(), saved as <calib>/pairs/<x>__<seq>.npz):
  per sequence s:   V1 kl_median, V2 kl_p90, V3 flip_rate (top-1 differs, at positions where gpu's top-1/top-2 margin
                    exceeds MARGIN = 1.0 logit)
  pooled over all sequences:  V4 |mean(log p_x(next) - log p_gpu(next))|,  V5 |mean(H_x - H_gpu)|, and V4t, V5t, the
                    same over the tail positions n-TAIL-1 .. n-1 of each sequence (TAIL = 256)
  threshold = max(K * max over trusted x of the statistic, FLOOR[statistic]),  K = 3
VERDICT for a judged run set (one candidate variant over all sequences): PASS iff every exact rule holds and every
statistic is <= its threshold (V1-V3 for every sequence; V4, V5, V4t, V5t pooled over the run set's sequences).
Otherwise FAIL, with every exceeded statistic listed (ratio = value / threshold).

usage:
  <venv-tools python> -I t6_contract_eval.py thresholds <calib dir> <out json>
  <venv-tools python> -I t6_contract_eval.py judge-measures <thresholds json> <measure dir> <prefix> <out json>
      (a run set already measured: <measure dir>/<prefix>__<seq>.npz; used for the controls)
  <venv-tools python> -I t6_contract_eval.py judge-run-set <thresholds json> <ref root> <run set dir> <binary sha256>
      <model json> <out json>
      (T8: a candidate run set written by t8_run.py; the folder's name is the run set, e.g. default_whole; the
      binary's SHA-256 and the model json ({shard file name: SHA-256}) are T7's)
"""
import glob
import hashlib
import json
import sys
import os
from pathlib import Path

import numpy as np

K = 3.0
MARGIN = 1.0
TRUSTED = ["chunk", "fa_off", "unfused", "cpu", "decode"]
TAIL = 256
FLOOR = {"V1_kl_median": 1e-5, "V2_kl_p90": 1e-4, "V3_flip_rate": 1e-3, "V4_abs_mean_dlogp": 1e-3,
         "V5_abs_mean_dH": 1e-3, "V4t_tail_abs_mean_dlogp": 1e-3, "V5t_tail_abs_mean_dH": 1e-3}
PER_SEQ = ["V1_kl_median", "V2_kl_p90", "V3_flip_rate"]
# V4t, V5t (the D-FTI6 re-review's RS1): V4 and V5 over the tail positions only, n-TAIL-1 .. n-1 of every sequence,
# the rows a whole/chunk/split run computes in single-token steps after the prompt read (contract §3). Pooled over
# the 7 sequences: 1,799 positions (1,792 with a next token).
POOLED = ["V4_abs_mean_dlogp", "V5_abs_mean_dH", "V4t_tail_abs_mean_dlogp", "V5t_tail_abs_mean_dH"]


def per_seq_stats(m):
    sel = m["margin"] > MARGIN
    return {"V1_kl_median": float(np.median(m["kl"])), "V2_kl_p90": float(np.percentile(m["kl"], 90)),
            "V3_flip_rate": float(np.mean(~m["top1eq"][sel])) if sel.any() else 0.0}


def _signed(ms):
    dl = np.concatenate([(m["lpx_next"] - m["lp_next"])[np.isfinite(m["lp_next"]) & np.isfinite(m["lpx_next"])]
                         for m in ms])
    dh = np.concatenate([m["h_x"] - m["h_ref"] for m in ms])
    return float(dl.mean()), float(dh.mean()), int(dh.size)


def _tail(m):
    n = m["kl"].size
    return {k: m[k][n - TAIL - 1:] for k in ("lp_next", "lpx_next", "h_ref", "h_x")}


def pooled_stats(ms):
    dl, dh, n = _signed(ms)
    tl, th_, nt = _signed([_tail(m) for m in ms])
    return {"V4_abs_mean_dlogp": abs(dl), "V5_abs_mean_dH": abs(dh),
            "V4t_tail_abs_mean_dlogp": abs(tl), "V5t_tail_abs_mean_dH": abs(th_),
            "signed_mean_dlogp": dl, "signed_mean_dH": dh, "positions": n,
            "tail_signed_mean_dlogp": tl, "tail_signed_mean_dH": th_, "tail_positions": nt}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def thresholds(calib, out):
    calib = Path(calib)
    seqs = sorted({Path(p).stem.split("__")[1] for p in glob.glob(str(calib / "pairs" / "*.npz"))})
    per = {s: {x: per_seq_stats(np.load(calib / "pairs" / f"{x}__{s}.npz")) for x in TRUSTED} for s in seqs}
    pooled = {x: pooled_stats([np.load(calib / "pairs" / f"{x}__{s}.npz") for s in seqs]) for x in TRUSTED}
    th = {"K": K, "margin": MARGIN, "trusted": TRUSTED, "sequences": seqs, "per_sequence": {}, "pooled": {},
          "trusted_values": {"per_sequence": per, "pooled": pooled}}
    for s in seqs:
        th["per_sequence"][s] = {st: max(K * max(per[s][x][st] for x in TRUSTED), FLOOR[st]) for st in PER_SEQ}
    for st in POOLED:
        th["pooled"][st] = max(K * max(pooled[x][st] for x in TRUSTED), FLOOR[st])
    # leave-one-out (C9): each trusted variant judged against thresholds from the other four
    loo = {}
    for x in TRUSTED:
        others = [y for y in TRUSTED if y != x]
        worst = 0.0
        for s in seqs:
            for st in PER_SEQ:
                t = max(K * max(per[s][y][st] for y in others), FLOOR[st])
                worst = max(worst, per[s][x][st] / t)
        for st in POOLED:
            t = max(K * max(pooled[y][st] for y in others), FLOOR[st])
            worst = max(worst, pooled[x][st] / t)
        loo[x] = {"worst_ratio_to_threshold": worst, "passes": worst <= 1.0}
    th["leave_one_out"] = loo
    Path(out).write_text(json.dumps(th, indent=1), encoding="utf-8")
    print(json.dumps({"thresholds_per_sequence": th["per_sequence"], "thresholds_pooled": th["pooled"],
                      "leave_one_out": loo}, indent=1))


def judge(th, ms_by_seq, exact=None):
    """ms_by_seq: {sequence: measure dict}. exact: {rule: bool} from the run checks (None for controls)."""
    fails, worst = [], {}
    for s, m in ms_by_seq.items():
        st = per_seq_stats(m)
        for k, v in st.items():
            t = th["per_sequence"][s][k]
            worst[f"{k}@{s}"] = v / t
            if v > t:
                fails.append(f"{k}@{s}: {v:.4g} > {t:.4g} (x{v / t:.2f})")
    ps = pooled_stats(list(ms_by_seq.values()))
    for k in POOLED:
        t = th["pooled"][k]
        worst[k] = ps[k] / t
        if ps[k] > t:
            fails.append(f"{k}: {ps[k]:.4g} > {t:.4g} (x{ps[k] / t:.2f})")
    for rule, ok in (exact or {}).items():
        if not ok:
            fails.append(f"exact rule {rule} failed")
    top = sorted(worst.items(), key=lambda kv: -kv[1])[:5]
    return {"verdict": "PASS" if not fails else "FAIL", "fails": fails, "largest_ratios": top, "pooled": ps}


def judge_measures(th_path, mdir, prefix, out):
    th = json.loads(Path(th_path).read_text(encoding="utf-8"))
    ms = {s: np.load(Path(mdir) / f"{prefix}__{s}.npz") for s in th["sequences"]
          if (Path(mdir) / f"{prefix}__{s}.npz").exists()}
    if set(ms) != set(th["sequences"]):
        raise SystemExit(f"{prefix}: sequences missing: {sorted(set(th['sequences']) - set(ms))}")
    r = judge(th, ms)
    Path(out).write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(prefix, r["verdict"], "| largest ratios:", [(k, round(v, 2)) for k, v in r["largest_ratios"]][:3])


ALLOWED_ENV = {"STRATA_LOGITS_DIR", "STRATA_LOGITS_ALL_ROWS", "STRATA_GUARD_LOG"}
# A3: the only options a T8 run may carry (a whitelist: anything else, an approximation or not, fails), with the
# values the contract fixes. --tokens-file, --follow-file, --max-new and --model are checked by X1/X2 instead.
FIXED = {"--max-context": "16384", "--lookup": "0", "--window": "1", "--eos-ids": "0"}
FLAGS = {"--no-graphs", "--no-gpu-experts"}
VALUED = set(FIXED) | {"--model", "--tokens-file", "--follow-file", "--max-new", "--prompt-batch", "--stage-ahead"}
PROMPT_BATCH = {"whole": 1024, "chunk": 128, "split": 1024}
RUN_SETS = [f"{p}_{q}" for p in ("default", "graphs_off", "cpu") for q in ("whole", "chunk", "decode")] + \
    ["default_split"]


def parse_argv(argv):
    opts, bad = {}, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in FLAGS:
            if a in opts:
                bad.append(f"{a} twice")
            opts[a] = True
            i += 1
        elif a in VALUED and i + 1 < len(argv):
            if a in opts:
                bad.append(f"{a} twice")
            opts[a] = argv[i + 1]
            i += 2
        else:
            bad.append(a)
            i += 1
    return opts, bad


def signature_ok(ix, ids, path):
    """X1: exactly n index lines; line i is row i, position i, token ids[i]; and the path's shape. Rows 0..L-2 are
    the prompt read (src read, last false), in calls of exactly PB rows but the last; rows L-1..n-1 are single-token
    steps (src eval, n 1, last true), one call each. L = 1 for decode, n - TAIL otherwise."""
    n = len(ids)
    if len(ix) != n or any(r["row"] != i or r["pos"] != i or r["token"] != ids[i] for i, r in enumerate(ix)):
        return False
    L = 1 if path == "decode" else n - TAIL
    read, steps = ix[:L - 1], ix[L - 1:]
    if any(r["src"] != "eval" or r["n"] != 1 or not r["last"] for r in steps):
        return False
    if len({r["call"] for r in steps}) != len(steps):
        return False
    if path == "decode":
        return not read
    if any(r["src"] != "read" or r["last"] for r in read):
        return False
    calls = {}
    for r in read:
        calls.setdefault(r["call"], []).append(r)
    sizes = [len(g) for _, g in sorted(calls.items())]
    pb = PROMPT_BATCH[path]
    return (all(all(r["n"] == len(g) for r in g) for g in calls.values())
            and all(s == pb for s in sizes[:-1]) and 0 < sizes[-1] <= pb and sum(sizes) == L - 1)


def exact_rules(run_dir, run_set, ids, binary_sha256, model_sha256):
    """The exact rules of one candidate run (contract §5). model_sha256: {shard file name: SHA-256}, T7's.
    Returns ({rule: ok}, rows or None)."""
    import re
    placement, path = run_set.rsplit("_", 1)
    rj = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    argv, env, err = rj["argv"], rj["env_strata"], (run_dir / "stderr.txt").read_text(encoding="utf-8", errors="replace")
    ok = {"exit_0": rj["exit"] == 0}
    # S2 (attempts): an earlier attempt <seq>.attempt<k> is lawful only with the re-entry that traced it to the
    # environment, named in the judged run's attempt.json together with every earlier attempt
    prev = sorted(p.name for p in run_dir.parent.glob(run_dir.name + ".attempt*"))
    aj = run_dir / "attempt.json"
    att = json.loads(aj.read_text(encoding="utf-8")) if aj.exists() else None
    ok["S2_attempts_traced"] = (not prev and att is None) or \
        (att is not None and bool(str(att.get("reentry", "")).strip()) and sorted(att.get("previous", [])) == prev)
    n, V = len(ids), 131072
    ixp = run_dir / "dump" / "logits.jsonl"
    ix = [json.loads(l) for l in ixp.read_text(encoding="utf-8").splitlines() if l] if ixp.exists() else []
    ok["X1_coverage_and_signature"] = signature_ok(ix, ids, path)
    ok["X2_inputs"] = (rj["ids_sha256"] == hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()
                       and rj.get("n_ids") == n)
    ok["X3_binary"] = rj["binary_sha256"] == binary_sha256
    # X4 (the re-review's RB1): the model file. Every shard's SHA-256 as t8_run.py hashed it equals T7's, and
    # --model names the first shard
    first = min(model_sha256) if model_sha256 else None
    margv = argv[argv.index("--model") + 1] if "--model" in argv[:-1] else ""
    ok["X4_model"] = bool(model_sha256) and rj.get("model_sha256") == model_sha256 and \
        Path(margv).name == first
    fp = run_dir / "dump" / "logits.f32"
    fin = fp.exists() and fp.stat().st_size == n * V * 4
    if fin:
        dump = np.memmap(fp, dtype=np.float32, mode="r")
        for r0 in range(0, dump.size, 1 << 26):
            fin = fin and bool(np.isfinite(dump[r0:r0 + (1 << 26)]).all())
    ok["A1_finite_all_rows"] = fin
    m = re.search(r"(\d+) of (\d+) routed experts in VRAM", err)
    gpu_experts = int(m.group(1)) if m else -1
    opts, bad = parse_argv(argv)
    ok["A2_placement"] = ((gpu_experts == 0) if placement == "cpu" else (gpu_experts > 0)) and \
        opts.get("--no-gpu-experts", False) == (placement == "cpu") and \
        opts.get("--no-graphs", False) == (placement == "graphs_off")
    ok["A2_path"] = (opts.get("--prompt-batch") == (str(PROMPT_BATCH[path]) if path in PROMPT_BATCH else None)
                     and opts.get("--stage-ahead") == ("16384" if path == "split" else None))
    ok["A3_approximations_off"] = (not bad and all(opts.get(k) == v for k, v in FIXED.items())
                                   and set(env) <= ALLOWED_ENV)
    if placement == "graphs_off":
        g = run_dir / "guard.jsonl"
        lines = [json.loads(l) for l in g.read_text().splitlines() if l] if g.exists() else []
        ok["R1_guard_finite"] = bool(lines) and all(l["nonfinite"] == 0 for l in lines)
    good = ok["X1_coverage_and_signature"] and ok["A1_finite_all_rows"]
    return ok, (np.arange(n, dtype=np.int64) if good else None)


def judge_run_set(th_path, ref_root, run_set_dir, binary_sha256, model_json, out):
    """One candidate run set as written by t8_run.py: <run_set_dir>/<seq>/; the run set's name is the folder's.
    Before judging, each gpu reference file's SHA-256 is checked against the one T4 recorded in its run.json (the
    re-review's RS4); a mismatch stops the judge without a verdict, since the reference, not the candidate, is wrong.
    The thresholds file's SHA-256 is recorded in the output, to be compared with the contract's."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from t6_measure import measure
    run_set = Path(run_set_dir).name
    if run_set not in RUN_SETS:
        raise SystemExit(f"{run_set}: not a run set of the contract ({RUN_SETS})")
    th = json.loads(Path(th_path).read_text(encoding="utf-8"))
    model_sha256 = json.loads(Path(model_json).read_text(encoding="utf-8"))
    seqs = json.loads(Path(os.environ.get("PETITCHONK_WORK", ".")) / "fti" / "sequences" / "sequences.json".read_text(encoding="utf-8"))["sequences"]
    ref_ok = {}
    for s in th["sequences"]:
        rf = Path(ref_root) / "gpu" / s
        want = json.loads((rf / "run.json").read_text(encoding="utf-8"))["logits_sha256"]
        if sha256(rf / "logits.f32") != want:
            raise SystemExit(f"reference gpu/{s}: logits.f32 does not match T4's SHA-256 {want[:12]}; no verdict")
        ref_ok[s] = want
    ms, exact, per_run, pos0 = {}, {}, {}, {}
    for s in th["sequences"]:
        ids = seqs[s]["ids"]
        d = Path(run_set_dir) / s
        if not (d / "run.json").exists():
            per_run[s] = {"run_present": False}
            exact[f"run_present@{s}"] = False
            continue
        ok, idx = exact_rules(d, run_set, ids, binary_sha256, model_sha256)
        per_run[s] = ok
        for k, v in ok.items():
            exact[f"{k}@{s}"] = v
        if idx is not None:
            cand = np.memmap(d / "dump" / "logits.f32", dtype=np.float32, mode="r").reshape(-1, 131072)
            ms[s] = measure(Path(ref_root) / "gpu" / s / "logits.f32", None, ids,
                            transform=lambda z, r0, r1, a, c=cand, i=idx: np.asarray(c[i[r0:r1]], dtype=np.float64))
            pos0[s] = {"kl": float(ms[s]["kl"][0]), "top1_equal": bool(ms[s]["top1eq"][0])}
    # with a sequence missing (X1 or A1 failed, or no run), the verdict is FAIL on the exact rules; the statistics of
    # the sequences that could be measured are still reported, for the diagnostic (re-review N-c), and are partial
    r = judge(th, ms, exact) if ms else {"verdict": "FAIL", "fails": [], "largest_ratios": [], "pooled": {}}
    if set(ms) != set(th["sequences"]):
        r["verdict"] = "FAIL"
        r["fails"].append(f"statistics incomplete: no measure for {sorted(set(th['sequences']) - set(ms))}")
        r["statistics_partial"] = True
    r.update({"run_set": run_set, "exact": per_run, "binary_sha256": binary_sha256, "model_sha256": model_sha256,
              "thresholds_sha256": sha256(th_path), "reference_gpu_sha256": ref_ok,
              "position_0_observed": pos0})   # S5: reported, no verdict of its own (contract §5)
    Path(out).write_text(json.dumps(r, indent=1), encoding="utf-8")
    print(run_set, r["verdict"], r["fails"][:5])


if __name__ == "__main__":
    if sys.argv[1] == "thresholds":
        thresholds(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "judge-measures":
        judge_measures(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif sys.argv[1] == "judge-run-set":
        judge_run_set(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
