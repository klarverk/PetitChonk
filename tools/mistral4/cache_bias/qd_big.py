"""Exploration: the delta on UD-Q3_K_XL, the larger test, as fixed in pub\\explore\\qd2\\RULE.md (22 Nov 2026).

Expects llama.cpp's reference in <out>\\ref_q3kxl (run_qd_big.bat runs P5's p4_ref.py first). Then, per window of
pub\\explore\\qd2_windows.jsonl, our engine (the steering build) on UD-Q3_K_XL at lambda 0 and lambda 0.5: prompt =
the first 64 ids, --follow-file = the next 1,024; logits reduced to per-token NLL and deleted. Then the measures.

usage: <venv-tools python> -I qd_big.py <out dir holding RULE.md and ref_q3kxl> [--plan]
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PETITCHONK_WORK", "."))   # the work folder (tools/mistral4/README.md)
EXE = Path(os.environ.get("PETITCHONK_BINARY") or ROOT / r"steer\build-steer\strata-poly.exe")   # an engine that takes --cache-bias for mistral4
MODEL = ROOT / (r"models\unsloth-Mistral-Small-4-119B-2603-GGUF@bd93c721\UD-Q3_K_XL"
                r"\Mistral-Small-4-119B-2603-UD-Q3_K_XL-00001-of-00003.gguf")
WINDOWS_FILE = ROOT / r"pub\explore\qd2_windows.jsonl"
WARM_PROFILE = ROOT / r"pub\p6\run1\b4port\len_8192\expert-profile.txt"
CUDA = os.path.join(os.environ.get("PETITCHONK_CUDA", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"), "bin", "x64")   # 13.0: never 13.2.x on RTX 50
P, F = 64, 1024
LAMS = (0.0, 0.5)
VOCAB = 131072
SUMMARY = re.compile(r"prompt (\d+) tokens ([\d.]+) tok/s, generated (\d+) tokens ([\d.]+) tok/s \((\w+)\); "
                     r"experts from VRAM ([\d.]+)%")


def log(out, msg):
    line = f"{time.strftime('%H:%M:%S', time.gmtime())}Z {msg}"
    print(line, flush=True)
    with open(out / "progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def engine(d: Path, prompt, follow, lam):
    d.mkdir(parents=True)
    (d / "prompt.txt").write_text(",".join(map(str, prompt)), encoding="utf-8")
    (d / "follow.txt").write_text(",".join(map(str, follow)), encoding="utf-8")
    prof = d / "expert-profile.txt"
    shutil.copyfile(WARM_PROFILE, prof)
    argv = [str(EXE), "--model", str(MODEL), "--tokens-file", str(d / "prompt.txt"), "--max-new", str(len(follow)),
            "--max-context", "16384", "--lookup", "0", "--window", "1", "--eos-ids", "0",
            "--expert-profile", str(prof), "--follow-file", str(d / "follow.txt")]
    if lam > 0:
        argv += ["--cache-bias", str(lam)]
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("STRATA_")}
    env["PATH"] = CUDA + ";" + env["PATH"]
    env["STRATA_LOGITS_DIR"] = str(d / "dump")
    t = time.time()
    p = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    (d / "stdout.txt").write_text(p.stdout, encoding="utf-8")
    (d / "stderr.txt").write_text(p.stderr, encoding="utf-8")
    r = {"exit": p.returncode, "seconds": round(time.time() - t, 1), "lam": lam,
         "bias_line": any("cache bias" in l for l in p.stderr.splitlines())}
    if (m := SUMMARY.search(p.stderr)):
        r.update(tg_tps=float(m.group(4)), vram_share=round(float(m.group(6)) / 100, 4))
    (d / "result.json").write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    if p.returncode != 0 or "tg_tps" not in r or (lam > 0) != r["bias_line"]:
        raise SystemExit(f"run {d} failed: {r}")
    return r


def nll(d: Path, follow):
    """The 1,023 scored tokens' NLL: rows at positions 64..1086 fed follow[0..1022], scored on follow[1..1023]."""
    ix = [json.loads(l) for l in (d / "dump" / "logits.jsonl").read_text(encoding="utf-8").splitlines()]
    want = {P + i: i for i in range(F - 1)}
    raw = np.memmap(d / "dump" / "logits.f32", dtype="<f4", mode="r").reshape(-1, VOCAB)
    out = np.full(F - 1, np.nan)
    for x in ix:
        q = x["pos"]
        if q in want and x["last"]:
            i = want[q]
            if x["token"] != follow[i]:
                raise SystemExit(f"{d}: row pos {q} fed {x['token']}, not follow[{i}] = {follow[i]}")
            v = raw[x["row"]].astype(np.float64)
            m = v.max()
            out[i] = -(v[follow[i + 1]] - m - np.log(np.exp(v - m).sum()))
    del raw
    if np.isnan(out).any():
        raise SystemExit(f"{d}: {int(np.isnan(out).sum())} scored rows missing")
    return out


def ref(out: Path, ids):
    got = {}
    for line in (out / "ref_q3kxl" / "logprobs.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["from"] != 1 or r["n"] != 4096 or r.get("nonfinite"):
            raise SystemExit(f"reference {r['id']}: from {r['from']}, n {r['n']}, nonfinite {r.get('nonfinite')}")
        lp = r["logp"][P:P + F - 1]
        if any(v is None for v in lp):
            raise SystemExit(f"reference {r['id']}: a null logp")
        got[r["id"]] = -np.array(lp, dtype=np.float64)
    if set(got) != set(ids):
        raise SystemExit(f"reference windows {sorted(got)} differ from {sorted(ids)}")
    return got


def ratio(x, y, ids, rng):
    bd, bn = [], []
    for w in ids:
        d = x[w] - y[w]
        for s in range(0, len(d), 64):
            bd.append(d[s:s + 64].sum())
            bn.append(len(d[s:s + 64]))
    bd, bn = np.array(bd), np.array(bn)
    pick = rng.integers(0, len(bd), size=(10000, len(bd)))
    boot = np.exp(bd[pick].sum(1) / bn[pick].sum(1))
    return {"ratio": round(float(np.exp(bd.sum() / bn.sum())), 4), "ci95": [round(float(np.quantile(boot, 0.025)), 4),
            round(float(np.quantile(boot, 0.975)), 4)], "blocks": len(bd), "tokens": int(bn.sum())}


def main(out: Path, plan_only=False):
    wins = [json.loads(l) for l in WINDOWS_FILE.read_text(encoding="utf-8").splitlines()]
    ids = [w["id"] for w in wins]
    problems = [str(p) for p in (EXE, MODEL, WARM_PROFILE, out / "RULE.md") if not p.exists()]
    if plan_only:
        print(f"{len(ids)} windows: {' '.join(ids)}; {2 * len(ids)} engine runs; problems: {problems or 'none'}")
        sys.exit(1 if problems else 0)
    if problems:
        sys.exit("not started: " + "; ".join(problems))
    refs = ref(out, ids)
    log(out, f"start; {len(ids)} windows, lambda {LAMS}; engine {EXE}")
    ours = {lam: {} for lam in LAMS}
    speed = []
    for w in wins:
        prompt, follow = w["ids"][:P], w["ids"][P:P + F]
        for lam in LAMS:
            d = out / "runs" / w["id"] / f"lam{lam}"
            r = engine(d, prompt, follow, lam)
            ours[lam][w["id"]] = nll(d, follow)
            (d / "nll.json").write_text(json.dumps(ours[lam][w["id"]].tolist()) + "\n", encoding="utf-8")
            shutil.rmtree(d / "dump")
            speed.append({"window": w["id"], "lam": lam, "tg_tps": r["tg_tps"], "vram_share": r["vram_share"]})
            log(out, f"{w['id']} lam {lam}: {r['seconds']} s, writing {r['tg_tps']} tok/s, VRAM {r['vram_share']}; "
                     f"reduced, logits deleted")
    rng = np.random.default_rng(20261122)
    a = np.concatenate([ours[0.0][w] for w in ids])
    g = np.concatenate([refs[w] for w in ids])
    res = {"alignment_corr_lam0_vs_llamacpp": round(float(np.corrcoef(a, g)[0, 1]), 4),
           "primary_lam05_vs_llamacpp": ratio(ours[0.5], refs, ids, rng),
           "lam0_vs_llamacpp": ratio(ours[0.0], refs, ids, rng),
           "lam05_vs_lam0": ratio(ours[0.5], ours[0.0], ids, rng),
           "llamacpp_ppl": round(float(np.exp(g.mean())), 3)}
    lo, hi = res["primary_lam05_vs_llamacpp"]["ci95"]
    res["verdict"] = ("NO ADDED LOSS ABOVE 1%" if hi <= 1.01 else "ADDED LOSS ABOVE 1%" if lo > 1.01 else
                      "UNRESOLVED AT THIS SIZE")
    for lam in LAMS:
        v = [s["tg_tps"] for s in speed if s["lam"] == lam]
        res[f"writing_tps_lam{lam}"] = {"mean": round(float(np.mean(v)), 2), "min": min(v), "max": max(v)}
    (out / "speed.json").write_text(json.dumps(speed, indent=1) + "\n", encoding="utf-8")
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    log(out, "RESULT " + json.dumps(res))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2] == "--plan":
        main(Path(sys.argv[1]), plan_only=True)
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
