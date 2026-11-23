"""Exploration (not a release check): the quant x delta smoke test, as fixed in pub\\explore\\qd1\\RULE.md (22 Nov 2026).

Quants Q4KM (the reference), Q3KM, Q3KXL, Q2KXL; delta (--cache-bias) 0, 0.25, 0.5; the steering build. Quant by
quant, each read from disk once. Speed: the 8,192-token prompt b4["8192"][1], 257 written; lambda 0, 0.25, 0.5, 0.
Quality: B2 windows w00 w24 w48, prompt = the first 64 ids (a batch, never steered), --follow-file = the next 512
(token by token, steered); Q4KM lambda 0 twice (A, the reference, and A2, the noise floor). After each quality run
its logits are reduced to per-token NLL, KL(A || run) and top-1 agreement with A, and deleted; A's log-probabilities
are kept in ref\\ until the end. A quant that fails is recorded and skipped; a Q4KM failure stops the test.

usage: <venv-tools python> -I qd_smoke.py <out dir holding RULE.md> [--plan]
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
U = ROOT / r"models\unsloth-Mistral-Small-4-119B-2603-GGUF@bd93c721"
MODELS = {
    "Q4KM": U / r"UD-Q4_K_M\Mistral-Small-4-119B-2603-UD-Q4_K_M-00001-of-00003.gguf",
    "Q3KM": ROOT / (r"models\bartowski-mistralai_Mistral-Small-4-119B-2603-GGUF@f569d04e"
                    r"\mistralai_Mistral-Small-4-119B-2603-Q3_K_M\mistralai_Mistral-Small-4-119B-2603-Q3_K_M-00001-of-00002.gguf"),
    "Q3KXL": U / r"UD-Q3_K_XL\Mistral-Small-4-119B-2603-UD-Q3_K_XL-00001-of-00003.gguf",
    "Q2KXL": U / r"UD-Q2_K_XL\Mistral-Small-4-119B-2603-UD-Q2_K_XL.gguf",
}
REF = "Q4KM"
LAMS = (0.0, 0.25, 0.5)
SPEED_ORDER = ((0.0, 1), (0.25, 1), (0.5, 1), (0.0, 2))
B2 = ROOT / r"pub\p4\freeze_a\b2_windows.jsonl"
PROMPTS = ROOT / r"pub\p4\freeze_b\prompts.json"
WARM_PROFILE = ROOT / r"pub\p6\run1\b4port\len_8192\expert-profile.txt"
CUDA = os.path.join(os.environ.get("PETITCHONK_CUDA", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"), "bin", "x64")   # 13.0: never 13.2.x on RTX 50
WINDOWS = ("w00", "w24", "w48")
P, F = 64, 512
GEN = 257
VOCAB = 131072
SUMMARY = re.compile(r"prompt (\d+) tokens ([\d.]+) tok/s, generated (\d+) tokens ([\d.]+) tok/s \((\w+)\); "
                     r"experts from VRAM ([\d.]+)%")
LOAD = re.compile(r"loaded in ([\d.]+) s; (\d+) of 4608 routed experts in VRAM \(([\d.]+) GiB\)")


def log(out, msg):
    line = f"{time.strftime('%H:%M:%S', time.gmtime())}Z {msg}"
    print(line, flush=True)
    with open(out / "progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def quality_arms(q):
    return (("A", 0.0), ("A2", 0.0), ("l0.25", 0.25), ("l0.5", 0.5)) if q == REF else \
           (("l0", 0.0), ("l0.25", 0.25), ("l0.5", 0.5))


def engine(d: Path, model: Path, prompt, lam, follow=None, gen=None, logits=False):
    """One engine run. Returns its record; "failed" holds the reason when it did not run as asked."""
    d.mkdir(parents=True)
    (d / "prompt.txt").write_text(",".join(map(str, prompt)), encoding="utf-8")
    prof = d / "expert-profile.txt"
    shutil.copyfile(WARM_PROFILE, prof)
    argv = [str(EXE), "--model", str(model), "--tokens-file", str(d / "prompt.txt"),
            "--max-new", str(len(follow) if follow else gen), "--max-context", "16384", "--lookup", "0",
            "--window", "1", "--eos-ids", "0", "--expert-profile", str(prof)]
    if follow:
        (d / "follow.txt").write_text(",".join(map(str, follow)), encoding="utf-8")
        argv += ["--follow-file", str(d / "follow.txt")]
    if lam > 0:
        argv += ["--cache-bias", str(lam)]
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("STRATA_")}
    env["PATH"] = CUDA + ";" + env["PATH"]
    if logits:
        env["STRATA_LOGITS_DIR"] = str(d / "dump")
    t = time.time()
    p = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    (d / "stdout.txt").write_text(p.stdout, encoding="utf-8")
    (d / "stderr.txt").write_text(p.stderr, encoding="utf-8")
    r = {"exit": p.returncode, "seconds": round(time.time() - t, 1), "lam": lam,
         "bias_line": any("cache bias" in l for l in p.stderr.splitlines())}
    if (m := LOAD.search(p.stderr)):
        r.update(load_s=float(m.group(1)), experts_loaded=int(m.group(2)), experts_GiB=float(m.group(3)))
    if (m := SUMMARY.search(p.stderr)):
        r.update(prompt_tps=float(m.group(2)), generated=int(m.group(3)), tg_tps=float(m.group(4)),
                 vram_share=round(float(m.group(6)) / 100, 4))
    if p.returncode != 0:
        r["failed"] = f"exit {p.returncode}: " + " | ".join(p.stderr.strip().splitlines()[-3:])[-400:]
    elif "tg_tps" not in r:
        r["failed"] = "no summary line"
    elif (lam > 0) != r["bias_line"]:
        r["failed"] = "the cache-bias line does not match the lambda asked"
    (d / "result.json").write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    return r


def rows(d: Path, follow):
    """The 511 scored rows' log-probabilities (float32), checked: positions 64..574 fed follow[0..510]."""
    ix = [json.loads(l) for l in (d / "dump" / "logits.jsonl").read_text(encoding="utf-8").splitlines()]
    want = {P + i: i for i in range(F - 1)}
    raw = np.memmap(d / "dump" / "logits.f32", dtype="<f4", mode="r").reshape(-1, VOCAB)
    out = np.empty((F - 1, VOCAB), np.float32)
    seen = set()
    for x in ix:
        q = x["pos"]
        if q in want and x["last"]:
            i = want[q]
            if x["token"] != follow[i]:
                raise RuntimeError(f"{d}: row pos {q} fed {x['token']}, not follow[{i}] = {follow[i]}")
            v = raw[x["row"]].astype(np.float64)
            v -= v.max()
            out[i] = v - np.log(np.exp(v).sum())
            seen.add(q)
    if len(seen) != F - 1:
        raise RuntimeError(f"{d}: {len(seen)} scored rows, not {F - 1}")
    del raw
    return out


def compare(a, x, target):
    a, x = a.astype(np.float64), x.astype(np.float64)
    return {"nll": (-x[np.arange(F - 1), target]).tolist(),
            "kl": (np.exp(a) * (a - x)).sum(1).tolist(),
            "top1_agree": (x.argmax(1) == a.argmax(1)).tolist()}


def plan(out: Path):
    problems = [str(p) for p in [EXE, B2, PROMPTS, WARM_PROFILE, *MODELS.values()] if not p.exists()]
    if not (out / "RULE.md").exists():
        problems.append("RULE.md missing")
    extra = [p.name for p in out.iterdir() if p.name not in ("RULE.md", "plan.txt")] if out.exists() else []
    if extra:
        problems.append(f"the out dir holds more than RULE.md: {extra}")
    b2 = {r["id"] for r in map(json.loads, B2.read_text(encoding="utf-8").splitlines())}
    problems += [f"no B2 window {w}" for w in WINDOWS if w not in b2]
    if len(json.loads(PROMPTS.read_text(encoding="utf-8"))["b4"]["8192"][1]) != 8192:
        problems.append("the 8,192-token prompt is not 8,192 ids")
    runs = []
    for q in MODELS:
        runs += [f"{q} speed lam {lam} rep {rep}" for lam, rep in SPEED_ORDER]
        runs += [f"{q} quality {w} {arm} lam {lam}" for w in WINDOWS for arm, lam in quality_arms(q)]
    return problems, runs


def main(out: Path):
    problems, runs = plan(out)
    if problems:
        sys.exit("not started: " + "; ".join(problems))
    b2 = {r["id"]: r["ids"] for r in map(json.loads, B2.read_text(encoding="utf-8").splitlines())}
    ids8k = json.loads(PROMPTS.read_text(encoding="utf-8"))["b4"]["8192"][1]
    log(out, f"start; {len(runs)} runs; engine {EXE}")
    (out / "ref").mkdir()
    speed, quality, failed = [], {}, {}
    for q, model in MODELS.items():
        ok = True
        for lam, rep in SPEED_ORDER:
            r = engine(out / "speed" / q / f"lam{lam}_{rep}", model, ids8k, lam, gen=GEN)
            log(out, f"{q} speed lam {lam} rep {rep}: {r['seconds']} s, writing {r.get('tg_tps')} tok/s, "
                     f"VRAM {r.get('vram_share')}, experts {r.get('experts_loaded')}" +
                     (f"  FAILED {r['failed']}" if "failed" in r else ""))
            if "failed" in r:
                failed[q] = f"speed lam {lam} rep {rep}: {r['failed']}"
                ok = False
                break
            speed.append({"quant": q, "lam": lam, "rep": rep, **{k: r.get(k) for k in
                          ("tg_tps", "prompt_tps", "vram_share", "experts_loaded", "experts_GiB", "load_s")}})
        if not ok and q == REF:
            break
        for w in WINDOWS if ok else ():
            prompt, follow = b2[w][:P], b2[w][P:P + F]
            target = np.array(follow[1:])
            for arm, lam in quality_arms(q):
                d = out / "quality" / q / w / arm
                r = engine(d, model, prompt, lam, follow=follow, logits=True)
                if "failed" in r:
                    log(out, f"{q} {w} {arm} lam {lam}: FAILED {r['failed']}")
                    failed[q] = f"quality {w} {arm}: {r['failed']}"
                    ok = False
                    break
                try:
                    lp = rows(d, follow)
                except RuntimeError as e:
                    log(out, f"{q} {w} {arm}: FAILED {e}")
                    failed[q] = f"quality {w} {arm}: {e}"
                    ok = False
                    break
                if q == REF and arm == "A":
                    np.save(out / "ref" / f"{w}.npy", lp)
                res = compare(np.load(out / "ref" / f"{w}.npy"), lp, target)
                res["tg_tps"] = r.get("tg_tps")
                (d / "per_token.json").write_text(json.dumps(res) + "\n", encoding="utf-8")
                shutil.rmtree(d / "dump")
                quality.setdefault(q, {}).setdefault(arm, {})[w] = res
                log(out, f"{q} {w} {arm} lam {lam}: {r['seconds']} s, writing {r.get('tg_tps')} tok/s, "
                         f"VRAM {r.get('vram_share')}; reduced, logits deleted")
            if not ok:
                break
        if not ok and q == REF:
            break
    (out / "speed.json").write_text(json.dumps(speed, indent=1) + "\n", encoding="utf-8")
    analyse(out, speed, quality, failed)
    shutil.rmtree(out / "ref")
    log(out, "the reference log-probabilities in ref\\ deleted")


def analyse(out: Path, speed, quality, failed):
    rng = np.random.default_rng(20261122)
    ref_runs = [s["tg_tps"] for s in speed if s["quant"] == REF and s["lam"] == 0.0]
    base = float(np.mean(ref_runs)) if ref_runs else None
    cells = []
    floor_upper = None
    for q in MODELS:
        for lam in LAMS:
            runs = [s for s in speed if s["quant"] == q and s["lam"] == lam]
            c = {"quant": q, "lam": lam}
            if runs:
                tg = [s["tg_tps"] for s in runs]
                c.update(writing=tg, writing_mean=round(float(np.mean(tg)), 2),
                         speed_ratio=round(float(np.mean(tg)) / base, 3) if base else None,
                         vram_share=[s["vram_share"] for s in runs], experts_loaded=runs[0]["experts_loaded"],
                         prompt_tps=[s["prompt_tps"] for s in runs])
                c["fast"] = c["speed_ratio"] is not None and c["speed_ratio"] >= 1.20
            arm = ("A2" if lam == 0.0 else f"l{lam}") if q == REF else f"l{'0' if lam == 0.0 else lam}"
            per = quality.get(q, {}).get(arm, {})
            if per and all(w in per for w in WINDOWS) and REF in quality and "A" in quality[REF]:
                bd, bn, kl, agree = [], [], [], []
                for w in WINDOWS:
                    d = np.array(per[w]["nll"]) - np.array(quality[REF]["A"][w]["nll"])
                    for s in range(0, F - 1, 64):
                        bd.append(d[s:s + 64].sum())
                        bn.append(len(d[s:s + 64]))
                    kl += per[w]["kl"]
                    agree += per[w]["top1_agree"]
                bd, bn = np.array(bd), np.array(bn)
                pick = rng.integers(0, len(bd), size=(10000, len(bd)))
                boot = np.exp(bd[pick].sum(1) / bn[pick].sum(1))
                lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
                c.update(ppl_ratio=round(float(np.exp(bd.sum() / bn.sum())), 5), ci95=[round(lo, 5), round(hi, 5)],
                         kl_mean=round(float(np.mean(kl)), 5), top1_agree=round(float(np.mean(agree)), 4),
                         quality_tokens=int(bn.sum()),
                         decode_tps=[per[w]["tg_tps"] for w in WINDOWS])
                if q == REF and lam == 0.0:
                    floor_upper = hi
            if q in failed:
                c["failed"] = failed[q]
            cells.append(c)
    for c in cells:
        if "ci95" not in c:
            c["quality_label"] = "not measured"
        elif c["quant"] == REF and c["lam"] == 0.0:
            c["quality_label"] = "noise floor (A2 against A)"
        elif floor_upper is None or floor_upper > 1.02:
            c["quality_label"] = "unclear"
        else:
            c["quality_label"] = ("no loss seen" if c["ci95"][1] <= 1.02 else
                                  "loss seen" if c["ci95"][0] > 1.02 else "unclear")
        c["candidate"] = bool(c.get("fast")) and c["quality_label"] in ("no loss seen", "unclear") \
            and not (c["quant"] == REF and c["lam"] == 0.0)
    res = {"base_writing_tps": round(base, 2) if base else None, "noise_floor_upper": floor_upper,
           "failed": failed, "cells": cells}
    (out / "result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    lines = ["| Quant | lambda | Writing tok/s | Speed ratio | VRAM share | Experts in VRAM | PPL ratio (95%) | KL | "
             "Top-1 agree | Quality label | Candidate |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            c["quant"], c["lam"], ", ".join(map(str, c.get("writing", []))) or "-", c.get("speed_ratio", "-"),
            ", ".join(map(str, c.get("vram_share", []))) or "-", c.get("experts_loaded", "-"),
            f"{c['ppl_ratio']} ({c['ci95'][0]}-{c['ci95'][1]})" if "ci95" in c else "-", c.get("kl_mean", "-"),
            c.get("top1_agree", "-"), c["quality_label"] + (f"; FAILED {c['failed']}" if "failed" in c else ""),
            "yes" if c["candidate"] else "no"))
    (out / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(out, "RESULT\n" + "\n".join(lines))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[2] == "--plan":
        pr, rn = plan(Path(sys.argv[1]))
        print("\n".join(rn))
        print(f"{len(rn)} runs; problems: {pr or 'none'}")
        sys.exit(1 if pr else 0)
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
