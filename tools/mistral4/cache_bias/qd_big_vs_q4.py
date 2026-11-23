r"""Exploration, context for pub\explore\qd2 (not part of its rule): the same 16 windows and 16,368 tokens against
llama.cpp UD-Q4_K_M (P5's b2_ref_gpu, which covers all 73 B2 windows). Reads only. Ratios and 95% intervals as in
qd_big.py (256 blocks of 64, 10,000 resamples, seed 20261122).
usage: <venv-tools python> -I qd_big_vs_q4.py"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import qd_big as b
out = b.ROOT / r"pub\explore\qd2"
ids = [json.loads(l)["id"] for l in b.WINDOWS_FILE.read_text(encoding="utf-8").splitlines()]
q4 = {}
for line in (b.ROOT / r"pub\p5\run1\b2_ref_gpu\logprobs.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    if r["id"] in ids:
        assert r["from"] == 1 and r["n"] == 4096
        q4[r["id"]] = -np.array(r["logp"][b.P:b.P + b.F - 1], dtype=np.float64)
q3 = b.ref(out, ids)
ours = {lam: {w: np.array(json.loads((out / "runs" / w / f"lam{lam}" / "nll.json").read_text(encoding="utf-8")))
              for w in ids} for lam in b.LAMS}
rng = np.random.default_rng(20261122)
res = {"llamacpp_q4_ppl": round(float(np.exp(np.concatenate([q4[w] for w in ids]).mean())), 3),
       "quant_loss_llamacpp_q3kxl_vs_q4": b.ratio(q3, q4, ids, rng),
       "ours_lam0_vs_llamacpp_q4": b.ratio(ours[0.0], q4, ids, rng),
       "ours_lam05_vs_llamacpp_q4": b.ratio(ours[0.5], q4, ids, rng)}
(out / "vs_q4_context.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps(res, indent=1))
