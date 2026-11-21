"""PCHONK-FTI-001 T6: the per-position measures between a reference logits file and another one, for the same
token sequence. One implementation, used by the calibration (trusted pairs), the negative controls, and T8's
evaluation of the candidate, so that every comparison uses the same arithmetic.

For each position i (row i holds the scores for the token after position i):
  kl[i]      KL(p_ref || p_x) in nats, over the full vocabulary, computed in float64 from the f32 logits
  dlp[i]     |log p_ref(next) - log p_x(next)|, next = ids[i+1]; NaN at the last position (no next token)
  top1eq[i]  argmax(ref) == argmax(x)
  margin[i]  ref's top-1 minus top-2 logit
  lp_next[i] log p_ref(next) (for the per-sequence mean, i.e. the perplexity); NaN at the last position
  lpx_next[i] log p_x(next): with lp_next, the signed difference (a perplexity ratio); NaN at the last position
  h_ref[i], h_x[i]  the entropies (nats) of ref's and x's rows: their signed mean difference shows a systematic
             sharpening or flattening (a temperature-like defect), which the unsigned measures miss
  xfinite[i] every logit of x's row is finite
Rows are read in chunks with np.memmap; nothing is held whole.

usage (library): measure(ref_path, x_path, ids, n_vocab=131072, chunk=256) -> dict of numpy arrays
"""
import numpy as np

V = 131072


def _log_softmax(z):
    m = z.max(axis=1, keepdims=True)
    s = np.log(np.exp(z - m).sum(axis=1, keepdims=True))
    return z - m - s


def measure(ref_path, x_path, ids, n_vocab=V, chunk=256, transform=None):
    """transform (negative controls only): x_path is ignored and the compared rows are
    transform(ref_rows_float64, r0, r1, ref_memmap) -> rows, a perturbed copy of the reference."""
    n = len(ids)
    a = np.memmap(ref_path, dtype=np.float32, mode="r")
    b = a if transform is not None else np.memmap(x_path, dtype=np.float32, mode="r")
    if a.size != n * n_vocab or b.size != n * n_vocab:
        raise ValueError(f"row counts: ref {a.size / n_vocab}, x {b.size / n_vocab}, ids {n}")
    a = a.reshape(n, n_vocab)
    b = b.reshape(n, n_vocab)
    nxt = np.asarray(list(ids[1:]) + [-1], dtype=np.int64)
    out = {k: np.empty(n, dtype=np.float64) for k in ("kl", "dlp", "margin", "lp_next", "lpx_next", "h_ref", "h_x")}
    out["top1eq"] = np.empty(n, dtype=bool)
    out["xfinite"] = np.empty(n, dtype=bool)
    for r0 in range(0, n, chunk):
        r1 = min(n, r0 + chunk)
        za = np.asarray(a[r0:r1], dtype=np.float64)
        zb = transform(za.copy(), r0, r1, a) if transform is not None else np.asarray(b[r0:r1], dtype=np.float64)
        fin = np.isfinite(zb).all(axis=1)
        out["xfinite"][r0:r1] = fin
        la = _log_softmax(za)
        lb = _log_softmax(np.where(np.isfinite(zb), zb, -1e30))
        pa = np.exp(la)
        out["kl"][r0:r1] = (pa * (la - lb)).sum(axis=1)
        out["h_ref"][r0:r1] = -(pa * la).sum(axis=1)
        out["h_x"][r0:r1] = -(np.exp(lb) * lb).sum(axis=1)
        top2 = np.partition(za, -2, axis=1)[:, -2:]
        out["margin"][r0:r1] = top2[:, 1] - top2[:, 0]
        out["top1eq"][r0:r1] = za.argmax(axis=1) == zb.argmax(axis=1)
        idx = np.arange(r1 - r0)
        nx = nxt[r0:r1]
        has = nx >= 0
        lpa = np.full(r1 - r0, np.nan)
        lpb = np.full(r1 - r0, np.nan)
        lpa[has] = la[idx[has], nx[has]]
        lpb[has] = lb[idx[has], nx[has]]
        out["lp_next"][r0:r1] = lpa
        out["lpx_next"][r0:r1] = lpb
        out["dlp"][r0:r1] = np.abs(lpa - lpb)
    del a, b
    return out
