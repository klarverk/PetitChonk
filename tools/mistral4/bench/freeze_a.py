"""PCHONK-PUB-001 P4, freeze part A: B2's windows and B3's items, rendered and tokenized (P4_benchmark_design.md §2, §5).

Runs no model. Deterministic: a second run writes byte-identical files.

usage: <venv-tools python> -I freeze_a.py <pub data dir> <lm-eval files dir> <tekken.json> <out dir>
  pub data dir      <work>\\pub\\data (its SHA256SUMS is checked first)
  lm-eval files dir fetch_mmlu_yaml.py's output (its SHA256SUMS is checked first)
  out dir           must not exist

Writes into <out dir>:
  b2_windows.jsonl        73 windows {"id","ids","from"}: BOS + 4,095 consecutive text tokens; scored from 1
  b3_items.jsonl          900 items: task, row, id, subject, context, choices, gold
  b3_sequences.jsonl      one per item and choice {"id","ids","from"}: BOS + context + " " + choice; scored from
                          the first continuation token
  qual_b2.jsonl           the text's tail after the 73 windows (BOS + 2,541 tokens): qualification only
  qual_b3_items.jsonl     3 undrawn items per task: qualification only
  qual_b3_sequences.jsonl their sequences
  qual_all.jsonl          qual_b2's window and qual_b3's sequences, the qualification runs' input
  summary.json            counts, lengths, checks and input hashes
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import mistral_common
import yaml
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

SEED = 20261119            # P4 §2, fixed 19 Nov
QUAL_SEED = 20261120       # the qualification items: drawn from the rows SEED did not draw
PER_TASK = 300
QUAL_PER_TASK = 3
WINDOW = 4096              # BOS + 4,095 text tokens
N_WINDOWS = 73             # the user, 20 Nov: "All 73 windows (Recommended)"
B2_TOTAL_RECORDED = 301477 # P4 §2: the rows joined as stored, with BOS
FILES = {"wikitext": "Salesforce__wikitext@b08601e0/test-00000-of-00001.jsonl",
         "arc_challenge": "allenai__ai2_arc@210d026f/test-00000-of-00001.jsonl",
         "commonsense_qa": "tau__commonsense_qa@94630fe3/validation-00000-of-00001.jsonl",
         "mmlu": "cais__mmlu@c30699e8/test-00000-of-00001.jsonl"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def check_sums(d):
    d = Path(d)
    bad = []
    for line in (d / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        h, name = line.split(maxsplit=1)
        if sha((d / name.lstrip("*")).read_bytes()) != h:
            bad.append(name)
    if bad:
        raise SystemExit(f"SHA256SUMS mismatch in {d}: {bad}")
    return sha((d / "SHA256SUMS").read_bytes())


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def dump(path, objs):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=True, separators=(",", ":")) + "\n")


# ---- B3 formats: lm-evaluation-harness v0.4.13 (ddd67220), as P4 §2 records them; check_render.py renders the same
# ---- contexts from the task files with Jinja and compares
def arc(r):
    ctx = f"Question: {r['question']}\nAnswer:"
    return ctx, list(r["choices"]["text"]), r["choices"]["label"].index(r["answerKey"])


def csqa(r):
    t = r["choices"]["text"]
    assert r["choices"]["label"] == ["A", "B", "C", "D", "E"], r["id"]
    ctx = (f"Question: {r['question'].strip()}\nA. {t[0]}\nB. {t[1]}\nC. {t[2]}\nD. {t[3]}\nE. {t[4]}\nAnswer:")
    return ctx, ["A", "B", "C", "D", "E"], ["A", "B", "C", "D", "E"].index(r["answerKey"].strip())


def mmlu_fn(descriptions):
    def f(r):
        c = r["choices"]
        ctx = (descriptions[r["subject"]] +
               f"{r['question'].strip()}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer:")
        return ctx, ["A", "B", "C", "D"], int(r["answer"])
    return f


def encode_pair(tok, ctx, cont):
    """lm-eval's _encode_pair split (trailing context whitespace moves to the continuation; the continuation ids are
    whole[len(context ids):]), with BOS first as FTI and mistral-common's encode(bos=True) put it."""
    n = len(ctx) - len(ctx.rstrip())
    if n:
        cont, ctx = ctx[-n:] + cont, ctx[:-n]
    whole = tok.encode(ctx + cont, bos=True, eos=False)
    c = tok.encode(ctx, bos=True, eos=False)
    return whole, len(c), whole[:len(c)] == c


def b3(tok, task, data, render, picks):
    items, seqs, prefix_bad = [], [], 0
    for i in picks:
        r = data[i]
        ctx, choices, gold = render(r)
        items.append({"task": task, "row": i, "id": r.get("id"), "subject": r.get("subject"), "context": ctx,
                      "choices": choices, "gold": gold})
        for k, ch in enumerate(choices):
            whole, frm, ok = encode_pair(tok, ctx, " " + ch)    # target_delimiter " "
            prefix_bad += not ok
            assert frm < len(whole), (task, i, k)
            seqs.append({"id": f"{task}:{i}:{k}", "ids": whole, "from": frm})
    return items, seqs, prefix_bad


def main(data_dir, lmeval_dir, tekken, out):
    data_dir, lmeval_dir, out = Path(data_dir), Path(lmeval_dir), Path(out)
    data_sums = check_sums(data_dir)
    lmeval_sums = check_sums(lmeval_dir)
    out.mkdir(parents=True, exist_ok=False)
    tok = MistralTokenizer.from_file(str(tekken)).instruct_tokenizer.tokenizer
    bos = tok.bos_id

    # ---- B2: the rows joined as stored; BOS + 4,095 consecutive text tokens per window, every text token once
    text = "".join(r["text"] for r in rows(data_dir / FILES["wikitext"]))
    ids = tok.encode(text, bos=True, eos=False)
    if len(ids) != B2_TOTAL_RECORDED or ids[0] != bos:
        raise SystemExit(f"B2: {len(ids)} tokens, not the recorded {B2_TOTAL_RECORDED}")
    body, step = ids[1:], WINDOW - 1
    if len(body) // step != N_WINDOWS:
        raise SystemExit(f"B2: {len(body) // step} full windows, not {N_WINDOWS}")
    windows = [{"id": f"w{k:02d}", "ids": [bos] + body[k * step:(k + 1) * step], "from": 1} for k in range(N_WINDOWS)]
    tail = body[N_WINDOWS * step:]
    dump(out / "b2_windows.jsonl", windows)
    dump(out / "qual_b2.jsonl", [{"id": "tail", "ids": [bos] + tail, "from": 1}])

    # ---- B3: 300 items per task by SEED; 3 qualification items per task from the rest by QUAL_SEED
    desc = {}
    for p in sorted(lmeval_dir.glob("mmlu_*.yaml")):
        y = yaml.safe_load(p.read_text(encoding="utf-8"))
        desc[y["dataset_name"]] = y["description"]
    tasks = [("arc_challenge", arc), ("commonsense_qa", csqa), ("mmlu", mmlu_fn(desc))]
    all_items, all_seqs, q_items, q_seqs, per_task, prefix_bad = [], [], [], [], {}, 0
    for task, render in tasks:
        data = rows(data_dir / FILES[task])
        picks = sorted(random.Random(SEED).sample(range(len(data)), PER_TASK))
        rest = sorted(set(range(len(data))) - set(picks))
        qpicks = sorted(random.Random(QUAL_SEED).sample(rest, QUAL_PER_TASK))
        it, sq, bad = b3(tok, task, data, render, picks)
        qit, qsq, qbad = b3(tok, task, data, render, qpicks)
        all_items += it; all_seqs += sq; q_items += qit; q_seqs += qsq; prefix_bad += bad + qbad
        per_task[task] = {"rows": len(data), "items": len(it), "sequences": len(sq),
                          "tokens": sum(len(s["ids"]) for s in sq), "max_len": max(len(s["ids"]) for s in sq),
                          "scored_tokens": sum(len(s["ids"]) - s["from"] for s in sq),
                          "qual_rows": qpicks}
    dump(out / "b3_items.jsonl", all_items)
    dump(out / "b3_sequences.jsonl", all_seqs)
    dump(out / "qual_b3_items.jsonl", q_items)
    dump(out / "qual_b3_sequences.jsonl", q_seqs)
    dump(out / "qual_all.jsonl", [{"id": "tail", "ids": [bos] + tail, "from": 1}] + q_seqs)

    summary = {
        "inputs": {"pub_data_SHA256SUMS": data_sums, "lmeval_SHA256SUMS": lmeval_sums,
                   "tekken_sha256": sha(Path(tekken).read_bytes()), "mistral_common": mistral_common.__version__,
                   "python": sys.version.split()[0]},
        "b2": {"tokens_with_bos": len(ids), "windows": N_WINDOWS, "window_len": WINDOW,
               "scored_tokens": N_WINDOWS * step, "tail_tokens": len(tail)},
        "b3": per_task, "b3_sequences": len(all_seqs),
        "b3_tokens": sum(len(s["ids"]) for s in all_seqs),
        "mmlu_subjects_drawn": sorted({i["subject"] for i in all_items if i["task"] == "mmlu"}),
        "prefix_property_violations": prefix_bad,
        "seeds": {"items": SEED, "qualification": QUAL_SEED},
        "files": {p.name: sha(p.read_bytes()) for p in sorted(out.glob("*.jsonl"))},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    print(json.dumps({k: summary[k] for k in ("b2", "b3_sequences", "b3_tokens", "prefix_property_violations")},
                     indent=1))
    print(json.dumps({t: {k: v for k, v in d.items() if k != "qual_rows"} for t, d in per_task.items()}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
