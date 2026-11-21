"""PCHONK-PUB-001 P4: render B3's frozen contexts again from lm-eval's own task files, and compare.

freeze_a.py writes each context with Python f-strings, from P4 §2's record of the formats. This renders the same
items a second way: from the task files fetched at v0.4.13 (`ddd67220`, fetch_mmlu_yaml.py), with a Jinja
environment set up as lm-eval's `utils.apply_template` (StrictUndefined, keep_trailing_newline). Zero-shot context =
the rendered `description` (if any) + the rendered `doc_to_text`; choices from `doc_to_choice`; gold from
`doc_to_target`. Every item, frozen and qualification, must agree in context, choices and gold.

usage: <python with jinja2 and PyYAML> -I check_render.py <pub data dir> <lm-eval files dir> <freeze dir>
Reads only; runs no model.
"""
import ast
import json
import sys
from pathlib import Path

import jinja2
import yaml

FILES = {"arc_challenge": "allenai__ai2_arc@210d026f/test-00000-of-00001.jsonl",
         "commonsense_qa": "tau__commonsense_qa@94630fe3/validation-00000-of-00001.jsonl",
         "mmlu": "cais__mmlu@c30699e8/test-00000-of-00001.jsonl"}
ENV = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined, keep_trailing_newline=True)


class Loader(yaml.SafeLoader):
    pass


Loader.add_multi_constructor("!", lambda l, tag, node: {"__tag__": tag})   # lm-eval's !function: reported, not run


def load(d, name):
    y = yaml.load((d / name).read_text(encoding="utf-8"), Loader=Loader)
    if "include" in y:
        base = load(d, y["include"])
        base.update({k: v for k, v in y.items() if k != "include"})
        y = base
    return y


def render(t, doc):
    if isinstance(t, dict):
        raise SystemExit(f"template uses a tag: {t}")
    return ENV.from_string(t).render(**doc) if isinstance(t, str) else t


def field_or_render(t, doc):
    """lm-eval's doc_to_text/doc_to_choice/doc_to_target: a string naming one of the doc's fields returns that
    field's value; any other string is a Jinja template."""
    return doc[t] if isinstance(t, str) and t in doc else render(t, doc)


def build(cfg, doc):
    ctx = (render(cfg["description"], doc) if cfg.get("description") else "") + field_or_render(cfg["doc_to_text"], doc)
    ch = field_or_render(cfg["doc_to_choice"], doc)
    choices = ast.literal_eval(ch) if isinstance(ch, str) else list(ch)
    tgt = field_or_render(cfg["doc_to_target"], doc)
    gold = tgt if isinstance(tgt, int) else int(tgt) if tgt.isdigit() else choices.index(tgt)
    return ctx, [str(c) for c in choices], gold


def main(data_dir, lmeval_dir, freeze):
    data_dir, lmeval_dir, freeze = Path(data_dir), Path(lmeval_dir), Path(freeze)
    cfg = {"arc_challenge": load(lmeval_dir, "arc_challenge.yaml"),
           "commonsense_qa": load(lmeval_dir, "commonsense_qa_default.yaml")}
    data = {t: [json.loads(l) for l in open(data_dir / f, encoding="utf-8")] for t, f in FILES.items()}
    n, bad = 0, []
    for name in ("b3_items.jsonl", "qual_b3_items.jsonl"):
        for line in open(freeze / name, encoding="utf-8"):
            it = json.loads(line)
            doc = data[it["task"]][it["row"]]
            c = cfg.get(it["task"]) or load(lmeval_dir, f"mmlu_{doc['subject']}.yaml")
            got = build(c, doc)
            n += 1
            if got != (it["context"], it["choices"], it["gold"]):
                bad.append({"file": name, "task": it["task"], "row": it["row"],
                            "context_equal": got[0] == it["context"], "choices_equal": got[1] == it["choices"],
                            "gold_equal": got[2] == it["gold"]})
    print(json.dumps({"items_checked": n, "mismatches": len(bad), "first": bad[:5],
                      "jinja2": jinja2.__version__}, indent=1))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(*sys.argv[1:4])
