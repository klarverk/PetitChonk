"""PCHONK-PUB-001 P4: fetch lm-evaluation-harness's B3 task files at the pinned commit (P4_benchmark_design.md §2).

B3's contexts follow lm-eval's task templates. This fetches, at v0.4.13 (`ddd67220`), through `gh api` (raw contents):
the ARC and CommonsenseQA task files, MMLU's `_default_template_yaml` and the 57 `mmlu_<subject>.yaml` (each sets its
subject's `description`). It writes them with a SHA256SUMS and checks the five files whose hashes P4 already recorded
(prefixes from the record's table). check_render.py renders the frozen contexts from these files.

usage: python -I fetch_mmlu_yaml.py <mmlu test jsonl> <out dir>
The out dir must not exist. Downloads only; runs no model.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = "EleutherAI/lm-evaluation-harness"
COMMIT = "ddd67220430a2470529f25fd5c05a576ca1057a0"
MMLU = "lm_eval/tasks/mmlu/default"
# (path in the repository, saved name, SHA-256 prefix P4 recorded or None)
FIXED = [("lm_eval/tasks/arc/arc_easy.yaml", "arc_easy.yaml", "96da1d9e"),
         ("lm_eval/tasks/arc/arc_challenge.yaml", "arc_challenge.yaml", "682621c2"),
         ("lm_eval/tasks/commonsense_qa/default.yaml", "commonsense_qa_default.yaml", "61699aec"),
         (f"{MMLU}/_default_template_yaml", "_default_template_yaml", "a0997593")]
KNOWN_SUBJECT = {"mmlu_abstract_algebra.yaml": "f026cc89"}


def fetch(path):
    r = subprocess.run(["gh", "api", "-H", "Accept: application/vnd.github.raw",
                        f"repos/{REPO}/contents/{path}?ref={COMMIT}"], capture_output=True)
    if r.returncode:
        raise SystemExit(f"fetch failed: {path}: {r.stderr.decode(errors='replace')}")
    return r.stdout


def main(mmlu, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    subjects = sorted({json.loads(l)["subject"] for l in open(mmlu, encoding="utf-8")})
    todo = FIXED + [(f"{MMLU}/mmlu_{s}.yaml", f"mmlu_{s}.yaml", KNOWN_SUBJECT.get(f"mmlu_{s}.yaml")) for s in subjects]
    sums, checked = [], []
    for path, name, known in todo:
        b = fetch(path)
        (out / name).write_bytes(b)
        h = hashlib.sha256(b).hexdigest()
        sums.append(f"{h} *{name}\n")
        if known:
            if not h.startswith(known):
                raise SystemExit(f"{name}: {h[:8]} is not the recorded {known}")
            checked.append(name)
    (out / "SHA256SUMS").write_text("".join(sums), encoding="utf-8", newline="\n")
    print(json.dumps({"commit": COMMIT, "files": len(todo), "subjects": len(subjects),
                      "recorded_hashes_match": checked}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
