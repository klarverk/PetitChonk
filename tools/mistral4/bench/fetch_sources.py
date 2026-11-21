"""PCHONK-PUB-001 P4: fetch the pinned benchmark sources and convert them to JSON lines.

Each file is downloaded from Hugging Face at its pinned revision, and its SHA-256 must equal the LFS object id the
Hub lists for that path at that revision. Then pyarrow (here: the system Python 3.13's pyarrow 22.0.0; neither
project venv has a Parquet reader) writes it as JSON lines next to it. Writes SHA256SUMS over every file it made.

Sources (P4_benchmark_design.md section 1; CommonsenseQA replaces HellaSwag, the user's decision of 20 Nov):
  B2  Salesforce/wikitext   b08601e0  wikitext-2-raw-v1/test
  B3  allenai/ai2_arc       210d026f  ARC-Challenge/test
  B3  tau/commonsense_qa    94630fe3  data/validation (the test split has no public answers)
  B3  cais/mmlu             c30699e8  all/test

usage: <python with pyarrow> -I fetch_sources.py <out dir>
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

SOURCES = [
    ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3", "wikitext-2-raw-v1/test-00000-of-00001.parquet"),
    ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Challenge/test-00000-of-00001.parquet"),
    ("tau/commonsense_qa", "94630fe30dad47192a8546eb75f094926d47e155", "data/validation-00000-of-00001.parquet"),
    ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", "all/test-00000-of-00001.parquet"),
]


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=120) as r:
        return r.read()


def main(out):
    out = Path(out)
    made = []
    for repo, rev, path in SOURCES:
        folder, name = path.rsplit("/", 1)
        tree = json.loads(get(f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/{folder}"))
        oid = next(f["lfs"]["oid"] for f in tree if f["path"] == path)
        data = get(f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}")
        got = hashlib.sha256(data).hexdigest()
        if got != oid:
            raise SystemExit(f"{repo}@{rev}/{path}: SHA-256 {got} is not the Hub's {oid}")
        d = out / f"{repo.replace('/', '__')}@{rev[:8]}"
        d.mkdir(parents=True, exist_ok=True)
        pf = d / name
        pf.write_bytes(data)
        rows = pq.read_table(pf).to_pylist()
        jl = pf.with_suffix(".jsonl")
        with open(jl, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        made += [pf, jl]
        print(f"{repo}@{rev[:8]} {path}: {len(data)} bytes, SHA-256 = the Hub's LFS id; {len(rows)} rows", flush=True)
    sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out).as_posix()}" for p in made]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8", newline="\n")
    print("SHA256SUMS", hashlib.sha256((out / "SHA256SUMS").read_bytes()).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1])
