"""Exploration: P5's B3 harness (../bench/p4_cand.py, imported unchanged) with every prompt read a few tokens
at a time, so that every row takes the few-token routing path where --cache-bias acts (calls under kStageRows = 32
rows), and optionally with --cache-bias.

It replaces only p4_cand.FLAGS, before p4_cand.main() runs:
  ["--lookup", "0", "--window", "1", "--eos-ids", "0", "--prompt-batch", <N>] + (["--cache-bias", <L>] if L > 0)
(P5's: the same with --prompt-batch 1024 and no bias.) The flags used are written to <out>.flags.json beside the out
dir before the run. Everything else is p4_cand.py's: modes, batches, checks, dumps, scoring of rows, run.json.

usage: <venv-tools python> -I b3_steered.py --lam L --prompt-batch N <p4_cand.py arguments>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))   # p4_cand.py
import p4_cand  # noqa: E402


def take(args, name, conv):
    i = args.index(name)
    v = conv(args[i + 1])
    del args[i:i + 2]
    return v


def main():
    args = sys.argv[1:]
    lam = take(args, "--lam", float)
    pb = take(args, "--prompt-batch", int)
    if not (0 <= lam <= 16) or not (1 <= pb < 32):
        sys.exit("--lam 0..16 and --prompt-batch 1..31 (under 32, so every call is steered)")
    flags = ["--lookup", "0", "--window", "1", "--eos-ids", "0", "--prompt-batch", str(pb)]
    if lam > 0:
        flags += ["--cache-bias", str(lam)]
    p4_cand.FLAGS = flags
    out = Path(args[args.index("--out") + 1])
    if "--execute" in args:
        Path(str(out) + ".flags.json").write_text(json.dumps({"flags": flags, "lam": lam, "prompt_batch": pb},
                                                             indent=1) + "\n", encoding="utf-8")
    sys.argv = ["p4_cand.py"] + args
    p4_cand.main()


if __name__ == "__main__":
    main()
