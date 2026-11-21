"""A stand-in for strata-poly in p4_cand.py's logic test: the serve and one-shot protocol and the logits dump, no model.

Row for position j of a sequence = a deterministic function of ids[:j+1] (so a row depends on its whole prefix, as a
real model's does). STUB_FAULT (environment) makes it misbehave on purpose: "reused" reports 1 reused token on the
second request; "shift" dumps the second request's rows with pos starting at 1; "nonfinite" puts one NaN in row 1.
"""
import json
import os
import struct
import sys
import zlib
from array import array

V = 131072


def row(prefix):
    seed = zlib.crc32(struct.pack(f"<{len(prefix)}i", *prefix))
    a = array("f", bytes(4 * V))
    x = seed or 1
    for k in range(0, V, 97):                 # sparse, cheap, deterministic
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        a[k] = (x % 2000) / 100.0 - 10.0
    return a


class Dump:
    def __init__(self, d):
        os.makedirs(d, exist_ok=True)
        self.f = open(os.path.join(d, "logits.f32"), "wb")
        self.ix = open(os.path.join(d, "logits.jsonl"), "w", encoding="utf-8", newline="\n")
        self.rows = self.calls = 0

    def seq(self, ids, shift=0, nonfinite=False):
        for j in range(len(ids)):
            r = row(ids[:j + 1])
            if nonfinite and j == 1:
                r[5] = float("nan")
            r.tofile(self.f)
            self.ix.write(json.dumps({"row": self.rows, "call": self.calls, "pos": j + shift, "token": ids[j],
                                      "src": "stub", "n": len(ids), "last": j == len(ids) - 1}) + "\n")
            self.rows += 1
        self.calls += 1
        self.f.flush()
        self.ix.flush()


def main(argv):
    fault = os.environ.get("STUB_FAULT", "")
    dump = Dump(os.environ["STRATA_LOGITS_DIR"])
    assert os.environ.get("STRATA_LOGITS_ALL_ROWS") == "1"
    if "--serve" in argv:
        print("READY 16384 stop", flush=True)
        k = 0
        for line in sys.stdin:
            line = line.strip()
            if line == "QUIT":
                break
            _, mx, ids = line.split(" ", 2)
            ids = [int(t) for t in ids.split(",")]
            dump.seq(ids, shift=1 if (fault == "shift" and k == 1) else 0, nonfinite=(fault == "nonfinite"))
            reused = 1 if (fault == "reused" and k == 1) else 0
            print(f"RESUME {reused}", flush=True)
            print("T 7", flush=True)
            print(f"DONE 1 {len(ids)} 1.0 1.0 length 0 0 {reused} 0 0 0 0 0.0 {len(ids) - reused} 0", flush=True)
            k += 1
        return 0
    ids = [int(t) for t in open(argv[argv.index("--tokens-file") + 1], encoding="utf-8").read().split(",")]
    dump.seq(ids, nonfinite=(fault == "nonfinite"))
    print("7")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
