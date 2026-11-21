"""P4 part B: B4, B5 and B6's procedures (P4_benchmark_design.md section 5), as one runner with subcommands.

  text     <out>         the prompts, from B2's frozen windows (pub\\p4\\freeze_a\\b2_windows.jsonl): BOS + consecutive
                         text tokens.  B4: 4 stretches of 1,024, 8,192 and 16,000 tokens (cold + 3 warm).  B5: L - 256
                         tokens for L = 16K, 32K, 64K, 128K, 256K.  The parity prompt: the first 1,024.
  b4port   <prompts> <out> [--lengths ...]   the port on its default user path
  b4llama  <out> [--quick]                   llama-bench, the tuned placement once, at B4's points (--quick: both)
  b5       <prompts> <out> [--levels ...]    the port at each occupied context until one fails
  b6       <prompts> <out> <T GB> ...        B4's 8,192-token point under a RAM ballast, and the parity check
  parity   <prompts> <out> [--ballast GiB]   one frozen-experts request with its logits dumped (B6's parity)

The port's default user path is the clean clone's run config (PETITCHONK_RUN_CONFIG, by default cc\\strata-mistral4-ud-q4_k_m.json in the work folder; setup's
own): its engine (the release asset), its arguments and its CUDA library folder.  Two changes, both recorded in P4:
--serve (the server's protocol) with --eos-ids 0, so that every request writes all its tokens; and the expert profile
is a fresh file in the run folder (a first install's), so that every engine start begins the same way.
Every run records the machine's state, VRAM and available RAM once a second, the drive's reads (typeperf), and the
GPU link's corrected errors (WHEA 17) over the run.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(os.environ.get("PETITCHONK_WORK", "."))   # the work folder (tools/mistral4/README.md)
HERE = Path(__file__).resolve().parent
B2 = ROOT / r"pub\p4\freeze_a\b2_windows.jsonl"
P4A_MANIFEST = ROOT / r"pub\p4\FROZEN_P4A_v1.sha256"   # B2's windows are checked against it before use
CC_CFG = Path(os.environ.get("PETITCHONK_RUN_CONFIG") or ROOT / r"cc\strata-mistral4-ud-q4_k_m.json")   # the run config setup wrote
ASSET_SHA = os.environ.get("PETITCHONK_ENGINE_SHA256",
                           "8788a1198ff00b0f054fec59ad2293c68b3627da9a74cc68701c5666c1462d21")   # the 1.0.0 release engine
BENCH = ROOT / r"ref\llama.cpp-c7518dbe\build\bin\llama-bench.exe"
MODEL = ROOT / (r"models\unsloth-Mistral-Small-4-119B-2603-GGUF@bd93c721\UD-Q4_K_M"
                r"\Mistral-Small-4-119B-2603-UD-Q4_K_M-00001-of-00003.gguf")
CUDA = os.environ.get("PETITCHONK_CUDA", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0")   # 13.0: never 13.2.x on RTX 50
BOS = 1
B4_LENGTHS = (1024, 8192, 16000)
B5_LEVELS = (32768, 65536)                             # the user, 21 Nov 13:36: "Trimmed, ~45 min (Recommended)"
GEN_TOKENS = 128
PARITY_LEN = 1024
PARITY_RESERVE_MIB = 768


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def avail_gib() -> float:
    m = MemStatus()
    m.dwLength = ctypes.sizeof(MemStatus)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2**30


def gpu_used_mib() -> int:
    s = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()
    return int(s[0]) if s else -1


def ps(cmd: str) -> str:
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout.strip()


def machine_state() -> dict:
    """RAM speed (3200 = XMP on), available RAM, VRAM in use, and the heaviest processes."""
    host = ps("$m=(Get-CimInstance Win32_PhysicalMemory | Select-Object -First 1).ConfiguredClockSpeed; \"$m\"; "
              "Get-Process | Sort-Object WorkingSet64 -Descending | Select-Object -First 8 | "
              "ForEach-Object { \"$($_.Name):$([math]::Round($_.WorkingSet64/1MB))\" }").split()
    return {"time": now(), "ram_MTs": host[0] if host else None, "top_processes_MiB": host[1:],
            "available_GiB": round(avail_gib(), 2), "gpu_used_MiB": gpu_used_mib(),
            "game_running": bool(ps("Get-Process -Name Wow,Battle.net -ErrorAction SilentlyContinue | "
                                    "Select-Object -First 1 | ForEach-Object Name"))}


def whea_since(t0: str) -> int:
    """WHEA 17 events since t0 (the System log is circular, 20 MB: a count above ~12,000 is a floor)."""
    out = ps(f"@(Get-WinEvent -FilterHashtable @{{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'; "
             f"StartTime=[datetime]'{t0}'}} -ErrorAction SilentlyContinue).Count")
    return int(out) if out.isdigit() else -1


class Sampler:
    """VRAM in use and available RAM once a second, and the drive's read rate through typeperf."""

    def __init__(self, d: Path):
        self.d, self.rows, self.stop = d, [], threading.Event()
        self.csv = d / "disk_reads.csv"
        self.tp = subprocess.Popen(["typeperf", r"\PhysicalDisk(_Total)\Disk Read Bytes/sec", "-si", "1", "-o",
                                    str(self.csv), "-y"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th.start()

    def run(self):
        while not self.stop.is_set():
            self.rows.append((round(time.time(), 1), gpu_used_mib(), round(avail_gib(), 2)))
            self.stop.wait(1.0)

    def end(self) -> dict:
        self.stop.set()
        self.th.join(10)
        self.tp.terminate()
        try:
            self.tp.wait(10)
        except subprocess.TimeoutExpired:
            self.tp.kill()
        rates = []
        if self.csv.is_file():
            for line in self.csv.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
                parts = [x.strip('"') for x in line.split('","')]
                try:
                    rates.append(float(parts[1].rstrip('"')))
                except (IndexError, ValueError):
                    pass
        (self.d / "samples.json").write_text(json.dumps(self.rows), encoding="utf-8")
        return {"peak_gpu_used_MiB": max((r[1] for r in self.rows), default=None),
                "min_available_GiB": min((r[2] for r in self.rows), default=None),
                "disk_read_GB": round(sum(rates) / 1e9, 2), "disk_read_peak_MBs": round(max(rates, default=0) / 1e6, 1),
                "samples": len(self.rows)}


def run_config() -> tuple:
    """The clean clone's run config: (engine, arguments without --expert-profile, PATH folders)."""
    cfg = json.loads(CC_CFG.read_text(encoding="utf-8"))
    exe = Path(cfg["exe"])
    if sha_file(exe) != ASSET_SHA:
        sys.exit(f"{exe} is not the release asset's strata-poly.exe")
    a = list(cfg["args"])
    i = a.index("--expert-profile")
    del a[i:i + 2]
    return exe, a, list(cfg.get("lib_dirs", []))


def set_arg(args: list, flag: str, value) -> list:
    a = list(args)
    if flag in a:
        a[a.index(flag) + 1] = str(value)
    else:
        a += [flag, str(value)]
    return a


def engine_env(lib_dirs: list, strata: dict) -> dict:
    e = {k: v for k, v in os.environ.items() if not k.upper().startswith("STRATA_")}
    e["PATH"] = ";".join(lib_dirs + [e["PATH"]])
    e.update(strata)
    return e


def serve(d: Path, args_extra: dict, requests: list, strata: dict | None = None, gen: int = GEN_TOKENS) -> dict:
    """One engine start on the default user path, its requests in order (each GEN <gen> <ids>), then QUIT."""
    exe, args, lib_dirs = run_config()
    for flag, value in args_extra.items():
        args = set_arg(args, flag, value)
    argv = [str(exe), "--serve"] + args + ["--expert-profile", str(d / "expert-profile.txt"), "--eos-ids", "0"]
    rec = {"argv": argv, "strata_env": strata or {}, "state_before": machine_state(), "start": now(), "requests": []}
    sampler = Sampler(d)
    with open(d / "stderr.txt", "w", encoding="utf-8", errors="replace") as err, \
            open(d / "stdout.txt", "w", encoding="utf-8", newline="\n") as out:
        p = subprocess.Popen(argv, env=engine_env(lib_dirs, strata or {}), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=err, text=True, encoding="utf-8", errors="replace",
                             bufsize=1)

        def line():
            s = p.stdout.readline()
            if not s:
                raise RuntimeError(f"the engine closed its output (exit {p.wait()})")
            if not s.startswith("T "):
                out.write(s)
            return s.rstrip("\n")

        try:
            t = time.time()
            while not (l := line()).startswith("READY"):
                pass
            rec["ready"], rec["ready_s"] = l, round(time.time() - t, 1)
            for name, ids in requests:
                p.stdin.write(f"GEN {gen} " + ",".join(map(str, ids)) + "\n")
                p.stdin.flush()
                r = {"name": name, "n": len(ids), "start": now()}
                while True:
                    l = line()
                    if l.startswith("ERR"):
                        raise RuntimeError(f"{name}: {l}")
                    if l.startswith("DONE "):
                        f = l.split()
                        r.update(done=l, generated=int(f[1]), prompt=int(f[2]), prompt_ms=float(f[3]),
                                 decode_ms=float(f[4]), finish=f[5], reused=int(f[8]), experts_vram=int(f[9]),
                                 experts=int(f[10]), read=int(f[14]))
                        r["pp_tps"] = round(1000.0 * r["read"] / r["prompt_ms"], 2) if r["prompt_ms"] > 0 else None
                        r["tg_tps"] = round(1000.0 * r["generated"] / r["decode_ms"], 2) if r["decode_ms"] > 0 else None
                        r["vram_share"] = round(r["experts_vram"] / r["experts"], 4) if r["experts"] else None
                        break
                rec["requests"].append(r)
                print(json.dumps({k: r.get(k) for k in ("name", "n", "read", "reused", "generated", "pp_tps",
                                                         "tg_tps", "vram_share", "finish")}), flush=True)
            p.stdin.write("QUIT\n")
            p.stdin.close()
            rec["exit"] = p.wait(timeout=900)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            rec["error"] = str(e)
            p.kill()
            rec["exit"] = p.wait()
    rec["end"] = now()
    rec["measured"] = sampler.end()
    rec["whea_17_during"] = whea_since(rec["start"])
    stderr = (d / "stderr.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    rec["engine_lines"] = [s for s in stderr if s.startswith("strata-poly:")]
    return rec


def load_prompts(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_text(a):
    out = Path(a.out)
    want = [l.split()[0] for l in P4A_MANIFEST.read_text(encoding="utf-8").splitlines()
            if l.rstrip().endswith("pub/p4/freeze_a/b2_windows.jsonl")]
    if len(want) != 1 or sha_file(B2) != want[0]:
        sys.exit("B2's windows do not match part A's manifest")
    out.mkdir(parents=True, exist_ok=False)
    windows = [json.loads(l) for l in open(B2, encoding="utf-8")]
    if any(w["ids"][0] != BOS for w in windows):
        sys.exit("a B2 window does not start with BOS")
    text = [t for w in windows for t in w["ids"][1:]]
    pr = {"source": str(B2), "source_sha256": sha_file(B2), "text_tokens": len(text), "b4": {}, "b5": {}}
    for L in B4_LENGTHS:
        pr["b4"][str(L)] = [[BOS] + text[j * (L - 1):(j + 1) * (L - 1)] for j in range(4)]
    for L in B5_LEVELS:
        n = L - 256
        if n - 1 > len(text):
            sys.exit(f"B5 {L}: needs {n - 1} text tokens, the windows have {len(text)}")
        pr["b5"][str(L)] = [BOS] + text[:n - 1]
    pr["parity"] = [BOS] + text[:PARITY_LEN - 1]
    (out / "prompts.json").write_text(json.dumps(pr, separators=(",", ":")), encoding="utf-8")
    summary = {"text_tokens": len(text), "b4": {k: [len(x) for x in v] for k, v in pr["b4"].items()},
               "b5": {k: len(v) for k, v in pr["b5"].items()}, "parity": len(pr["parity"]),
               "prompts_sha256": sha_file(out / "prompts.json"), "source_sha256": pr["source_sha256"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary))


def cmd_b4port(a):
    pr, out = load_prompts(Path(a.prompts)), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    runs = []
    for L in a.lengths:
        d = out / f"len_{L}"
        d.mkdir()
        reqs = [(f"{L}_{'cold' if j == 0 else f'warm{j}'}", ids) for j, ids in enumerate(pr["b4"][str(L)][:a.requests])]
        rec = serve(d, {}, reqs)
        (d / "run.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        runs.append({"length": L, "exit": rec["exit"], "error": rec.get("error")})
        if rec["exit"] != 0:
            break
    (out / "b4port.json").write_text(json.dumps(runs, indent=1) + "\n", encoding="utf-8")
    return 0 if all(r["exit"] == 0 for r in runs) and len(runs) == len(a.lengths) else 1


def cpu_from(k: int) -> str:
    """-ot for the routed experts of layers k..35 on the CPU (19 Nov's speed_table.py)."""
    return r"blk\.[0-9]+\.ffn_.*_exps=CPU" if k <= 0 else rf"blk\.([{k}-9]|[1-3][0-9])\.ffn_.*_exps=CPU"


def cmd_b4llama(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["PATH"] = rf"{CUDA}\bin\x64;" + env["PATH"]
    pts = "1024,8192,16000" if not a.quick else "512"
    reps = "1"                                         # trimmed P6: one repetition (19 Nov's spreads were small)
    gen = "128" if not a.quick else "16"
    runs = []
    # trimmed P6: llama.cpp's best placement on this card only; --quick (the qualification) keeps both
    for place, k in ((("baseline", 0), ("tuned", 4)) if a.quick else (("tuned", 4),)):
        for what, extra in (("prompt", ["-p", pts, "-n", "0"]), ("gen", ["-p", "0", "-n", gen, "-d", pts])):
            tag = f"{place}_{what}"
            argv = [str(BENCH), "-m", str(MODEL), "-ngl", "99", "-ot", cpu_from(k), "-fa", "1", "-t", "10",
                    *extra, "-r", reps, "-o", "json"]
            rec = {"tag": tag, "argv": argv, "state_before": machine_state(), "start": now()}
            d = out / tag
            d.mkdir()
            sampler = Sampler(d)
            r = subprocess.run(argv, env=env, capture_output=True, timeout=4 * 3600)
            (d / "stdout.json").write_bytes(r.stdout)
            (d / "stderr.txt").write_bytes(r.stderr)
            rec.update(exit=r.returncode, end=now(), measured=sampler.end())
            rec["whea_17_during"] = whea_since(rec["start"])
            try:
                rec["results"] = [{k2: t.get(k2) for k2 in ("n_prompt", "n_gen", "n_depth", "avg_ts", "stddev_ts")}
                                  for t in json.loads(r.stdout)]
            except ValueError:
                rec["results"] = None
            (d / "run.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
            print(json.dumps({k2: rec[k2] for k2 in ("tag", "exit", "results")}), flush=True)
            runs.append({"tag": tag, "exit": rec["exit"]})
            if rec["exit"] != 0:
                break
    (out / "b4llama.json").write_text(json.dumps(runs, indent=1) + "\n", encoding="utf-8")
    return 0 if all(r["exit"] == 0 for r in runs) and len(runs) == (4 if a.quick else 2) else 1


def cmd_b5(a):
    pr, out = load_prompts(Path(a.prompts)), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    runs = []
    for L in a.levels:
        d = out / f"ctx_{L}"
        d.mkdir()
        rec = serve(d, {"--max-context": L}, [(f"ctx_{L}", pr["b5"][str(L)])])
        rec["level"] = L
        (d / "run.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        ok = rec["exit"] == 0 and "error" not in rec and len(rec["requests"]) == 1
        runs.append({"level": L, "ok": ok, "exit": rec["exit"], "error": rec.get("error")})
        if not ok:
            break
    (out / "b5.json").write_text(json.dumps(runs, indent=1) + "\n", encoding="utf-8")
    return 0


class Ballast:
    def __init__(self, gib: float):
        self.p = subprocess.Popen([sys.executable, "-I", str(HERE / "ballast.py"), str(gib)], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True)
        self.locked = self.p.stdout.readline().strip()
        if not self.locked.startswith("LOCKED"):
            self.p.kill()
            raise RuntimeError(f"ballast {gib} GiB: {self.locked}")

    def release(self) -> str:
        self.p.stdin.write("\n")
        self.p.stdin.flush()
        r = self.p.stdout.readline().strip()
        self.p.wait(60)
        return r


def cmd_b6(a):
    pr, out = load_prompts(Path(a.prompts)), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    runs = []
    for T in a.targets:
        gib = 128 - T
        d = out / f"pc_{T}GB"
        d.mkdir()
        rec = {"target_GB": T, "ballast_GiB": gib, "available_before_GiB": round(avail_gib(), 2)}
        b = Ballast(gib)
        rec["ballast"] = b.locked
        try:
            reqs = [(f"8192_{'cold' if j == 0 else f'warm{j}'}", ids) for j, ids in enumerate(pr["b4"]["8192"][:a.requests])]
            rec["run"] = serve(d, {}, reqs)
        finally:
            rec["ballast_release"] = b.release()
        (d / "run.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        runs.append({"target_GB": T, "exit": rec["run"]["exit"], "error": rec["run"].get("error")})
    (out / "b6.json").write_text(json.dumps(runs, indent=1) + "\n", encoding="utf-8")
    return 0 if all(r["exit"] == 0 for r in runs) else 1


def cmd_parity(a):
    """One request with the experts' split frozen and a fixed VRAM reserve, every prompt row's logits dumped."""
    pr, out = load_prompts(Path(a.prompts)), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    dump = out / "dump"
    strata = {"STRATA_FREEZE_EXPERTS": "1", "STRATA_LOGITS_DIR": str(dump), "STRATA_LOGITS_ALL_ROWS": "1"}
    rec = {"ballast_GiB": a.ballast}
    b = Ballast(a.ballast) if a.ballast else None
    rec["ballast"] = b.locked if b else None
    try:
        rec["run"] = serve(out, {"--vram-reserve-mib": PARITY_RESERVE_MIB}, [("parity", pr["parity"])], strata, gen=1)
    finally:
        rec["ballast_release"] = b.release() if b else None
    files = sorted(p for p in dump.rglob("*") if p.is_file()) if dump.is_dir() else []
    rec["dump"] = {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha_file(p)} for p in files}
    (out / "run.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"exit": rec["run"]["exit"], "dump": rec["dump"]}), flush=True)
    return 0 if rec["run"]["exit"] == 0 and files else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("text"); s.add_argument("out")
    s = sub.add_parser("b4port"); s.add_argument("prompts"); s.add_argument("out")
    s.add_argument("--lengths", type=int, nargs="+", default=list(B4_LENGTHS)); s.add_argument("--requests", type=int, default=4)
    s = sub.add_parser("b4llama"); s.add_argument("out"); s.add_argument("--quick", action="store_true")
    s = sub.add_parser("b5"); s.add_argument("prompts"); s.add_argument("out")
    s.add_argument("--levels", type=int, nargs="+", default=list(B5_LEVELS))
    s = sub.add_parser("b6"); s.add_argument("prompts"); s.add_argument("out")
    s.add_argument("targets", type=int, nargs="+"); s.add_argument("--requests", type=int, default=4)
    s = sub.add_parser("parity"); s.add_argument("prompts"); s.add_argument("out")
    s.add_argument("--ballast", type=float, default=0.0)
    a = ap.parse_args()
    return {"text": cmd_text, "b4port": cmd_b4port, "b4llama": cmd_b4llama, "b5": cmd_b5, "b6": cmd_b6,
            "parity": cmd_parity}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
