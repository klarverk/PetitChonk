"""T3 (PCHONK-FTI-001): build, tokenize and freeze the fixed token sequences, and check tokenizer parity.

Sequences (D-FTI4), each cut at a paragraph or line boundary to a token target:
  en_prose   Jane Austen, Persuasion (Gutenberg #105), from CHAPTER III; ~2,000 tokens
  sv_prose   Hjalmar Bergman, Eros' begravning (Gutenberg #12613), from its first paragraph; ~2,000 tokens
  code_cpp   PolyStrata src/poly/cpu_experts.cpp at 7a6c0983 (MIT); ~2,500 tokens
  code_py    PolyStrata tools/gguf_reader.py at 7a6c0983 (MIT); the whole file
  math       an authored maths and structured-text document; the whole file
  chat_en    an authored four-turn chat, rendered with the model's template and default system prompt (render_chat.py)
  long_doc   A. Conan Doyle, The Hound of the Baskervilles (Gutenberg #2852), from Chapter 1; ~15,000 tokens

Reference ids: mistral-common 1.13.1 with tekken.json at a11f36be. Plain texts: encode(text, bos=True, eos=False),
as S6a. The chat: its special tokens by get_special_token, the text between them by encode(bos=False, eos=False).
Parity: mistral4/full's tools/strata_tokenizer.py, from the GGUF (from_gguf) and from the server's tokenizer folder
(from_dir), must give the same ids. Also: the tokenizer's unit tests on mistral4/full, and a screen against the
texts the candidate has already seen (regime §5.2).

usage: <venv-tools python> -I make_sequences.py <out dir>
"""
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
import unittest
import os
from pathlib import Path

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
import mistral_common

ROOT = Path(os.environ.get("PETITCHONK_WORK", "."))   # the work folder (tools/mistral4/README.md)
SRC = ROOT / "fti" / "sources"
REPO = Path(__file__).resolve().parents[3]   # this repository
FULL = REPO   # the candidate checkout: this repository
MODEL = ROOT / (r"models\unsloth-Mistral-Small-4-119B-2603-GGUF@bd93c721\UD-Q4_K_M"
                r"\Mistral-Small-4-119B-2603-UD-Q4_K_M-00001-of-00003.gguf")
TEKKEN = ROOT / r"models\mistralai-Mistral-Small-4-119B-2603@a11f36be\tekken.json"
TDIR = ROOT / r"serve-mistral4\tokenizer"
SEEN = ["metal spoon in a cup of hot tea", "gears and derailleurs of a road bicycle",
        "refrigerator moves heat out of its cabinet", "The lighthouse stood at the end of a narrow spit"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def gutenberg_body(name):
    t = (SRC / "gutenberg" / name).read_text(encoding="utf-8").replace("\r\n", "\n")
    s, e = t.find("*** START OF"), t.find("*** END OF")
    return t[t.find("\n", s) + 1:e]


def cut(text, start_marker, target, enc):
    """From start_marker, whole paragraphs (blank-line separated) until the next one would pass `target` ids."""
    i = text.find(start_marker)
    assert i >= 0, start_marker
    paras = re.split(r"(\n\s*\n)", text[i:])
    out = ""
    for j in range(0, len(paras), 2):
        cand = out + paras[j] + (paras[j + 1] if j + 1 < len(paras) else "")
        if len(enc(cand.rstrip() + "\n")) > target:
            break
        out = cand
    return out.rstrip() + "\n"


def cut_lines(text, target, enc):
    lines, out = text.split("\n"), ""
    for ln in lines:
        if len(enc(out + ln + "\n")) > target:
            break
        out += ln + "\n"
    return out


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    mt = MistralTokenizer.from_file(str(TEKKEN))
    tk = mt.instruct_tokenizer.tokenizer
    enc_plain = lambda s: tk.encode(s, bos=True, eos=False)  # noqa: E731

    def enc_chat(text):
        ids, pos = [], 0
        for m in re.finditer(r"</?s>|\[/?[A-Z_]+\]", text):
            try:
                sid = tk.get_special_token(m.group(0))
            except Exception:
                continue
            if m.start() > pos:
                ids += tk.encode(text[pos:m.start()], bos=False, eos=False)
            ids.append(sid)
            pos = m.end()
        if pos < len(text):
            ids += tk.encode(text[pos:], bos=False, eos=False)
        return ids

    spec = importlib.util.spec_from_file_location("st_full", FULL / "tools" / "strata_tokenizer.py")
    st = importlib.util.module_from_spec(spec)
    sys.modules["st_full"] = st
    spec.loader.exec_module(st)
    tok_gguf, tok_dir = st.Tokenizer.from_gguf(str(MODEL)), st.Tokenizer.from_dir(TDIR)

    git = lambda p: subprocess.run(["git", "-C", str(REPO), "show", f"7a6c0983:{p}"], capture_output=True,  # noqa: E731
                                   check=True).stdout.decode("utf-8")
    texts = {
        "en_prose": (cut(gutenberg_body("105-0.txt"), "CHAPTER III.", 2000, enc_plain), "plain",
                     "Jane Austen, Persuasion (1818), Project Gutenberg #105 (105-0.txt, via gutenberg.pglaf.org), "
                     "from CHAPTER III", "public domain"),
        "sv_prose": (cut(gutenberg_body("12613-0.txt"), "Fru Olga Janselius", 2000, enc_plain), "plain",
                     "Hjalmar Bergman, Eros' begravning (1915; text of the 1922 edition), Project Gutenberg #12613 "
                     "(12613-0.txt), from its first paragraph", "public domain (Bergman d. 1931)"),
        "code_cpp": (cut_lines(git("src/poly/cpu_experts.cpp"), 2500, enc_plain), "plain",
                     "PolyStrata src/poly/cpu_experts.cpp at 7a6c0983, from its first line", "MIT"),
        "code_py": (git("tools/gguf_reader.py"), "plain", "PolyStrata tools/gguf_reader.py at 7a6c0983, whole file",
                    "MIT"),
        "math": ((SRC / "authored" / "math_structured.txt").read_text(encoding="utf-8"), "plain",
                 "authored for T3 (fti/sources/authored/math_structured.txt)", "authored here"),
        "chat_en": ((SRC / "rendered" / "chat_en.rendered.txt").read_text(encoding="utf-8"), "chat",
                    "authored four-turn chat (fti/sources/authored/chat_en.json), rendered by render_chat.py with "
                    "the GGUF's template, its default system prompt, reasoning_effort none, date fixed 2026-11-19",
                    "authored here; the system prompt is Mistral's, from the model's template"),
        "long_doc": (cut(gutenberg_body("2852-0.txt"), "Chapter 1.", 15000, enc_plain), "plain",
                     "A. Conan Doyle, The Hound of the Baskervilles (1902), Project Gutenberg #2852 (2852-0.txt), "
                     "from Chapter 1", "public domain"),
    }
    seqs, ok_all = {}, True
    for name, (text, kind, source, licence) in texts.items():
        ref = enc_plain(text) if kind == "plain" else enc_chat(text)
        a = ([1] + tok_gguf.encode(text)) if kind == "plain" else tok_gguf.encode(text, parse_special=True)
        b = ([1] + tok_dir.encode(text)) if kind == "plain" else tok_dir.encode(text, parse_special=True)
        seen = [s for s in SEEN if s.lower() in text.lower()]
        first_diff = next((i for i, (x, y) in enumerate(zip(ref, a)) if x != y), None)
        ok = ref == a == b and not seen and all(i < 131072 for i in ref)
        ok_all &= ok
        (out / f"{name}.txt").write_text(text, encoding="utf-8", newline="\n")
        seqs[name] = {"kind": kind, "source": source, "licence": licence, "chars": len(text), "n_ids": len(ref),
                      "text_sha256": sha(text.encode("utf-8")), "ids_sha256": sha(json.dumps(ref).encode()),
                      "parity_from_gguf": ref == a, "parity_from_dir": ref == b, "first_difference": first_diff,
                      "seen_text_found": seen, "decodes_back": tk.decode(ref) == text.replace("<s>", "", 1)
                      if kind == "plain" else None, "ids": ref}
        print(f"{name:9s} {kind:5s} {len(text):6d} chars {len(ref):6d} ids  parity gguf {ref == a} dir {ref == b}"
              f"  seen {seen}", flush=True)
    # the tokenizer's unit tests on mistral4/full
    sys.path.insert(0, str(FULL / "tools"))
    tspec = importlib.util.spec_from_file_location("t_tekken", FULL / "tools" / "test_strata_tokenizer_tekken.py")
    tmod = importlib.util.module_from_spec(tspec)
    tspec.loader.exec_module(tmod)
    buf = io.StringIO()
    res = unittest.TextTestRunner(stream=buf, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tmod))
    tests = {"run": res.testsRun, "failures": len(res.failures), "errors": len(res.errors),
             "ok": res.wasSuccessful(), "log": buf.getvalue()}
    print("unit tests:", {k: tests[k] for k in ("run", "failures", "errors", "ok")})
    manifest = {"tokenizer": {"file": str(TEKKEN), "sha256": sha(TEKKEN.read_bytes()),
                              "package": f"mistral-common {mistral_common.__version__}"},
                "candidate_tokenizer": {"file": str(FULL / "tools" / "strata_tokenizer.py"),
                                        "sha256": sha((FULL / "tools" / "strata_tokenizer.py").read_bytes()),
                                        "branch_head": subprocess.run(["git", "-C", str(FULL), "rev-parse", "HEAD"],
                                                                      capture_output=True, text=True).stdout.strip()},
                "unit_tests": tests, "all_ok": ok_all and tests["ok"],
                "total_ids": sum(s["n_ids"] for s in seqs.values()), "sequences": seqs}
    (out / "sequences.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(f"{sha(p.read_bytes())}  {p.name}\n" for p in files), encoding="utf-8",
                                    newline="\n")
    print("total ids", manifest["total_ids"], "| all_ok", manifest["all_ok"])
    print("SHA256SUMS", sha((out / "SHA256SUMS").read_bytes()))


if __name__ == "__main__":
    main(sys.argv[1])
