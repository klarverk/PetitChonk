"""Tests for the Tekken pre-tokenizer of tools/strata_tokenizer.py (Mistral Small 4, llama.cpp architecture
`mistral4`, GGUF `tokenizer.ggml.pre` = `pixtral`), and for the refusal of any other pre-tokenizer for a `mistral4`
model. They need no model file.

    python -m unittest tools.test_strata_tokenizer_tekken -v

The expected pieces are the ORACLE's: tokenizer.json's own Split pre-tokenizer
(mistralai/Mistral-Small-4-119B-2603 at a11f36be), run by Hugging Face `tokenizers` 0.23.3.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import regex

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gguf_reader  # noqa: E402
import strata_tokenizer as ST  # noqa: E402

PIECES = {   # text -> the oracle's pieces
    "CamelCaseWord HTTPServer": ["Camel", "Case", "Word", " HTTPServer"],
    "it'd they'll": ["it", "'d", " they", "'ll"],
    "pi 3.14 x=1234": ["pi", " ", "3", ".", "1", "4", " x", "=", "1", "2", "3", "4"],
    "a  b\n\nc": ["a", " ", " b", "\n\n", "c"],
    "path/to/file.\n/next": ["path", "/to", "/file", ".\n/", "next"],
    "Kaffé Åke": ["Kaffé", " Åke"],
}


class _Fake:
    def __init__(self, md):
        self.metadata = md


def _from_metadata(md):
    """Tokenizer.from_gguf on a metadata dict, with GGUFFile replaced for the call."""
    saved = gguf_reader.GGUFFile
    gguf_reader.GGUFFile = lambda path: _Fake(md)
    try:
        return ST.Tokenizer.from_gguf("unused.gguf")
    finally:
        gguf_reader.GGUFFile = saved


BASE = {"tokenizer.ggml.tokens": ["a", "b", "ab"], "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.token_type": [1, 1, 1]}


class TekkenPreTokenizer(unittest.TestCase):
    def test_names_use_the_tekken_pattern_and_ignore_merges(self):
        for name in ("pixtral", "tekken"):
            tk = ST.Tokenizer(["a", "b", "ab"], ["a b"], [1, 1, 1], pre=name)
            self.assertEqual(tk.pre_pattern, ST.TEKKEN_PATTERN)
            self.assertTrue(tk.ignore_merges)

    def test_pieces_match_the_oracle(self):
        rx = regex.compile(ST.TEKKEN_PATTERN)
        for text, want in PIECES.items():
            self.assertEqual(rx.findall(text), want, text)


class Mistral4Refusal(unittest.TestCase):
    def test_mistral4_with_pixtral_loads(self):
        tk = _from_metadata(dict(BASE, **{"general.architecture": "mistral4", "tokenizer.ggml.pre": "pixtral"}))
        self.assertEqual(tk.pre_pattern, ST.TEKKEN_PATTERN)

    def test_mistral4_with_another_pre_refuses(self):
        with self.assertRaises(ValueError):
            _from_metadata(dict(BASE, **{"general.architecture": "mistral4", "tokenizer.ggml.pre": "qwen35"}))

    def test_other_architectures_keep_the_fallback(self):
        tk = _from_metadata(dict(BASE, **{"general.architecture": "qwen35moe", "tokenizer.ggml.pre": "unknown-name"}))
        self.assertEqual(tk.pre_pattern, ST.QWEN35_PATTERN)


if __name__ == "__main__":
    unittest.main()
