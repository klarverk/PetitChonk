"""Tests for setup.py's Mistral Small 4 (Petit Chonk's mistral4 backend): its pinned files, the estimates for the
16 GB card it was measured on, the install by strata-poly's steps, and the choices of its run config that are its
own - no guesses ahead (--lookup 0 --window 1), 16K recommended, the longest context its parity was verified at, and
the cache bias (--cache-bias: off by default, 0.25 or 0.5; Petit Chonk 1.1.0).  Mocked like tools/test_setup_glm.py,
whose harness this uses.

    python -m unittest tools.test_setup_mistral4
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools import test_setup_glm as glm_tests  # noqa: E402

FAM = setup.FAMILIES["mistral4"]
SIZE = "UD-Q4_K_M"
FILES = [("Mistral-Small-4-119B-2603-UD-Q4_K_M-00001-of-00003.gguf", 7875616,
          "fd3cc46082e4e64eb623eea4611b02583d1de4c20a59411bf1820925d5117dfe"),
         ("Mistral-Small-4-119B-2603-UD-Q4_K_M-00002-of-00003.gguf", 49299298464,
          "f357d8dc029824fa4515ae60d7a8b7e108546763614f16c00a1b4ea0cd94145c"),
         ("Mistral-Small-4-119B-2603-UD-Q4_K_M-00003-of-00003.gguf", 24456006464,
          "9dc960d67fb1ef23029d24878b4cf6c63b743ab26cf9137402549591bd770c50")]
RTX_5080 = 16303 / 1024                                  # nvidia-smi's memory.total, as setup reads it


class Pins(unittest.TestCase):
    def test_the_files(self):
        self.assertEqual(setup.HF_REVISIONS["unsloth/Mistral-Small-4-119B-2603-GGUF"],
                         "bd93c721735aa32c035c0f19e738cb3371fd56ff")
        self.assertEqual(FAM["hf"], "https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF/resolve/"
                                    "bd93c721735aa32c035c0f19e738cb3371fd56ff/")
        self.assertEqual(list(setup.family_models("mistral4")), [SIZE])
        self.assertEqual(setup.model_shards(FAM, SIZE), 3)
        self.assertEqual([setup.model_file(FAM, SIZE, i) for i in (1, 2, 3)], [f"{SIZE}/{f}" for f, _, _ in FILES])
        self.assertEqual(FAM["sha256"], {f: (n, sha) for f, n, sha in FILES})
        d = FAM["models"][SIZE]
        self.assertAlmostEqual(sum(n for _, n, _ in FILES) / 1e9, d["download_gb"], delta=0.05)
        self.assertLess(d["arena_gb"], d["download_gb"])
        self.assertEqual(d["ram_gb"], round(setup.poly_ram_need(d, RTX_5080)))

    def test_the_architecture(self):
        self.assertEqual(setup.POLY_GGUF_ARCHS["mistral4"], ("mistral4",))

    def test_the_about_line_says_the_ram_beside_the_card_it_ran_on(self):
        n = setup.poly_ram_need(FAM["models"][SIZE], RTX_5080)
        self.assertIn(f"~{n:.0f} GB of RAM beside a 16 GB card", FAM["about"])
        self.assertEqual(FAM["ran_on_gb"], 16)

    def test_no_figure_from_another_card(self):
        # the menu labels laptop_tps "8 GB card"; batch_gb would make setup read prompts 2,048 at a time, a batch
        # the parity runs did not use
        for key in ("laptop_tps", "batch_gb", "batch_from_gb", "tail_skip"):
            self.assertNotIn(key, FAM)


class OtherFamilies(unittest.TestCase):
    def test_no_other_family_has_mistral4_s_own_choices(self):
        for f, fam in setup.FAMILIES.items():
            if f != "mistral4":
                for key in ("engine_args", "context", "verified_context", "cuda_only", "cache_bias"):
                    self.assertNotIn(key, fam, f)
        for f, fam in setup.OWN_PATH_FAMILIES.items():
            self.assertNotIn("cache_bias", fam, f)

    def test_the_cache_bias_values_offered(self):
        # off (the default) and the two values measured on 22 Nov 2026 (decisions D-11-1, D-11-2)
        self.assertEqual(list(FAM["cache_bias"]), ["0.25", "0.5"])


class Platforms(unittest.TestCase):
    """Petit Chonk supports and tests Mistral Small 4 on NVIDIA cards (CUDA) only: its AMD, Intel and Mac engines are
    neither built nor tested, so setup does not offer it there."""

    def test_not_in_the_menu_on_amd_and_intel(self):
        self.assertNotIn("mistral4", setup.menu_families(True))
        self.assertEqual(setup.menu_families(False)[-1], "mistral4")
        self.assertIn("lfm25", setup.menu_families(True))

    def test_the_family_refuses_amd_and_intel(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            setup.poly_setup(types.SimpleNamespace(), "mistral4", {"arch": "gfx1201", "index": 0}, True, None, None,
                             64.0, None, [], 8080, None)
        self.assertIn("NVIDIA cards (CUDA) only", out.getvalue())

    def test_not_offered_on_a_mac(self):
        sys.path.insert(0, str(ROOT / "metal"))
        import setup_mac  # noqa: E402
        self.assertNotIn("mistral4", setup_mac.MAC_POLY_FAMILIES)
        self.assertIn("lfm25", setup_mac.MAC_POLY_FAMILIES)


class Main(unittest.TestCase):
    setUp, tearDown, main = glm_tests.Main.setUp, glm_tests.Main.tearDown, glm_tests.Main.main

    def config(self):
        return json.loads((self.t / "strata-mistral4-ud-q4_k_m.json").read_text())

    def test_install(self):
        code, out, _ = self.main(family="mistral4", ram=127.9, vram=RTX_5080)
        self.assertEqual(code, 0, out)
        cfg = self.config()
        tag = "mistral4-" + SIZE
        self.assertEqual(self.downloads, [FAM["hf"] + f"{SIZE}/{f}" for f, _, _ in FILES])
        self.assertEqual(self.verified, [(f, n, sha) for f, n, sha in FILES])
        self.assertEqual([Path(r[1]).name for r in self.runs], ["strata_tokenizer.py"])
        args = cfg["args"]
        pack = self.t / "data" / "packs" / tag.lower()
        self.assertEqual(Path(args[args.index("--model") + 1]), self.t / "models" / tag / SIZE / FILES[0][0])
        self.assertEqual(Path(args[args.index("--expert-profile") + 1]), pack / "expert-profile.txt")
        self.assertEqual(args[args.index("--max-context") + 1], "16384")      # --yes takes the recommended one
        self.assertEqual((args.count("--lookup"), args[args.index("--lookup") + 1]), (1, "0"))
        self.assertEqual((args.count("--window"), args[args.index("--window") + 1]), (1, "1"))
        for flag in ("--prompt-batch", "--draft-model", "--tail-skip", "--kv", "--cache-bias"):
            self.assertNotIn(flag, args)
        self.assertIn("cache bias: off", out)                                  # --yes: the cache bias is off
        self.assertEqual(cfg["model_name"], "mistral-small-4-119b-2603-ud-q4_k_m")
        self.assertEqual(cfg["display_name"], "Mistral Small 4 UD-Q4_K_M")
        self.assertNotIn("vision", cfg)
        self.assertNotIn("[experimental]", out)

    def test_the_context_question(self):
        code, out, _ = self.main(family="mistral4", ram=127.9, vram=RTX_5080, answers={})
        self.assertEqual(code, 0, out)
        for line in ("1) 8K tokens\n", "2) 16K tokens   (recommended)",
                     "3) 32K tokens   (0.4 GB more VRAM; not verified for this model)",
                     "4) 64K tokens   (1.1 GB more VRAM; not verified for this model)",
                     "5) 128K tokens   (2.6 GB more VRAM; not verified for this model)"):
            self.assertIn(line, out)
        args = self.config()["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "16384")     # Enter takes the recommended one

    def test_a_longer_context_is_kept_when_chosen(self):
        for argv, answers in ((["--context", "32768"], None), ((), {"Context?": "5"})):
            with self.subTest(argv=argv, answers=answers):
                self.downloads, self.runs, self.verified = [], [], []
                code, out, _ = self.main(argv, family="mistral4", ram=127.9, vram=RTX_5080, answers=answers)
                self.assertEqual(code, 0, out)
                args = self.config()["args"]
                want = "32768" if argv else "131072"
                self.assertEqual(args[args.index("--max-context") + 1], want)
                self.assertEqual(args[args.index("--lookup") + 1], "0")

    def test_other_families_keep_their_menu_and_the_engine_s_guesses(self):
        code, out, _ = self.main(family="lfm25", ram=32.0, vram=8.0, answers={})
        self.assertEqual(code, 0, out)
        self.assertIn("2) 32K tokens   (recommended; 8 GB card: 69-74 tokens/s after a full prompt)", out)
        self.assertNotIn("16K tokens", out)
        self.assertNotIn("not verified", out)
        self.assertFalse([q for q in self.asked if "Cache bias" in q], self.asked)   # not asked
        self.assertNotIn("Cache bias", out)
        args = json.loads((self.t / "strata-lfm25-ud-q4_k_xl.json").read_text())["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "32768")
        for flag in ("--lookup", "--window", "--cache-bias"):
            self.assertNotIn(flag, args)

    # The cache bias (Petit Chonk 1.1.0; the engine's --cache-bias, src/poly/mistral4.cpp): off by default, 0.25 or
    # 0.5 when chosen, and then in the run config the server starts the engine with.

    def args(self):
        return self.config()["args"]

    def test_the_cache_bias_question(self):
        code, out, _ = self.main(family="mistral4", ram=127.9, vram=RTX_5080, answers={})
        self.assertEqual(code, 0, out)
        self.assertEqual(len([q for q in self.asked if q.startswith("Cache bias?")]), 1, self.asked)
        self.assertIn("Cache bias? [1]: ", self.asked)                        # off is the answer Enter takes
        for line in ("1) off    (recommended: the model's own pick, with which its accuracy was verified)",
                     "2) 0.25   (faster writing; no quality loss seen in a short test)",
                     "3) 0.5    (faster still; a small quality loss seen in a short test)"):
            self.assertIn(line, out)
        self.assertIn("docs/MISTRAL4.md", out)
        self.assertNotIn("--cache-bias", self.args())
        self.assertIn("cache bias: off", out)

    def test_a_cache_bias_chosen_is_written(self):
        for argv, answers, want in ((["--cache-bias", "0.5"], None, "0.5"), (["--cache-bias", "0.25"], None, "0.25"),
                                    ((), {"Cache bias?": "2"}, "0.25"), ((), {"Cache bias?": "3"}, "0.5")):
            with self.subTest(argv=argv, answers=answers):
                self.downloads, self.runs, self.verified = [], [], []
                for c in self.t.glob("strata-mistral4-*.json*"):                 # a new install each time
                    c.unlink()
                code, out, _ = self.main(argv, family="mistral4", ram=127.9, vram=RTX_5080, answers=answers)
                self.assertEqual(code, 0, out)
                args = self.args()
                self.assertEqual((args.count("--cache-bias"), args[args.index("--cache-bias") + 1]), (1, want))
                self.assertEqual(args[args.index("--lookup") + 1], "0")         # the family's own options stay
                self.assertEqual(args[args.index("--window") + 1], "1")
                self.assertIn(f"cache bias: {want} (--cache-bias in the config", out)
                if argv:                                                       # given: not asked
                    self.assertNotIn("Cache bias:", out)

    def test_the_earlier_choice_is_what_enter_and_yes_keep(self):
        code, out, _ = self.main(["--cache-bias", "0.5"], family="mistral4", ram=127.9, vram=RTX_5080)
        self.assertEqual(code, 0, out)
        for answers in (None, {}):                                             # --yes, then Enter at the question
            with self.subTest(answers=answers):
                code, out, _ = self.main(family="mistral4", ram=127.9, vram=RTX_5080, answers=answers)
                self.assertEqual(code, 0, out)
                args = self.args()
                self.assertEqual(args[args.index("--cache-bias") + 1], "0.5")
                if answers is not None:
                    self.assertIn("Cache bias? [3]: ", self.asked)
        code, out, _ = self.main(["--cache-bias", "off"], family="mistral4", ram=127.9, vram=RTX_5080)
        self.assertEqual(code, 0, out)
        self.assertNotIn("--cache-bias", self.args())                           # turned off again
        code, out, _ = self.main(family="mistral4", ram=127.9, vram=RTX_5080)
        self.assertEqual(code, 0, out)
        self.assertNotIn("--cache-bias", self.args())                           # and stays off

    def test_another_family_is_told_it_has_no_cache_bias(self):
        code, out, _ = self.main(["--cache-bias", "0.5"], family="lfm25", ram=32.0, vram=8.0)
        self.assertEqual(code, 0, out)
        self.assertIn("--cache-bias", next(line for line in out.splitlines() if "not used with" in line))
        args = json.loads((self.t / "strata-lfm25-ud-q4_k_xl.json").read_text())["args"]
        self.assertNotIn("--cache-bias", args)


if __name__ == "__main__":
    unittest.main()
