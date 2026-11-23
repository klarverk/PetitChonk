"""Tests for the name setup shows (Petit Chonk 1.1.0): the window of the start script it writes is titled Petit
Chonk, for every model, and its step 6 prepares the model "for Petit Chonk".

    python -m unittest tools.test_setup_names
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class StartScript(unittest.TestCase):

    def script(self, model):
        with tempfile.TemporaryDirectory() as t, mock.patch.object(setup, "ROOT", Path(t)), \
                mock.patch.object(setup, "WIN", True):
            path = setup.write_run_script(model, Path(t) / f"strata-{model.lower()}.json", 8080)
            # write_text's newline translation makes setup's "\r\n" into "\r\r\n" on Windows (as in 1.0.0); cmd reads
            # either, so the lines are compared without their CRs
            return path.name, path.read_bytes().decode("utf-8").replace("\r", "")

    def test_the_window_is_titled_petit_chonk(self):
        for model in ("mistral4-UD-Q4_K_M", "qwen36-IQ3_S", "glm-UD-IQ2_XXS"):
            with self.subTest(model=model):
                name, text = self.script(model)
                self.assertEqual(name, f"run-{model.lower()}.bat")
                lines = text.split("\n")
                self.assertEqual(lines[:2], ["@echo off", f"title Petit Chonk {model}"])
                self.assertNotIn("PolyStrata", text)

    def test_the_rest_of_the_script_is_unchanged(self):
        _, text = self.script("mistral4-UD-Q4_K_M")
        lines = text.split("\n")
        self.assertTrue(lines[2].startswith('cd /d "'))
        self.assertIn('"--engine" "strata"', lines[3])
        self.assertTrue(lines[3].endswith('"--port" "8080" "--open"'))
        self.assertEqual(lines[4:], ["if errorlevel 1 pause", ""])


class StepSix(unittest.TestCase):

    def test_step_six_names_petit_chonk(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertEqual(src.count('step(6, "preparing the model for Petit Chonk")'), 3)
        self.assertNotIn("preparing the model for PolyStrata", src)

    def test_the_web_app_names_the_window_petit_chonk(self):
        html = (ROOT / "serve" / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn("close Petit Chonk's window and start it again", html)
        self.assertNotIn("close PolyStrata's window", html)


if __name__ == "__main__":
    unittest.main()
