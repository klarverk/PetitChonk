"""Tests for the ready-made engine's zip (tools/make_poly_release.py, Petit Chonk 1.1.0): beside the two programs and
BUILD.json it carries the licence notices of the code in them, the repository's LICENSE and ggml's, byte for byte;
and setup installs such a zip, notices and all.  Nothing is compiled: the build steps are stand-ins.

    python -m unittest tools.test_make_poly_release
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
import make_poly_release as MPR  # noqa: E402

SRC = "0123456789abcdef"
GPU = {"arch": 89, "driver": "581.57"}


def built(src, bdir, target, defs, vcvars, bat_name):
    """cmake_build's stand-in: the program it would have built, holding its target's name."""
    exe = bdir / setup.POLY_EXE if target == "strata-poly" else bdir / "bin" / setup.VEXE
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(target.encode())


def make_release(out: Path) -> Path:
    """make_poly_release's main() into `out`, with the toolchain, the clean-source check and the builds stood in."""
    clean = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    with mock.patch.object(sys, "argv", ["make_poly_release.py", str(out)]), \
            mock.patch.object(setup, "WIN", True), \
            mock.patch.object(MPR.subprocess, "run", return_value=clean), \
            mock.patch.object(setup, "find_nvcc", lambda *a, **k: ("nvcc.exe", (13, 0))), \
            mock.patch.object(setup, "find_vcvars", lambda *a, **k: "vcvars64.bat"), \
            mock.patch.object(setup, "get_llama_cpp", lambda vision=False: out / "llama.cpp"), \
            mock.patch.object(setup, "cmake_build", built), \
            mock.patch.object(setup, "source_hash", lambda parts: SRC), \
            mock.patch.object(setup, "source_version", lambda: "0.1.45"), \
            mock.patch("builtins.print"):
        if MPR.main() != 0:
            raise AssertionError("make_poly_release failed")
    return out / setup.POLY_ASSET.format(src=SRC)


class ReleaseZip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.zip = make_release(Path(self.tmp.name) / "out")

    def test_the_zip_carries_the_two_notices_beside_the_programs(self):
        with zipfile.ZipFile(self.zip) as f:
            self.assertEqual(sorted(f.namelist()), sorted([setup.POLY_EXE, setup.VEXE, "BUILD.json", "LICENSE.txt",
                                                           "LICENSE-ggml.txt"]))

    def test_the_notices_are_the_repository_files_byte_for_byte(self):
        with zipfile.ZipFile(self.zip) as f:
            self.assertEqual(f.read("LICENSE.txt"), (ROOT / "LICENSE").read_bytes())
            self.assertEqual(f.read("LICENSE-ggml.txt"), (ROOT / "third_party" / "ggml" / "LICENSE").read_bytes())

    def test_the_notices_name_their_holders(self):
        with zipfile.ZipFile(self.zip) as f:
            ours, ggml = f.read("LICENSE.txt").decode(), f.read("LICENSE-ggml.txt").decode()
        for holder in ("Niko1221", "the PolyStrata contributors", "the Petit Chonk contributors"):
            self.assertIn(holder, ours)
        self.assertIn("The ggml authors", ggml)
        for text in (ours, ggml):
            self.assertIn("The above copyright notice and this permission notice shall be included", text)

    def test_the_programs_and_build_json_are_as_before(self):
        with zipfile.ZipFile(self.zip) as f:
            self.assertEqual(f.read(setup.POLY_EXE), b"strata-poly")
            self.assertEqual(f.read(setup.VEXE), b"strata-vision")
            meta = json.loads(f.read("BUILD.json"))
        self.assertEqual((meta["source"], meta["src"], meta["archs"], meta["cuda"]),
                         ("release", SRC, MPR.ARCHS, "13.0"))

    def test_setup_installs_the_zip_with_its_notices(self):
        repo = Path(self.tmp.name) / "repo"
        repo.mkdir()
        eng = repo / setup.ENGINE_POLY_DIR
        with mock.patch.object(setup, "ROOT", repo), mock.patch.object(setup, "WIN", True), \
                mock.patch.object(setup, "POLY_PREBUILT_URL", str(self.zip.parent)), \
                mock.patch.object(setup, "source_hash", lambda parts: SRC), \
                mock.patch.object(setup, "cpu_info", lambda: ("an AVX2 CPU", True, False)), \
                mock.patch.object(setup, "pip_cuda_libs", lambda toolkit=13: None):
            self.assertTrue(setup.get_poly_prebuilt(GPU))
        self.assertEqual((eng / setup.POLY_EXE).read_bytes(), b"strata-poly")
        self.assertEqual((eng / "LICENSE.txt").read_bytes(), (ROOT / "LICENSE").read_bytes())
        self.assertEqual((eng / "LICENSE-ggml.txt").read_bytes(),
                         (ROOT / "third_party" / "ggml" / "LICENSE").read_bytes())


if __name__ == "__main__":
    unittest.main()
