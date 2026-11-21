# Changelog

All notable changes to Petit Chonk are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Petit Chonk's version numbers are its own. They are independent of PolyStrata's and the Strata project's, and of the engine
version that `strata-poly` reports. Each release names the PolyStrata commit it is built on. The public interface
that the version numbers describe is:
- the installer's command-line options;
- the server's HTTP API;
- the run-config format;
- `strata-poly`'s command-line options.

## [Unreleased]

### Changed

- Setup's first line names Petit Chonk and Mistral Small 4, in place of PolyStrata's line about its own models.

## [1.0.0] - 2026-11-21

The first release. Upstream base: PolyStrata
[`3e1642161198c50ff1abed4046f4e4f476099dbb`](https://github.com/simplexai-labs/PolyStrata/commit/3e1642161198c50ff1abed4046f4e4f476099dbb).

### Added

- Native text inference for Mistral Small 4 (`mistralai/Mistral-Small-4-119B-2603`) in the `strata-poly` engine:
  a `mistral4` backend chosen from the GGUF's `general.architecture`. It keeps a compressed (MLA latent) attention
  cache in FP16 and uses PolyStrata's expert cache and CPU/GPU execution. Supported and tested on NVIDIA cards
  (CUDA), on Windows 11.
- Mistral Small 4's Tekken pre-tokenizer in `tools/strata_tokenizer.py`. A `mistral4` tokenizer refuses any other
  pre-tokenizer instead of falling back to Qwen's.
- Chat serving for Mistral Small 4. Its chat template gets the `strftime_now` date helper; its reasoning
  (`[THINK]`…`[/THINK]`) is returned apart from the answer; and a request's reasoning effort maps to the two its
  template takes: `none`, or `high` for any level of thinking. Other models' prompts and parsing are unchanged.
- Setup installs Mistral Small 4 (`--family mistral4`): Unsloth's UD-Q4_K_M, pinned and checked by SHA-256. Its run
  config takes one token a step (`--lookup 0 --window 1`) and recommends a 16K context, the longest its parity with
  llama.cpp was verified at; longer contexts are offered as not verified. Setup offers it on NVIDIA cards only.
- `strata-poly` options `--tokens-file` and `--follow-file`: the ids of `--tokens` and `--follow`, read from a
  file.
- An opt-in logits dump for verification runs (`STRATA_LOGITS_DIR`). It is off unless the variable is set.
- The verification tools in `tools/mistral4/`: the full-model check against llama.cpp, with the thresholds the
  verdict used, the accuracy benchmark harness, and the speed runner.
- `docs/MISTRAL4.md`: what runs, what setup writes, how the port was verified, and the accuracy and speed figures
  with their scope.
- A Sponsor button (`.github/FUNDING.yml`, Patreon).
- The Petit Chonk logos, dark and light (`docs/media/`).

### Changed

- The name in the web app and the window titles: PolyStrata becomes Petit Chonk.
- Setup downloads its ready-made `strata-poly` engine from Petit Chonk's releases. A checkout with no matching
  release asset compiles the engine, as before.
- The README describes Petit Chonk: what is supported and needed, the install, and its accuracy and speed against
  llama.cpp. It credits PolyStrata, [Strata](https://github.com/Niko1221/Strata), llama.cpp, Mistral AI and
  Unsloth.
- `LICENSE` adds Petit Chonk's copyright line beside the Strata project's and PolyStrata's.
- `SECURITY.md` sends vulnerability reports to this repository.

[Unreleased]: https://github.com/klarverk/PetitChonk/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/klarverk/PetitChonk/compare/3e1642161198c50ff1abed4046f4e4f476099dbb...v1.0.0
