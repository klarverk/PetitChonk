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

## [1.1.0] - 2026-11-23

Upstream base unchanged: PolyStrata `v0.1.45-poly.10`,
[`3e1642161198c50ff1abed4046f4e4f476099dbb`](https://github.com/simplexai-labs/PolyStrata/commit/3e1642161198c50ff1abed4046f4e4f476099dbb).

### Added

- The cache bias for Mistral Small 4 (`--cache-bias`), off by default. While the model writes, each expert already
  on the card counts λ more when a token picks its experts; the picked experts keep the router's own weights. It
  acts only on calls of fewer than 32 tokens, so long prompts are read as before.
  - `strata-poly` accepts `--cache-bias` for `mistral4` files whose expert weights are renormalized and unscaled,
    and refuses it for others.
  - Setup asks for Mistral Small 4 only: off, 0.25 or 0.5. A value other than off goes into the run config and
    later setup runs keep it. The setup option `--cache-bias off|0.25|0.5` sets it without asking.
- `docs/MISTRAL4.md`: what the cache bias changes and what it doesn't, the measurements of 22 November 2026 with
  their scope, and how they were made. Also the release engine's own speed with the bias (23 November), and its
  accuracy re-check with the bias off.
- The scripts behind those measurements, in `tools/mistral4/cache_bias/`, with their sources in
  `tools/mistral4/SOURCES.md`.
- The README's "Built with" list: PolyStrata `v0.1.45-poly.10`, ggml from llama.cpp `3cf03257`, CUDA 13.0 and
  Python 3.10 or newer.
- The release zip carries ggml's and PolyStrata's licence notices.

### Changed

- Setup's first line names Petit Chonk and Mistral Small 4, in place of PolyStrata's line about its own models.
- The start script that setup writes (`run-<model>.bat`) titles its window Petit Chonk, and setup's step 6 says it
  prepares the model for Petit Chonk. 1.0.0 renamed the other window titles; these two still said PolyStrata. The
  web app's settings note now says to close Petit Chonk's window.

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

[Unreleased]: https://github.com/klarverk/PetitChonk/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/klarverk/PetitChonk/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/klarverk/PetitChonk/compare/3e1642161198c50ff1abed4046f4e4f476099dbb...v1.0.0
