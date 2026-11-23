# Mistral Small 4 verification tools

These are the tools that checked Petit Chonk's Mistral Small 4 port against llama.cpp, and that ran its accuracy
benchmarks. They are copies of the tools that produced the results, changed only where they named folders on the
machine they ran on. [SOURCES.md](SOURCES.md) lists each file's original with both SHA-256 values.

## What is here

- **`fti/`: the full-model check.** The port's output distributions are compared with llama.cpp's on the same model
  file, under a contract fixed before any of the port's outputs existed.
  - `make_sequences.py` builds the 7 test sequences and checks the tokenizer against `mistral-common`.
  - `ref_logits.cpp` and `build_ref_logits.bat` are llama.cpp's side: they write full logits rows for a token
    sequence.
  - `t4_runs.py` runs the reference in its trusted variants: GPU, chunked prompts, flash attention off, unfused
    operations, CPU, and token-by-token decoding.
  - `t6_measure.py`, `t6_contract_eval.py` and `t6_rule.py` are the contract as code. They derive thresholds from
    the reference variants (`thresholds`), and judge a run set (`judge-run-set`).
  - `thresholds_UD-Q4_K_M.json` holds the thresholds the 1.0.0 verdict used, for Unsloth's `UD-Q4_K_M` file.
  - `t8_run.py` runs the port (`strata-poly`) on the sequences in each run set.
- **`bench/`: the accuracy benchmarks.** Both engines score the same token ids.
  - `fetch_sources.py` fetches WikiText-2, ARC-Challenge, CommonsenseQA and MMLU from Hugging Face at pinned
    revisions, and checks their SHA-256. The data is not stored in this repository.
  - `fetch_mmlu_yaml.py` fetches lm-evaluation-harness's task files at `v0.4.13`.
  - `freeze_a.py` cuts and draws the windows and items with fixed seeds and tokenizes them. `check_render.py`
    re-renders every item from lm-evaluation-harness's own task files and compares.
  - `p4_cand.py` scores sequences with the port, in serve mode. It checks that no prompt is reused, that every row
    maps to its request, and it deletes the large dumps as it goes.
  - `ref_logprobs.cpp`, `build_ref_logprobs.bat` and `p4_ref.py` are llama.cpp's side, with the same settings as
    the full-model check's GPU and CPU variants.
  - `p4_score.py` computes the measures: paired perplexity with a bootstrap interval, and task accuracy with
    agreement and McNemar's test.
  - `p4_qual.py` and `p4_qual_runs.py` qualify the harness on items outside the benchmark. `p5_runs.py` runs the
    benchmark steps in order.
  - `test/test_p4_cand.py` tests the port's scorer against a stub engine, with no model needed.
- **`speed/`: the speed measurements** (the README's speed figures).
  - `p4b.py` runs them:
    - `text` builds the prompts from the benchmark's WikiText windows;
    - `b4port` times the port on its default user path, the run config setup writes, through the server's
      protocol, a cold request and three warm ones;
    - `b4llama` times `llama-bench` at the same points;
    - `b5` times longer contexts;
    - `b6` and `parity` emulate a PC with less RAM.

    Every run records the machine's state, VRAM and available RAM each second, and the drive's reads.
  - `ballast.py` locks a given amount of RAM until told to release it. It is the less-RAM emulation, and needs no
    administrator rights.
- **`cache_bias/`: the cache-bias measurements** of 22 November 2026 (docs/MISTRAL4.md,
  [The cache bias](../../docs/MISTRAL4.md#the-cache-bias)). They need an engine that takes `--cache-bias` for
  Mistral Small 4, as 1.1.0's does.
  - `qd_smoke.py` is the four-file test. Per file, it runs:
    - the speed runs: the 8,192-token prompt, then 257 tokens written, at λ 0, 0.25 and 0.5;
    - the quality runs: three WikiText-2 windows, each a 64-token prompt and then 512 tokens fed one at a time.

    Each quality run's logits are reduced to per-token values and deleted. Its rule file (`RULE.md`) must be in
    the output folder before it runs.
  - `qd_vs_ref.py` pairs those per-token values with llama.cpp's reference for UD-Q4_K_M, the benchmark's
    `b2_ref_gpu` output.
  - `qd_split.py` uses llama.cpp's reference on each file (`bench/p4_ref.py`, variant `gpu`). It splits each
    ratio into the file's own loss and the bias's.
  - `qd_big.py` is the larger UD-Q3_K_XL test: 16 windows, 1,024 tokens each, at λ 0 and 0.5. It expects
    llama.cpp's reference on that file in the output folder first. `qd_big_vs_q4.py` sets the result beside
    llama.cpp's UD-Q4_K_M.
  - `b3_steered.py` runs the multiple-choice harness (`bench/p4_cand.py`, imported unchanged). It reads every
    prompt a few tokens at a time, so the bias acts on every token, and adds `--cache-bias`.

## Data and licences

The benchmarks use these datasets. They are fetched at pinned revisions by `fetch_sources.py`, not stored in this
repository, and keep their own licences:

| Benchmark | Dataset (Hugging Face) | Licence |
|---|---|---|
| Perplexity | `Salesforce/wikitext`, `wikitext-2-raw-v1`, test split | CC BY-SA 3.0 and GFDL |
| ARC-Challenge | `allenai/ai2_arc`, `ARC-Challenge`, test split | CC BY-SA 4.0 |
| CommonsenseQA | `tau/commonsense_qa`, validation split | MIT (the authors' statement) |
| MMLU | `cais/mmlu`, `all`, test split | MIT |

The prompt formats are those of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
`v0.4.13` (MIT), whose task files `fetch_mmlu_yaml.py` fetches.

## Setting up

- **Windows, an NVIDIA card, CUDA 13.0.** Never use CUDA 13.2.x: its compiler gives wrong results on RTX 50 cards.
  The scripts use CUDA 13.0 at its default install path; `PETITCHONK_CUDA` points them elsewhere.
- **llama.cpp at `c7518dbe`,** built for CUDA with shared libraries. Set `REF` to its folder before running
  `build_ref_logits.bat` or `build_ref_logprobs.bat`.
- **Python:**
  - `mistral-common` 1.13.1, `numpy` and `PyYAML` for most scripts;
  - `jinja2` for `check_render.py`;
  - Run each script with `python -I`.
- **A work folder,** named by `PETITCHONK_WORK` (default: the current folder), laid out as the scripts expect:
  - `models\unsloth-Mistral-Small-4-119B-2603-GGUF@bd93c721\UD-Q4_K_M\…` holds the model;
  - `ref\llama.cpp-c7518dbe\build\bin` holds llama.cpp's binaries;
  - `fti\sources` holds the texts `make_sequences.py` reads. They are not included here, so the 7 sequences can't
    be rebuilt from this repository alone;
  - `fti\sequences`, `fti\ref` and `pub\…` are where the tools write.
- **The port:** `PETITCHONK_BINARY` names the `strata-poly` executable to test.
- **`p5_runs.py`** checks a manifest of the frozen benchmark inputs before it runs. The one built in is Petit Chonk
  1.0.0's, which lists that machine's folders. For your own run, build the inputs with `freeze_a.py`, write a
  manifest of them, and set `PETITCHONK_P4A_MANIFEST_SHA256`.
- **`speed/p4b.py`** reads setup's run config for the model: `PETITCHONK_RUN_CONFIG`, by default
  `cc\strata-mistral4-ud-q4_k_m.json` in the work folder. It refuses an engine other than the 1.0.0 release's
  unless `PETITCHONK_ENGINE_SHA256` names it.

- **`cache_bias/`** reads the same work folder:
  - `PETITCHONK_BINARY` names the engine (default: the exploration build's `steer\build-steer\strata-poly.exe`);
  - the model files are under `models\…`, as `qd_smoke.py` lists them;
  - the benchmark's frozen inputs are `pub\p4\freeze_a\b2_windows.jsonl` and `pub\p4\freeze_b\prompts.json`;
  - the warm expert profile is `pub\p6\run1\b4port\len_8192\expert-profile.txt`;
  - llama.cpp's reference outputs are `pub\p5\run1\b2_ref_gpu` and `pub\explore\qd1_ref\<file>`;
  - the larger test's 16 windows are in `pub\explore\qd2_windows.jsonl`: the benchmark's `w02`, `w06`, … `w62`,
    one per line, as in `b2_windows.jsonl`.

Each script's own docstring gives its usage and exactly what it checks.

## What a result from these tools means

The tools compare the port with llama.cpp on the same model file and inputs. A pass means the port computes what
llama.cpp computes, within llama.cpp's own spread between legitimate settings, for that file on those inputs. That
is evidence about the port, not about the model's quality. It is also closed-loop self-evaluation by the people
who wrote the port, not independent verification.
