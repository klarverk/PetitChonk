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

Each script's own docstring gives its usage and exactly what it checks.

## What a result from these tools means

The tools compare the port with llama.cpp on the same model file and inputs. A pass means the port computes what
llama.cpp computes, within llama.cpp's own spread between legitimate settings, for that file on those inputs. That
is evidence about the port, not about the model's quality. It is also closed-loop self-evaluation by the people
who wrote the port, not independent verification.
