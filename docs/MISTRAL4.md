# Mistral Small 4 in Petit Chonk

Petit Chonk runs [Mistral Small 4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) on one NVIDIA
graphics card plus system RAM. The model is a 119B mixture of experts: 36 layers, each with 128 experts, of which
4 plus 1 shared expert work on each token. The `strata-poly` engine keeps as many experts as fit in a cache on the
card and the rest in RAM. Experts that do not fit in RAM either are read back from the model file when a token needs
them.

This page gives the details behind the [README](../README.md): what runs, what setup writes, how the port was
checked, and the accuracy and speed figures with their scope.

## What runs

- **Text only.** The model's image input is not supported.
- **One model file is verified:** Unsloth's `UD-Q4_K_M` quantization, revision `bd93c721…` (3 files, 73.8 GB). Its
  routed experts are 70.6 GB (Q4_K, Q5_K and Q6_K).
- **NVIDIA cards (CUDA).** Tested on Windows 11 with an RTX 5080 (16 GB) and 128 GB of RAM. The AMD (HIP), Intel
  (SYCL) and Apple (Metal) backends are neither built nor tested, and setup does not offer the model there.
- **The attention cache** is the model's compressed (latent) form, in FP16: 23,040 bytes per token of context
  (36 layers × 320 values × 2 bytes).
- The engine picks this code from the GGUF's `general.architecture`, `mistral4`. The tokenizer is Mistral's Tekken
  (the GGUF names its pre-tokenizer `pixtral`). It refuses any other pre-tokenizer for this architecture.

## What setup writes

`SETUP.bat --family mistral4` (or the menu) writes a run config with:
- `--max-context 16384`. The menu recommends 16K, the longest context the port was checked at. 32K, 64K and 128K
  are offered as "not verified for this model".
- `--lookup 0 --window 1`: one token a step. The engine's default guesses ahead, which was not checked for this
  model.
- An expert profile in the model's pack. The engine counts which experts are used, and keeps the most-used ones on
  the card at the next start.
- Prompts read 1,024 tokens at a time.

**Its estimates for this model** (setup's `--check` and its warnings):
- about 5.8 GiB of VRAM before any expert at a 32K context (measured on the RTX 5080);
- so about 10.9 GB of the experts fit a 16 GB card;
- about 70 GB of RAM beside a 16 GB card. Below that, the experts that do not fit come from the drive while it
  answers (see [Speed](#speed)).

## Thinking

Mistral Small 4 answers directly or reasons first. Its chat template knows two settings, `none` and `high`:
- no thinking, or none asked for, gives `none`;
- any level of thinking a client asks for (low, medium, high) gives `high`.

The reasoning (`[THINK]`…`[/THINK]`) comes back apart from the answer, as other models' reasoning does. The
template's date helper (`strftime_now`) is provided for this model only.

## Not supported yet

- **Tool calls:** Mistral's tool-call format is not read, so a tool call arrives as plain text.
- **Image input.**
- **Quantizations other than `UD-Q4_K_M`:** they may run, but they are not verified.
- **Guessing ahead** (`--lookup`, `--window`), several requests at once, and sampling were not checked.

## How it was verified

The port's outputs were compared with llama.cpp's on the same model file. The acceptance rules were fixed before
any output of the port existed.

- **The reference:** llama.cpp at commit `c7518dbe`, on the same GGUF. The thresholds come from its own spread
  between legitimate settings: chunked prompts, flash attention off, unfused operations, the CPU instead of the
  GPU, and token-by-token decoding.
- **The inputs:** 7 fixed token sequences (English and Swedish prose, C++ and Python code, mathematics, a chat, a
  long document), 26,016 positions, the longest 14,743 tokens.
  - The tokenizer gives the same token ids as Mistral's own (`mistral-common`) on all 7.
  - Separately, it was checked on 44 million characters.
- **The rule:** a frozen acceptance contract (v3) judged every position's full output distribution, not only the
  top token.
- **The result:** `PASS` in all 10 run sets (70 runs). They cover whole, chunked and split prompts and
  token-by-token decoding, with the experts on the card or on the CPU, and CUDA graphs on or off. The worst
  statistic reached 0.468 of its threshold.
- **On the release source,** after the merge with current PolyStrata, a re-check of 3 run sets under the same
  frozen rules passed: worst 0.388.
- **The ready-made engine** (the release asset, built for RTX 20–50) passed the same re-check: worst 0.429. On the
  CPU its outputs were byte-identical to the native build's in a spot check; on the GPU they differ slightly, as
  the re-check allows.
- **What this shows:** the port computes what llama.cpp computes, within llama.cpp's own spread, for this model
  file on these inputs, up to 16K tokens. It is closed-loop self-evaluation, not independent verification. Other
  quantizations, cards, operating systems, sampling and longer contexts are not covered.

## Accuracy benchmarks

Both engines scored the same token ids from the same `UD-Q4_K_M` file: Petit Chonk's native build of the release
source, and llama.cpp `c7518dbe`. The windows, items, formats and scoring were frozen before either engine ran.

**WikiText-2 perplexity** (the test text, 73 windows of 4,096 tokens, 298,935 scored tokens):

| | Petit Chonk | llama.cpp | Ratio (95% interval) |
|---|---|---|---|
| All positions | 5.8893 | 5.8644 | 1.0042 (0.9987–1.0100) |
| Second half of each window | 5.7430 | 5.7103 | 1.0057 (0.9963–1.0159) |

- The intervals are a bootstrap over windows (10,000 resamples).
- **For scale:** llama.cpp's own GPU and CPU paths, on 8 of the windows, give a ratio of 0.9974 (0.9822–1.0097).
- **The per-token differences** between the two engines (median 0.077, 99th percentile 1.58 nats) are like
  llama.cpp's GPU against its CPU (0.089 and 1.74).
- **One token is an exception.** At window 36, position 1,706, Petit Chonk's log-probability for `' War'` is −16.88,
  llama.cpp's −0.50.
  - It is reproducible.
  - llama.cpp's own CPU path gives −10.34 at that token.
  - Its cause is not established.

**Multiple choice** (300 items per task, drawn with seed 20261119, scored by log-likelihood in
lm-evaluation-harness's formats at `v0.4.13`):

| Task | Petit Chonk | llama.cpp |
|---|---|---|
| ARC-Challenge | 62.7% (188) | 62.3% (187) |
| CommonsenseQA | 67.7% (203) | 68.0% (204) |
| MMLU | 76.3% (229) | 76.7% (230) |
| All 900 | 68.9% (620) | 69.0% (621) |

- The engines picked the same answer on 93.9% of the items.
- McNemar's test on the items where they differ: p = 1.0 for every task and pooled.
- ARC's length-normalized accuracy is 65.3% for both.
- These are parity measures: they show the two engines score the model the same way, not how good the model is.
- The datasets and their licences are listed in [`tools/mistral4/README.md`](../tools/mistral4/README.md#data-and-licences).

## Speed

Writing speed is close to llama.cpp's: about the same on the first answer, 20–50% faster once its expert cache has
warmed up. Reading long prompts is where it gains most, about 3×. Writing speed depends mostly on RAM bandwidth and
VRAM: more of either helps.

One PC: RTX 5080 16 GB, i9-10850K, 128 GB of DDR4-3200 (XMP on), Windows 11, the `UD-Q4_K_M` file. Single runs,
21 November 2026.
- **Petit Chonk:** the release engine with the run config setup writes. Each prompt length is one engine start: a
  cold request (just started, with a fresh expert profile, so the cache holds experts chosen without knowing which
  are used most), then three warm ones on other text.
- **llama.cpp:** `llama-bench` at `c7518dbe`, the experts of layers 0–3 on the card (its best placement there),
  flash attention on, 10 threads, one repetition.
- Tokens per second. "Writing" is 128 tokens after the prompt (for llama.cpp, after a context of that depth).

| Prompt | Reading, Petit Chonk cold | warm | llama.cpp | Writing, Petit Chonk cold | warm | llama.cpp |
|---|---|---|---|---|---|---|
| 1,024 | 99.4 | 147.0, 194.2, 198.1 | 103.0 | 13.4 | 16.1, 18.3, 16.7 | 13.5 |
| 8,192 | 181.7 | 379.9, 378.9, 381.1 | 116.8 | 15.6 | 15.9, 20.2, 16.1 | 13.3 |
| 16,000 | 67.6 | 290.0, 288.5, 289.9 | 113.0 | 16.6 | 18.7, 17.2, 18.4 | 13.5 |

**Longer contexts** (they run; their outputs were not checked):

| Context | Prompt | Read in | Reading | Writing |
|---|---|---|---|---|
| 32K | 32,512 tokens | 173 s | 187.8 | 13.3 |
| 64K | 65,280 tokens | 751 s | 86.9 | 12.6 |

**With 64 GB of RAM:** the same PC with 64 GiB locked away.
- The engine kept 28.4 GiB of the experts outside VRAM in RAM, and read the rest from the drive as tokens needed
  them (556 GB over the run).
- With 8,192-token prompts it wrote 3.1 (cold), then 3.9, 3.8 and 3.9 tokens/s, and read prompts at 222 (cold),
  then about 320 tokens/s.

**Not measured:** other cards, CPUs, RAM speeds and PCs; contexts above 64K; PCs with less than 64 GB; several
requests at once; a profile that has learned from use.

## Reproducing the checks

The tools that produced these results are in [`tools/mistral4/`](../tools/mistral4/), with what each does and what
it needs. The model file, the data and llama.cpp are fetched separately.
