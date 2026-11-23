# Mistral Small 4 in Petit Chonk

Petit Chonk runs [Mistral Small 4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) on one NVIDIA
graphics card plus system RAM. The model is a 119B mixture of experts: 36 layers, each with 128 experts, of which
4 plus 1 shared expert work on each token. The `strata-poly` engine keeps as many experts as fit in a cache on the
card and the rest in RAM. Experts that do not fit in RAM either are read back from the model file when a token needs
them.

This page gives the details behind the [README](../README.md): what runs, what setup writes, how the port was
checked, and the accuracy and speed figures with their scope. It also covers [the cache bias](#the-cache-bias), an
option for faster writing: what it changes, and what was measured.

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
- No `--cache-bias` unless you choose one. Setup asks; off is the default (see [The cache bias](#the-cache-bias)).

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
- **1.0.0's ready-made engine** (the release asset, built for RTX 20–50) passed the same re-check: worst 0.429. On
  the CPU its outputs were byte-identical to the native build's in a spot check; on the GPU they differ slightly,
  as the re-check allows.
- **1.1.0's ready-made engine** passed the same re-check with the cache bias off: worst 0.383.
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

## The cache bias

New in 1.1.0, and **off by default**. It makes writing faster by letting the model prefer the experts that are
already on the graphics card. The cost is that the text is no longer exactly the model's own.

### What it changes

For every token, each of the 36 layers scores its 128 experts and uses the 4 with the highest scores, plus the
shared expert, which is always used. With a cache bias of λ:
- every expert already on the card gets λ added to its score before the 4 are picked;
- the 4 picked then work with the weights the model gives them from their own scores, as without the bias.

So the bias changes which experts are picked, not how much each one counts. More of the picks are then experts on
the card, fewer are read from RAM, and since writing speed is limited by reading experts from RAM, the model writes
faster. The text differs a little from what it writes without the bias.

### What it steers, and what not

The bias acts only when the engine computes fewer than 32 tokens at once:
- **Steered:** the tokens the model writes, one at a time. Also any prompt, or piece of a prompt, shorter than 32
  tokens, for example a short message read on top of a conversation the server has kept in its cache.
- **Not steered:** prompt pieces of 32 tokens or more, read with the model's own pick. So is the first token written
  right after such a piece.

In practice, the bias changes how the model writes, not how it reads a long prompt.

### Turning it on, checking it, turning it off

- **Setup asks** for Mistral Small 4 only: "Cache bias?", with `off` (recommended), `0.25` or `0.5`, the two values
  that were measured. Enter keeps the model's earlier choice; on a first install, that is off. To set it without
  the question: `SETUP.bat --family mistral4 --cache-bias 0.25` (or `0.5`, or `off`).
- **Where it is kept:** a value other than off goes into the model's run config (`strata-mistral4-ud-q4_k_m.json`)
  as `--cache-bias <λ>`. Later setup runs keep it until you choose `off`, which removes it.
- **How to see that it is on:** when the engine starts with it, it prints a line beginning
  `strata-poly: cache bias 0.25:`. With the bias off there is no such line.
- **When the engine refuses it:** it accepts the bias for Mistral Small 4 only when the file's expert weights are
  renormalized and unscaled, which is true of every file measured below. Otherwise it stops with
  "mistral4: --cache-bias needs expert_weights_norm and an expert_weights_scale of 1". Setup offers the bias for
  no other model.

**Why it is off by default:** every accuracy check on this page ([How it was verified](#how-it-was-verified),
[Accuracy benchmarks](#accuracy-benchmarks)) was made with the model's own pick. The bias has only the smaller
measurements below.

### Speed of the release engine (23 November 2026)

This is the engine in the 1.1.0 download, on the PC in [Speed](#speed), with `UD-Q4_K_M`. The test is the speed
test of the measurements below: an 8,192-token prompt, then 257 tokens written. The exploration build's runs from
22 November are listed beside it.

| Cache bias | Release engine, writing tokens/s | Mean | Speed against off | Exploration build, 22 November | Experts served from the card |
|---|---|---|---|---|---|
| off | 15.67 and 18.22 | 16.95 | 1.00 | 17.71 and 15.68 | 37% |
| 0.25 | 22.95 and 23.20 | 23.08 | **1.36** | 19.84 (1.19) | 55% |
| 0.5 | 34.40 and 33.49 | 33.95 | **2.00** | 28.93 (1.73) | 77% |

- **How the runs were set up:**
  - two runs per value, each a fresh engine start, in the order off, 0.25, 0.5, 0.5, 0.25, off, so that any drift
    over the runs falls evenly on each value;
  - each started from a fresh copy of the same warm expert profile, with a 16K context and
    `--lookup 0 --window 1`.
- **The rule, fixed before the runs:** if 0.25 or 0.5 had not been faster than off on the release engine, setup
  would no longer call them faster. Both are faster.
- **How precise it is:** the two runs with the bias off differ by 15%, so the figures are not precise to that.
- **Against the exploration build:**
  - both builds served the same share of experts from the card;
  - the release engine wrote faster at 0.25 and 0.5 than the exploration build's single runs did, and the cause is
    not established.
- **Speed only:** quality was not measured again on the release engine. The quality results below come from the
  exploration build, which has the same engine source.

### What was measured (22 November 2026)

These measurements ran on the PC in [Speed](#speed), with an exploration build of 1.1.0's engine source: 1.0.0's
source plus the cache-bias change, built natively, not the release engine (its speed is in the section above). One
run used 1.0.0's native build instead, as noted where it appears. Each test's rule was fixed before it ran.

Which experts are on the card depends on what the engine has read, so steered runs do not repeat byte for byte. The
tests compare by statistics. The speed figures are single runs.

**UD-Q4_K_M, the file setup installs:**

| Cache bias | Writing, tokens/s | Speed against off | Experts served from the card | Perplexity against llama.cpp (95% interval) |
|---|---|---|---|---|
| off | 17.71 and 15.68 | 1.00 | 37% | 0.996 (0.983–1.010) and 0.991 (0.976–1.004) |
| 0.25 | 19.84 | 1.19 | 55% | 0.999 (0.979–1.020) |
| 0.5 | 28.93 | 1.73 | 77% | 1.032 (1.005–1.063) |

- **Speed:** an 8,192-token prompt, then 257 tokens written.
  - One run per value, and two with the bias off. Those two differ by 12%, so a gain of that size is within one
    run's spread.
  - The ratios are against the mean of the two runs with the bias off (16.70 tokens/s).
- **Quality:** perplexity on 1,533 tokens of WikiText-2, from three of the benchmark's windows. Each window was a
  64-token prompt, then 512 tokens fed one at a time, so the bias acted on every scored token.
  - Each engine's tokens were paired one for one with llama.cpp `c7518dbe`'s on the same file.
  - The intervals are a bootstrap over 24 blocks of 64 tokens (10,000 resamples).
  - For scale: llama.cpp's own CPU path against its GPU path, on one of the three windows, gives 0.993
    (0.976–1.011).
- **How to read it:**
  - At 0.25 the ratio's interval contains 1.0: **no loss seen**, at this size.
  - At 0.5 the interval lies above 1.0: **a loss of about 3% in perplexity**.

  These are the descriptions setup gives. The test's own rule compared each value with the bias off, against a 2%
  bar, and called both "unclear": 1,533 tokens cannot show whether a loss is above 2%.
- **Not measured on this file:** task accuracy with the bias; longer texts; prompts other than these.

**UD-Q3_K_XL, measured but not offered** (setup installs UD-Q4_K_M only). This smaller file fits more experts on the
card, so the bias was measured further on it:
- **Speed** (the same test, against UD-Q4_K_M with the bias off): 1.47× off, 1.63× at 0.25 and 2.35× at 0.5
  (39.2 tokens/s).
- **The bias's own cost, a larger test:** 16 other WikiText-2 windows (`w02`, `w06`, … `w62`). Each was a 64-token
  prompt, then 1,024 tokens fed one at a time: 16,368 scored tokens.
  - At 0.5, against llama.cpp on the same file: 1.0071 (0.9946–1.0197).
  - The test asked whether the bias's loss is under 1%, and at this size it cannot say. The best estimate is +0.7%;
    the most the interval allows is about +2%.
  - The speed of those one-at-a-time tokens: 23.1 tokens/s off, 35.8 at 0.5 (1.55×).
- **The file's own cost:**
  - In perplexity, llama.cpp on UD-Q3_K_XL against llama.cpp on UD-Q4_K_M gives 1.054 (1.043–1.067), on the same
    16,368 tokens.
  - In task accuracy, on the multiple-choice benchmark above (900 items, the same harness and 1.0.0's native build,
    the bias off), it scored 66.1% against llama.cpp UD-Q4_K_M's 69.0%. That is −2.9 points (−4.9 to −0.9), McNemar p = 0.0073: a loss.
- **The bias on task accuracy:** the same 900 items, with every prompt read 16 tokens at a time so the bias acts on
  every token, which is harsher than real use.
  - At 0.5 it scored 67.2% against 66.8% with the bias off: +0.4 points (−1.6 to +2.5), p = 0.75.
  - So no change was seen at this size, though about 13% of the answers changed, in both directions.

**Four files at a glance** (the same speed and quality tests as UD-Q4_K_M above). Here quality is against llama.cpp
on UD-Q4_K_M, so it combines the file's own loss with the bias's:

| File | Speed, off / 0.25 / 0.5 (against UD-Q4_K_M off) | Perplexity, off / 0.25 / 0.5 (against llama.cpp UD-Q4_K_M) |
|---|---|---|
| Unsloth UD-Q4_K_M | 1.00 / 1.19 / 1.73 | 0.996 / 0.999 / 1.032 |
| bartowski Q3_K_M | 1.51 / 2.04 / 2.66 | 1.031 / 1.071 / 1.071 |
| Unsloth UD-Q3_K_XL | 1.47 / 1.63 / 2.35 | 1.044 / 1.047 / 1.053 |
| Unsloth UD-Q2_K_XL | 1.68 / 2.30 / 2.82 | 1.243 / 1.242 / 1.268 |

- **The engine is not the source of these losses.** With the bias off, the engine matched llama.cpp on each file
  within the interval: UD-Q4_K_M 0.996, Q3_K_M 0.997, UD-Q3_K_XL 0.995, UD-Q2_K_XL 1.015 (0.994–1.036). That is on
  1,533 tokens, which is not a verification of those files.
- **Each interval spans several percent** at this size (about 3 to 6%, more for UD-Q2_K_XL). A difference of a few
  percent between two rows is therefore not established.
- **The Unsloth files** are revision `bd93c721`, and bartowski's Q3_K_M is revision `f569d04e`.

### How the measurements were made

- **The engine:** the exploration build above, with CUDA 13.0, run with `--lookup 0 --window 1` as setup writes it.
  - The speed and perplexity runs used a 16K context. Each started from a fresh copy of the same warm expert
    profile: the one learned by the 8,192-token runs behind this page's speed figures.
  - The multiple-choice runs used the benchmark harness's own settings.
- **The scripts** are in [`tools/mistral4/cache_bias/`](../tools/mistral4/cache_bias/). Each one's docstring says
  exactly what it does.
  - `qd_smoke.py`: the four-file test (speed and quality).
  - `qd_vs_ref.py` and `qd_split.py`: the pairing with llama.cpp, and the split into the file's loss and the bias's.
  - `qd_big.py` and `qd_big_vs_q4.py`: the larger UD-Q3_K_XL test.
  - `b3_steered.py`: the multiple choice with every token steered.
- **llama.cpp's side** is the benchmark's `bench/p4_ref.py`. The multiple-choice runs use the benchmark harness
  (`bench/p4_cand.py`, `bench/p4_score.py`) unchanged, with only the model, or the reading size and the bias,
  changed.
- **The rules,** fixed before each run:
  - the four-file test called a variant fast at 1.20× UD-Q4_K_M or more, and labelled quality "no loss seen"
    (interval's upper end at most 1.02), "loss seen" (lower end above 1.02) or "unclear";
  - the larger test had its 1% bar;
  - the multiple choice used McNemar's exact test at 5%.
- **What these results are:** benchmarked speed from single runs, and a low-confidence indication of quality: one
  kind of text, perplexity rather than task accuracy except where stated, and small samples. It is closed-loop
  self-evaluation by the people who wrote the change, not independent verification.

## Reproducing the checks

The tools that produced these results are in [`tools/mistral4/`](../tools/mistral4/), with what each does and what
it needs. The model file, the data and llama.cpp are fetched separately.
