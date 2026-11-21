<h1 align="center">Petit Chonk</h1>

<p align="center"><b>Mistral Small 4 on one 16 GB graphics card and system RAM</b><br>
A fork of <a href="https://github.com/simplexai-labs/PolyStrata">PolyStrata</a></p>

Petit Chonk brings [Mistral Small 4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) (119B parameters,
6.5B active per token) to PolyStrata's mixture-of-experts engine, `strata-poly`. The experts used most sit in the
graphics card's memory and the rest in system RAM, and when RAM runs short they are read back from the drive. A
100B-class model then runs on one 16 GB card.

**Why not Qwen?** If Qwen fits your needs, use it: PolyStrata runs it well. Petit Chonk exists so that a European
open-weight model has a place in the same engine. Mistral Small 4 comes from Mistral AI in Paris under the Apache 2.0
licence, with reasoning you switch on per request. Local AI is more robust when it doesn't depend on one region's
models.

**Verified, not just running.** Petit Chonk's outputs were compared with llama.cpp's on the same model file, under
acceptance rules fixed before any of its results existed. See [How it was checked](#how-it-was-checked).

## What is supported

- **One model:** Mistral Small 4, text only, from Unsloth's `UD-Q4_K_M` file
  ([`unsloth/Mistral-Small-4-119B-2603-GGUF`](https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF),
  revision `bd93c721`, 73.8 GB). Other quantizations may run, but are not verified.
- **NVIDIA graphics cards (CUDA).** Tested on Windows 11 with an RTX 5080 (16 GB) and 128 GB of RAM. The
  ready-made engine is built for RTX 20 to RTX 50 cards; only the RTX 5080 has been tried.
- **PolyStrata's other models** are still in the code, but this fork does not test them. For those, use
  [PolyStrata](https://github.com/simplexai-labs/PolyStrata).
- **The AMD (HIP), Intel (SYCL) and Apple (Metal) backends are neither built nor tested here**, and setup does not
  offer Mistral Small 4 on them. If they matter to you, sponsorship is welcome and would make that work possible:
  [Patreon](https://patreon.com/Patternwright).

## What you need

- An NVIDIA card with 16 GB of VRAM. Setup's estimate for a smaller card is that more of every answer runs on
  the CPU, so it is slower (not measured).
- **RAM:** about 70 GB beside a 16 GB card, by setup's estimate. With 64 GB it still runs, but its experts come
  from the drive while it writes: about 4 tokens/s instead of 16–20 (measured, see [Speed](#speed)).
- About 75 GB of disk for the model file, on a fast SSD.
- Windows 11 (tested). Linux is not tested.

Writing speed depends mostly on RAM bandwidth and VRAM: more of either helps.

## Install

On Windows, run `START-HERE.bat` (or `SETUP.bat`) and pick Mistral Small 4, or:

```
SETUP.bat --family mistral4
```

Setup downloads the ready-made `strata-poly` engine from this repository's releases, the model file (pinned and
checked by SHA-256) and the Python packages, and writes a run config. A model file you already have can be used in
place with `--gguf-dir <its folder>`.

The run config setup writes is the one that was verified:
- **16K tokens of context** (recommended). 32K, 64K and 128K are offered, marked "not verified for this model".
- **One token a step** (`--lookup 0 --window 1`): the engine's guessing ahead is off for this model.

The server then answers at `http://127.0.0.1:8080/v1` (OpenAI) and `/v1/messages` (Anthropic), with a web app at
`http://127.0.0.1:8080/`.

## Thinking

Mistral Small 4 answers directly or reasons first. Its chat template has two settings, `none` and `high`:
- no thinking, or none asked for, gives `none`;
- any level a client asks for (low, medium, high) gives `high`.

The reasoning comes back apart from the answer, as other models' reasoning does.

## Not supported yet

- **Tool calls:** Mistral's tool-call format is not read, so a tool call arrives as plain text.
- **Images:** the model's image input is not supported.
- **Guessing ahead** (speculative decoding) is off for this model.
- **Contexts above 16K** run, but their outputs have not been checked against llama.cpp.

## Accuracy

Both engines ran the same `UD-Q4_K_M` file on the same inputs, scored the same way. The figures are from Petit
Chonk's native build of the release source. The ready-made engine gave byte-identical results to it on the CPU in a
spot check, and passed the same frozen check as below (worst statistic 0.43 of its threshold).

| Benchmark | Petit Chonk | llama.cpp |
|---|---|---|
| WikiText-2 perplexity (73 windows of 4,096 tokens; lower is better) | 5.889 | 5.864 |
| ARC-Challenge (300 questions) | 62.7% | 62.3% |
| CommonsenseQA (300) | 67.7% | 68.0% |
| MMLU (300) | 76.3% | 76.7% |
| All 900 | 68.9% | 69.0% |

- **Perplexity ratio:** 1.0042, with a 95% interval of 0.9987–1.0100 (bootstrap over windows).
- **The questions:** the two engines picked the same answer on 93.9% of them. Where they differed, neither was
  right more often (McNemar p = 1.0 on every set).
- **One outlier:** on one token of 298,935 Petit Chonk's probability differs from llama.cpp's far more than on any
  other. Its cause is not established. llama.cpp's own CPU and GPU paths disagree at that token too.
- The questions were drawn by a fixed seed and the scoring fixed before either engine ran. The formats are
  lm-evaluation-harness's.

## Speed

Writing speed is close to llama.cpp's: about the same on the first answer, 20–50% faster once its expert cache has
warmed up. Reading long prompts is where it gains most, about 3×.

Tokens per second on one PC: RTX 5080 16 GB, i9-10850K, 128 GB of DDR4-3200, Windows 11. The model is `UD-Q4_K_M`,
with the run config setup writes. Petit Chonk's figures are after its expert cache has warmed up (three requests).
llama.cpp is `llama-bench` with the experts of the first four layers on the card, its best placement there.

| Prompt | Reading the prompt: Petit Chonk | llama.cpp | Writing: Petit Chonk | llama.cpp |
|---|---|---|---|---|
| 1,024 tokens | 147–198 | 103 | 16–18 | 13.5 |
| 8,192 tokens | 379–381 | 117 | 16–20 | 13.3 |
| 16,000 tokens | 288–290 | 113 | 17–19 | 13.5 |

- **The first request after a start is slower,** while the expert cache fills. With a 16,000-token prompt it read
  68 tokens/s and wrote 16.6.
- **Longer contexts** (they run; outputs not checked): a 32,512-token prompt was read in 2.9 minutes and then
  13.3 tokens/s written. A 65,280-token prompt took 12.5 minutes, then 12.6 tokens/s.
- **With 64 GB of RAM** (emulated by locking 64 GiB on the same PC): about 4 tokens/s written, and prompts read at
  about 320 tokens/s.
- **Not measured:** other cards and PCs, contexts above 64K, PCs with less than 64 GB, several requests at once.

## How it was checked

- **The reference:** llama.cpp (commit `c7518dbe`) on the same model file. The thresholds come from llama.cpp's own
  spread between legitimate settings: chunked prompts, flash attention off, unfused operations, the CPU instead of
  the GPU, and token-by-token decoding.
- **The inputs:** 7 fixed token sequences, 26,016 positions, up to 16K tokens long. The tokenizer gives the same
  token ids as Mistral's own (`mistral-common`) on all 7, and on 44 million characters checked separately.
- **The rule:** a frozen acceptance contract judged every position's full output distribution, not only the top
  token.
- **The result:** passed in all 10 run sets (70 runs). They cover whole, chunked and split prompts and token-by-token
  decoding, with the experts on the card or on the CPU. The worst statistic reached 0.47 of its threshold.
  - On the release source, after merging current PolyStrata, a re-check passed (worst 0.39).
  - The ready-made engine passed the same re-check (worst 0.43).
- **What this shows:** Petit Chonk computes what llama.cpp computes, within llama.cpp's own spread, for this model
  file on these inputs. It is the project's own check, not an independent one. Other quantizations, cards,
  operating systems, sampling and contexts above 16K are not covered.

The tools are in [`tools/mistral4/`](tools/mistral4/). Details: [docs/MISTRAL4.md](docs/MISTRAL4.md).

## Using the rest of PolyStrata

Apart from Mistral Small 4 and the name in the web app and window titles, the code here is PolyStrata's.
[PolyStrata's README](https://github.com/simplexai-labs/PolyStrata#readme) and the [docs](docs/) folder describe it.
The translated READMEs (`README.de.md`, `README.es.md`, `README.fr.md`, `README.ja.md`, `README.pt-BR.md`,
`README.tr.md`, `README.zh-CN.md`) are PolyStrata's, and describe PolyStrata.

## Credits

Petit Chonk is built on other people's work:

- **[PolyStrata](https://github.com/simplexai-labs/PolyStrata)**, created by [VecSzn](https://github.com/VecSzn),
  and its contributors. It provides the `strata-poly` engine with its expert cache and CPU/GPU execution, the
  installer, the server and the web app. Petit Chonk takes its updates.
- **[Strata](https://github.com/Niko1221/Strata)** by Niko1221 and its contributors, which PolyStrata is built on.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**. It runs Mistral Small 4 with its DeepSeek2 graph, which is
  the reference for this port.
- **[Mistral AI](https://mistral.ai)**, which made Mistral Small 4 and publishes it under the
  [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) licence.
- **[Unsloth](https://unsloth.ai)**, whose GGUF quantizations of Mistral Small 4
  ([`unsloth/Mistral-Small-4-119B-2603-GGUF`](https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF))
  Petit Chonk is verified on.

## License

MIT, like PolyStrata and [Strata](https://github.com/Niko1221/Strata). [LICENSE](LICENSE) keeps both projects'
copyright notices and adds Petit Chonk's.

Petit Chonk's changes are not sent upstream. If a problem also happens on unchanged PolyStrata, report it to
[PolyStrata](https://github.com/simplexai-labs/PolyStrata/issues).
