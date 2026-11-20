<h1 align="center">Petit Chonk</h1>

<p align="center"><b>Mistral Small 4 on one 16 GB graphics card and system RAM</b><br>
A fork of <a href="https://github.com/simplexai-labs/PolyStrata">PolyStrata</a></p>

> **Status: early.** No Mistral Small 4 support has been written, built or tested yet. Everything that runs here
> today is PolyStrata's.

## The goal

Run [Mistral Small 4](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) (`mistralai/Mistral-Small-4-119B-2603`)
on a normal PC by adding it to PolyStrata's second engine, `strata-poly`.

Mistral Small 4 is a 119B mixture-of-experts model. Each of its 36 layers has 128 experts. For each token the model
uses four of them plus one shared expert. Only a small part of the weights is used per token, but all of them must be
stored somewhere. `strata-poly` already handles models like this. It keeps a cache of experts on the graphics card
and the rest in system RAM, and it can run experts on the CPU or the GPU. For Mistral Small 4 it still needs:

- the model's attention (multi-head latent attention) and its compressed attention cache;
- the model's tokenizer (Tekken) and chat format, including both reasoning settings, `none` and `high`.

The first target is text only: Windows 11, one NVIDIA RTX 5080 with 16 GB plus 32 GB of RAM, one request at a
time and one GGUF file.

Two rules come before speed:

- **Faithful.** The port computes what the model computes: the four chosen experts and the shared expert, with
  no experts skipped and no values clipped.
- **Measured.** A result counts only after it has been compared with a reference on the same model file and
  inputs. A model that loads, or that writes fluent text, has not passed.

The last step decides whether the port is worth using. It measures the port against
[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) on the same machine.

## Plan

1. Record the starting point: source commits, model file hashes, the machine and the build.
2. Make one Mistral transformer layer match a reference.
3. Run all layers. Compare the final outputs on fixed inputs, and check that the tokenizer gives the same token IDs
   as the official one.
4. Add chat: both reasoning settings, streaming, tool calls, and saving and restoring conversations.
5. Measure speed and memory on the target machine with 8K, 32K and 64K tokens of context, against ik_llama.cpp.

Vision, speculative decoding, multiple GPUs and a quantized attention cache wait until all five steps are done.

Step 1 has not started.

## Using it today

Apart from its name in the web app and the window titles, the code here is PolyStrata's. The installer, server,
web app and supported models are PolyStrata's too, and [PolyStrata's README](https://github.com/simplexai-labs/PolyStrata#readme)
and the [docs](docs/) folder describe them.

## Credits

Petit Chonk is built on other people's work:

- **[PolyStrata](https://github.com/simplexai-labs/PolyStrata)**, created by [VecSzn](https://github.com/VecSzn),
  and its contributors. It provides the `strata-poly` engine with its expert cache and CPU/GPU execution, the
  installer, the server and the web app. Petit Chonk takes its updates.
- **[Strata](https://github.com/Niko1221/Strata)** by Niko1221 and its contributors, which PolyStrata is built on.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**. It runs Mistral Small 4 with its DeepSeek2 graph, which is
  the reference for this port.
- **[Mistral AI](https://mistral.ai)**, which made Mistral Small 4.

## License

MIT, like PolyStrata and [Strata](https://github.com/Niko1221/Strata). [LICENSE](LICENSE) keeps both projects'
copyright notices.

Petit Chonk's changes are not sent upstream. If a problem also happens on unchanged PolyStrata, report it to
[PolyStrata](https://github.com/simplexai-labs/PolyStrata/issues).
