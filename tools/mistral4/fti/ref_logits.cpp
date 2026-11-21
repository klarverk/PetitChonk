// T4 reference logits tool (coordination PCHONK-FTI-001): capture_l0.cpp (S6b) without its tensor capture, plus
// what the full-text-inference reference runs need. With --ctx 256 --batch 1024 and --unfused 0 it computes exactly
// what `capture_l0 --cap 0` computes (same model and context parameters, same calls), so its logits.f32 must be
// byte-identical to capture_l0's: that is T4's qualification.
//
//   --prompt-file F / --follow-file F   token ids from a file (comma, space or newline separated; a JSON list
//                                        works), because a 15K-token sequence does not fit a Windows command line.
//                                        --prompt / --follow still take ids inline.
//   --ctx N --batch N                    n_ctx and n_batch (defaults 256 and 1024, capture_l0's). The prompt is one
//                                        llama_decode, so n_batch must hold it.
//   --unfused 1                          a callback that asks for every node except result_output (as capture_l0's
//                                        capture runs did), so the scheduler does not fuse the graph; it writes no
//                                        tensors. It records the amax rescale of each layer's routed-expert down input
//                                        (the first EXP node after "ffn_moe_swiglu-<L>", shape [1, n_used, n]) into
//                                        amax.jsonl: call, layer, values, max, how many are not 1, non-finite count.
//
// Every prompt token is an output (the logits of every position); the follow ids go one token a call (the decode
// path). Output: logits.f32 (n_vocab floats a row, prompt rows then follow rows), steps.jsonl, amax.jsonl.
//
// usage: ref_logits --model F (--prompt ids | --prompt-file F) [--follow ids | --follow-file F] --ubatch N
//                   --gpu 0|1 --fa auto|on|off --threads N --out DIR [--experts-cpu 0|1] [--ctx N] [--batch N]
//                   [--unfused 0|1]
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct Unfused {
    FILE * amax = nullptr;
    int call = -1;
    int pending_layer = -1;        // the layer whose ffn_moe_swiglu was seen, until its EXP node
    std::vector<uint8_t> raw;
};

static std::vector<llama_token> parse_ids(const std::string & s) {
    std::vector<llama_token> v;
    const char * p = s.c_str();
    while (*p) {
        if (std::isdigit((unsigned char) *p) || (*p == '-' && std::isdigit((unsigned char) p[1]))) {
            char * end = nullptr;
            v.push_back((llama_token) std::strtol(p, &end, 10));
            p = end;
        } else {
            ++p;
        }
    }
    return v;
}

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// the layer a node name belongs to (capture_l0's rule): the k of the last "-<k>" that ends the name or is followed
// by a space; -1 for a name without one
static int layer_of(const char * name) {
    int layer = -1;
    for (const char * p = name; *p; ++p) {
        if (*p != '-' || !std::isdigit((unsigned char) p[1])) continue;
        const char * q = p + 1;
        int k = 0;
        while (std::isdigit((unsigned char) *q)) k = k * 10 + (*q++ - '0');
        if (*q == '\0' || *q == ' ') layer = k;
    }
    return layer;
}

static bool cb_unfused(ggml_tensor * t, bool ask, void * ud) {
    Unfused & u = *(Unfused *) ud;
    if (ask) return std::strcmp(t->name, "result_output") != 0;
    if (std::strcmp(t->name, "embd") == 0 || u.call < 0) { ++u.call; u.pending_layer = -1; }
    if (std::strncmp(t->name, "ffn_moe_swiglu-", 15) == 0) {
        u.pending_layer = layer_of(t->name);
        return true;
    }
    if (u.pending_layer >= 0 && std::strcmp(ggml_op_desc(t), "EXP") == 0 && t->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(t)) {
        const int64_t n = ggml_nelements(t);
        u.raw.resize((size_t) n * sizeof(float));
        ggml_backend_tensor_get(t, u.raw.data(), 0, u.raw.size());
        const float * e = (const float *) u.raw.data();
        float most = 0.f;
        long long not_one = 0, bad = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (!std::isfinite(e[i])) { ++bad; continue; }
            if (e[i] > most) most = e[i];
            not_one += e[i] != 1.f;
        }
        std::fprintf(u.amax, "{\"call\":%d,\"layer\":%d,\"ne\":[%lld,%lld,%lld],\"n\":%lld,\"max\":%g,\"not_one\":%lld,"
                             "\"nonfinite\":%lld}\n",
                     u.call, u.pending_layer, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                     (long long) n, most, not_one, bad);
        u.pending_layer = -1;
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, out = ".", fa = "auto";
    std::vector<llama_token> prompt, follow;
    int ubatch = 1024, gpu = 1, threads = 10, experts_cpu = 0, n_ctx = 256, n_batch = 1024, unfused = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--model") model_path = v;
        else if (a == "--prompt") prompt = parse_ids(v);
        else if (a == "--prompt-file") prompt = parse_ids(read_file(v));
        else if (a == "--follow") follow = parse_ids(v);
        else if (a == "--follow-file") follow = parse_ids(read_file(v));
        else if (a == "--ubatch") ubatch = std::atoi(v.c_str());
        else if (a == "--gpu") gpu = std::atoi(v.c_str());
        else if (a == "--fa") fa = v;
        else if (a == "--threads") threads = std::atoi(v.c_str());
        else if (a == "--out") out = v;
        else if (a == "--experts-cpu") experts_cpu = std::atoi(v.c_str());
        else if (a == "--ctx") n_ctx = std::atoi(v.c_str());
        else if (a == "--batch") n_batch = std::atoi(v.c_str());
        else if (a == "--unfused") unfused = std::atoi(v.c_str());
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (prompt.empty()) { std::fprintf(stderr, "no prompt\n"); return 2; }
    if ((int) prompt.size() > n_batch) {
        std::fprintf(stderr, "the prompt (%zu) is larger than --batch (%d)\n", prompt.size(), n_batch);
        return 2;
    }
    if ((int) (prompt.size() + follow.size()) > n_ctx) {
        std::fprintf(stderr, "prompt + follow (%zu) exceed --ctx (%d)\n", prompt.size() + follow.size(), n_ctx);
        return 2;
    }
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t no_devices[1] = { nullptr };
    if (gpu) mp.n_gpu_layers = -1;
    else { mp.n_gpu_layers = 0; mp.devices = no_devices; }
    llama_model_tensor_buft_override overrides[2] = { { nullptr, nullptr }, { nullptr, nullptr } };
    if (experts_cpu) {
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu) { std::fprintf(stderr, "no CPU device\n"); return 1; }
        overrides[0] = { "blk\\.([1-9][0-9]*)\\.ffn_.*_exps", ggml_backend_dev_buffer_type(cpu) };
        mp.tensor_buft_overrides = overrides;
    }
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }

    Unfused u;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) n_ctx;
    cp.n_batch = (uint32_t) n_batch;
    cp.n_ubatch = (uint32_t) ubatch;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = threads;
    cp.flash_attn_type = fa == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                       : fa == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED : LLAMA_FLASH_ATTN_TYPE_AUTO;
    if (!gpu) cp.op_offload = false;
    if (unfused) {
        u.amax = std::fopen((out + "/amax.jsonl").c_str(), "w");
        cp.cb_eval = cb_unfused;
        cp.cb_eval_user_data = &u;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context init failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    FILE * lf = std::fopen((out + "/logits.f32").c_str(), "wb");
    FILE * steps = std::fopen((out + "/steps.jsonl").c_str(), "w");
    if (!lf || !steps) { std::fprintf(stderr, "cannot write in %s\n", out.c_str()); return 1; }

    llama_batch b = llama_batch_init((int32_t) prompt.size(), 0, 1);
    b.n_tokens = (int32_t) prompt.size();
    for (int i = 0; i < b.n_tokens; ++i) {
        b.token[i] = prompt[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
    }
    int rc = llama_decode(ctx, b);
    if (rc != 0) { std::fprintf(stderr, "prompt decode failed: %d\n", rc); return 1; }
    for (int i = 0; i < b.n_tokens; ++i) std::fwrite(llama_get_logits_ith(ctx, i), sizeof(float), n_vocab, lf);
    std::fprintf(steps, "{\"kind\":\"prompt\",\"n_tokens\":%d,\"pos0\":0}\n", b.n_tokens);
    llama_batch_free(b);

    int pos = (int) prompt.size();
    for (llama_token t : follow) {
        llama_batch s = llama_batch_init(1, 0, 1);
        s.n_tokens = 1; s.token[0] = t; s.pos[0] = pos; s.n_seq_id[0] = 1; s.seq_id[0][0] = 0; s.logits[0] = 1;
        rc = llama_decode(ctx, s);
        if (rc != 0) { std::fprintf(stderr, "decode at %d failed: %d\n", pos, rc); return 1; }
        std::fwrite(llama_get_logits_ith(ctx, 0), sizeof(float), n_vocab, lf);
        std::fprintf(steps, "{\"kind\":\"decode\",\"n_tokens\":1,\"pos0\":%d,\"token\":%d}\n", pos, t);
        llama_batch_free(s);
        ++pos;
    }
    std::fclose(lf);
    std::fclose(steps);
    if (u.amax) std::fclose(u.amax);
    std::printf("done: %d prompt tokens, %zu follow tokens, n_vocab %d, ctx %d, batch %d, ubatch %d, gpu %d, "
                "experts-cpu %d, fa %s, unfused %d, calls %d\n", (int) prompt.size(), follow.size(), n_vocab, n_ctx,
                n_batch, ubatch, gpu, experts_cpu, fa.c_str(), unfused, u.call + 1);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
