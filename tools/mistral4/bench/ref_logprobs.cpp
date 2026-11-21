// P4 reference log-probability tool (coordination PCHONK-PUB-001; benchmarks B2 and B3): ref_logits.cpp (FTI T4,
// scripts\fti\t4, c66897fa...) with many sequences per process, writing only what the scoring needs.
//
// The model loading, placement (--gpu, --experts-cpu), context parameters (--ctx, --batch, --ubatch, --fa,
// --threads, n_seq_max 1) and --unfused are ref_logits's. Each sequence is computed the way ref_logits computes its
// prompt: one llama_decode of the whole sequence with every token an output, positions 0..n-1, sequence 0, on an
// empty memory. So a sequence's logits rows must be byte-identical to ref_logits's logits.f32 for the same ids and
// options (no follow ids): --rows writes them for that qualification.
//
// Input (--in F): JSON lines, one sequence a line:   {"id": "<string>", "ids": [token ids], "from": k}
//   Scores the positions whose next token index is >= k: log p(ids[j] | ids[0..j-1]) for j = k .. n-1, taken from
//   logits row j-1. Requires 1 <= k <= n-1, n <= --batch and n <= --ctx, ids unique, no other keys. Blank lines are
//   skipped. Every line is checked before the model loads; the token ids are checked against n_vocab after it
//   loads, before any decode.
//
// Between sequences: llama_memory_clear(llama_get_memory(ctx), true). It resets the cells and heads of the KV cache
//   and zeroes its buffers, which is the state of a new context (llama_memory_bufs::emplace zeroes them at
//   creation). It runs before every sequence, the first included, and the tool checks that the memory then holds
//   no position of sequence 0.
//
// Scoring, per scored row r (fp32, n_vocab values), in double precision over the full vocabulary:
//   m = max_v r[v];  lse = m + log(sum_v exp(r[v] - m));  logp = r[target] - lse.
//   argmax = the lowest token id holding the row maximum.
//   A row with any non-finite logit gets logp null and argmax -1; "nonfinite" counts the non-finite logits of the
//   scored rows (unscored rows are not read).
//
// Output (--out DIR, required): DIR/logprobs.jsonl, one line a sequence, in input order, flushed after each line:
//   {"id":<as given>,"n":n,"from":k,"logp":[n-k values],"argmax":[n-k ids],"nonfinite":c,"seconds":s}
//   logp values are printed with %.17g (they round-trip as doubles). seconds = wall time of the sequence's memory
//   clear, decode and scoring. The tool refuses to start if DIR/logprobs.jsonl (or a --rows file) already exists.
// --rows ID (repeatable): also write every logits row of that sequence (n rows of n_vocab floats: ref_logits's
//   logits.f32 layout for a prompt without follow ids) to DIR/rows_<index>.f32, index = the sequence's 0-based
//   position in the input (5 digits); its line then carries "rows":"rows_<index>.f32" before "seconds". The ID is
//   matched against the decoded input id and must be ASCII (Windows passes argv in the ANSI code page).
// --unfused 1: ref_logits's callback (no graph fusion; amax log in DIR/amax.jsonl), each line with a "seq" field
//   (the 0-based input index). "call" counts graph evaluations over the whole run.
//
// usage: ref_logprobs --model F --in F --out DIR [--ubatch N] [--gpu 0|1] [--fa auto|on|off] [--threads N]
//                     [--experts-cpu 0|1] [--ctx N] [--batch N] [--unfused 0|1] [--rows ID]...
//   defaults as ref_logits: ubatch 1024, gpu 1, fa auto, threads 10, experts-cpu 0, ctx 256, batch 1024, unfused 0.
//   --threads also sets the number of threads that score rows (each row is scored by one thread, so the result
//   does not depend on it).
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

struct Unfused {
    FILE * amax = nullptr;
    int call = -1;
    int seq = -1;                  // the 0-based input index of the sequence being decoded
    int pending_layer = -1;        // the layer whose ffn_moe_swiglu was seen, until its EXP node
    std::vector<uint8_t> raw;
};

static bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

static bool file_exists(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fclose(f);
    return true;
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
        std::fprintf(u.amax, "{\"seq\":%d,\"call\":%d,\"layer\":%d,\"ne\":[%lld,%lld,%lld],\"n\":%lld,\"max\":%g,"
                             "\"not_one\":%lld,\"nonfinite\":%lld}\n",
                     u.seq, u.call, u.pending_layer, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                     (long long) n, most, not_one, bad);
        u.pending_layer = -1;
    }
    return true;
}

// ---- input: a strict reader for {"id": string, "ids": [integers], "from": integer} ----

struct Seq {
    std::string id_json;               // the id as written in the input (a JSON string), echoed in the output
    std::string id;                    // decoded, for --rows and the uniqueness check
    std::vector<llama_token> ids;
    int from = -1;
    int line = 0;                      // 1-based line number in the input
    bool rows = false;
};

struct Reader {
    const char * p;
    const char * e;
    std::string err;

    void ws() { while (p < e && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) ++p; }
    bool fail(const std::string & m) { if (err.empty()) err = m; return false; }
    bool eat(char c) { ws(); if (p < e && *p == c) { ++p; return true; } return false; }

    static void put_utf8(std::string & o, uint32_t c) {
        if (c < 0x80) o.push_back((char) c);
        else if (c < 0x800) { o.push_back((char) (0xC0 | (c >> 6))); o.push_back((char) (0x80 | (c & 0x3F))); }
        else if (c < 0x10000) {
            o.push_back((char) (0xE0 | (c >> 12))); o.push_back((char) (0x80 | ((c >> 6) & 0x3F)));
            o.push_back((char) (0x80 | (c & 0x3F)));
        } else {
            o.push_back((char) (0xF0 | (c >> 18))); o.push_back((char) (0x80 | ((c >> 12) & 0x3F)));
            o.push_back((char) (0x80 | ((c >> 6) & 0x3F))); o.push_back((char) (0x80 | (c & 0x3F)));
        }
    }

    bool hex4(uint32_t & v) {
        if (e - p < 4) return fail("short \\u escape");
        v = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = *p++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t) (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t) (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t) (c - 'A' + 10);
            else return fail("bad \\u escape");
        }
        return true;
    }

    bool str(std::string & out) {
        ws();
        if (p >= e || *p != '"') return fail("expected a string");
        ++p;
        out.clear();
        for (;;) {
            if (p >= e) return fail("unterminated string");
            const unsigned char c = (unsigned char) *p++;
            if (c == '"') return true;
            if (c < 0x20) return fail("control character in a string");
            if (c != '\\') { out.push_back((char) c); continue; }
            if (p >= e) return fail("unterminated escape");
            const char x = *p++;
            switch (x) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    uint32_t c1 = 0, c2 = 0;
                    if (!hex4(c1)) return false;
                    if (c1 >= 0xD800 && c1 <= 0xDBFF) {
                        if (e - p < 2 || p[0] != '\\' || p[1] != 'u') return fail("unpaired surrogate");
                        p += 2;
                        if (!hex4(c2)) return false;
                        if (c2 < 0xDC00 || c2 > 0xDFFF) return fail("unpaired surrogate");
                        c1 = 0x10000 + ((c1 - 0xD800) << 10) + (c2 - 0xDC00);
                    } else if (c1 >= 0xDC00 && c1 <= 0xDFFF) {
                        return fail("unpaired surrogate");
                    }
                    put_utf8(out, c1);
                    break;
                }
                default: return fail("bad escape");
            }
        }
    }

    bool integer(long long & v) {
        ws();
        bool neg = false;
        if (p < e && *p == '-') { neg = true; ++p; }
        if (p >= e || !std::isdigit((unsigned char) *p)) return fail("expected an integer");
        long long x = 0;
        int digits = 0;
        while (p < e && std::isdigit((unsigned char) *p)) {
            if (++digits > 12) return fail("integer too long");
            x = x * 10 + (*p++ - '0');
        }
        if (p < e && (*p == '.' || *p == 'e' || *p == 'E')) return fail("expected an integer, found a fraction");
        v = neg ? -x : x;
        return true;
    }
};

static bool parse_line(const char * b, const char * e, Seq & s, std::string & err) {
    Reader r{ b, e, {} };
    bool have_id = false, have_ids = false, have_from = false;
    if (!r.eat('{')) { err = "expected an object"; return false; }
    if (!r.eat('}')) {
        for (;;) {
            std::string key;
            if (!r.str(key)) { err = r.err; return false; }
            if (!r.eat(':')) { err = "expected ':' after \"" + key + "\""; return false; }
            if (key == "id") {
                if (have_id) { err = "id given twice"; return false; }
                r.ws();
                const char * s0 = r.p;
                if (!r.str(s.id)) { err = "id: " + r.err; return false; }
                s.id_json.assign(s0, r.p);
                have_id = true;
            } else if (key == "ids") {
                if (have_ids) { err = "ids given twice"; return false; }
                if (!r.eat('[')) { err = "ids: expected a list"; return false; }
                if (!r.eat(']')) {
                    for (;;) {
                        long long v = 0;
                        if (!r.integer(v)) { err = "ids: " + r.err; return false; }
                        if (v < 0 || v > 2147483647LL) { err = "ids: token id out of range"; return false; }
                        s.ids.push_back((llama_token) v);
                        if (r.eat(',')) continue;
                        if (r.eat(']')) break;
                        err = "ids: expected ',' or ']'";
                        return false;
                    }
                }
                have_ids = true;
            } else if (key == "from") {
                if (have_from) { err = "from given twice"; return false; }
                long long v = 0;
                if (!r.integer(v)) { err = "from: " + r.err; return false; }
                if (v < 0 || v > 2147483647LL) { err = "from out of range"; return false; }
                s.from = (int) v;
                have_from = true;
            } else {
                err = "unknown key \"" + key + "\"";
                return false;
            }
            if (r.eat(',')) continue;
            if (r.eat('}')) break;
            err = "expected ',' or '}'";
            return false;
        }
    }
    r.ws();
    if (r.p != e) { err = "text after the object"; return false; }
    if (!have_id || !have_ids || !have_from) { err = "a line needs id, ids and from"; return false; }
    return true;
}

// ---- scoring ----

// lp[j], am[j], bad[j] for rows[j] and its target tg[j]; each row is scored by one thread, start to end in order
static void score_rows(const std::vector<const float *> & rows, const llama_token * tg, int n_vocab, int n_threads,
                       std::vector<double> & lp, std::vector<int> & am, std::vector<long long> & bad) {
    const int m = (int) rows.size();
    auto work = [&](int j0, int j1) {
        for (int j = j0; j < j1; ++j) {
            const float * r = rows[j];
            long long nb = 0;
            float mx = 0.f;
            int arg = -1;
            for (int v = 0; v < n_vocab; ++v) {
                const float x = r[v];
                if (!std::isfinite(x)) { ++nb; continue; }
                if (arg < 0 || x > mx) { mx = x; arg = v; }
            }
            bad[j] = nb;
            if (nb != 0) { lp[j] = NAN; am[j] = -1; continue; }
            const double md = (double) mx;
            double sum = 0.0;
            for (int v = 0; v < n_vocab; ++v) sum += std::exp((double) r[v] - md);
            lp[j] = (double) r[tg[j]] - (md + std::log(sum));
            am[j] = arg;
        }
    };
    const int nt = std::max(1, std::min(n_threads, m));
    std::vector<std::thread> pool;
    for (int t = 1; t < nt; ++t) {
        pool.emplace_back(work, (int) ((long long) m * t / nt), (int) ((long long) m * (t + 1) / nt));
    }
    work(0, (int) ((long long) m / nt));
    for (auto & th : pool) th.join();
}

int main(int argc, char ** argv) {
    std::string model_path, in_path, out, fa = "auto";
    std::vector<std::string> rows_ids;
    int ubatch = 1024, gpu = 1, threads = 10, experts_cpu = 0, n_ctx = 256, n_batch = 1024, unfused = 0;
    if (argc % 2 == 0) { std::fprintf(stderr, "options come in pairs (option value)\n"); return 2; }
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--model") model_path = v;
        else if (a == "--in") in_path = v;
        else if (a == "--out") out = v;
        else if (a == "--rows") rows_ids.push_back(v);
        else if (a == "--ubatch") ubatch = std::atoi(v.c_str());
        else if (a == "--gpu") gpu = std::atoi(v.c_str());
        else if (a == "--fa") fa = v;
        else if (a == "--threads") threads = std::atoi(v.c_str());
        else if (a == "--experts-cpu") experts_cpu = std::atoi(v.c_str());
        else if (a == "--ctx") n_ctx = std::atoi(v.c_str());
        else if (a == "--batch") n_batch = std::atoi(v.c_str());
        else if (a == "--unfused") unfused = std::atoi(v.c_str());
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || in_path.empty() || out.empty()) {
        std::fprintf(stderr, "--model, --in and --out are required\n");
        return 2;
    }

    // ---- read and check every sequence before the model loads ----
    std::string text;
    if (!read_file(in_path, text)) { std::fprintf(stderr, "cannot read %s\n", in_path.c_str()); return 2; }
    std::vector<Seq> seqs;
    {
        std::set<std::string> seen;
        size_t pos = (text.size() >= 3 && std::memcmp(text.data(), "\xEF\xBB\xBF", 3) == 0) ? 3 : 0;
        int line_no = 0;
        while (pos < text.size()) {
            size_t nl = text.find('\n', pos);
            if (nl == std::string::npos) nl = text.size();
            ++line_no;
            const char * b = text.data() + pos;
            const char * e = text.data() + nl;
            pos = nl + 1;
            const char * q = b;
            while (q < e && std::isspace((unsigned char) *q)) ++q;
            if (q == e) continue;
            Seq s;
            s.line = line_no;
            std::string err;
            if (!parse_line(b, e, s, err)) {
                std::fprintf(stderr, "%s line %d: %s\n", in_path.c_str(), line_no, err.c_str());
                return 2;
            }
            const int n = (int) s.ids.size();
            if (n < 2 || s.from < 1 || s.from > n - 1) {
                std::fprintf(stderr, "%s line %d: needs 1 <= from <= n-1 (n %d, from %d)\n", in_path.c_str(), line_no,
                             n, s.from);
                return 2;
            }
            if (n > n_batch || n > n_ctx) {
                std::fprintf(stderr, "%s line %d: the sequence (%d) is larger than --batch (%d) or --ctx (%d)\n",
                             in_path.c_str(), line_no, n, n_batch, n_ctx);
                return 2;
            }
            if (!seen.insert(s.id).second) {
                std::fprintf(stderr, "%s line %d: id %s is not unique\n", in_path.c_str(), line_no, s.id_json.c_str());
                return 2;
            }
            seqs.push_back(std::move(s));
        }
    }
    if (seqs.empty()) { std::fprintf(stderr, "no sequences in %s\n", in_path.c_str()); return 2; }
    for (const std::string & id : rows_ids) {
        for (unsigned char c : id) {
            if (c >= 0x80) {
                std::fprintf(stderr, "--rows %s: ids named on the command line must be ASCII (Windows passes argv in "
                                     "the ANSI code page, the input is UTF-8)\n", id.c_str());
                return 2;
            }
        }
        bool found = false;
        for (Seq & s : seqs) if (s.id == id) { s.rows = true; found = true; }
        if (!found) { std::fprintf(stderr, "--rows %s: no such id in the input\n", id.c_str()); return 2; }
    }
    auto rows_name = [](size_t k) {
        char name[32];
        std::snprintf(name, sizeof(name), "rows_%05zu.f32", k);
        return std::string(name);
    };
    const std::string out_path = out + "/logprobs.jsonl";
    if (file_exists(out_path)) { std::fprintf(stderr, "%s exists; not overwriting it\n", out_path.c_str()); return 2; }
    for (size_t k = 0; k < seqs.size(); ++k) {
        if (seqs[k].rows && file_exists(out + "/" + rows_name(k))) {
            std::fprintf(stderr, "%s/%s exists; not overwriting it\n", out.c_str(), rows_name(k).c_str());
            return 2;
        }
    }
    FILE * of = std::fopen(out_path.c_str(), "wb");
    if (!of) { std::fprintf(stderr, "cannot write in %s\n", out.c_str()); return 1; }

    // ---- model and context: ref_logits's ----
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
        if (!u.amax) { std::fprintf(stderr, "cannot write in %s\n", out.c_str()); return 1; }
        cp.cb_eval = cb_unfused;
        cp.cb_eval_user_data = &u;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "context init failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (const Seq & s : seqs) {
        for (llama_token t : s.ids) {
            if (t < 0 || t >= n_vocab) {
                std::fprintf(stderr, "%s line %d: token id %d is not below n_vocab %d\n", in_path.c_str(), s.line, t,
                             n_vocab);
                return 2;
            }
        }
    }
    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) { std::fprintf(stderr, "the context has no memory\n"); return 1; }

    // ---- one sequence after another ----
    using clk = std::chrono::steady_clock;
    const auto t_run = clk::now();
    long long n_tokens_total = 0, n_scored_total = 0, nonfinite_total = 0;
    std::vector<const float *> rowp;
    std::vector<double> lp;
    std::vector<int> am;
    std::vector<long long> bad;
    std::string line;
    char num[64];
    for (size_t k = 0; k < seqs.size(); ++k) {
        const Seq & s = seqs[k];
        const int n = (int) s.ids.size();
        const int m = n - s.from;
        const auto t0 = clk::now();

        llama_memory_clear(mem, true);
        if (llama_memory_seq_pos_max(mem, 0) != -1) {
            std::fprintf(stderr, "sequence %zu (line %d): the memory is not empty after the clear\n", k, s.line);
            return 1;
        }
        u.seq = (int) k;

        llama_batch b = llama_batch_init(n, 0, 1);
        b.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            b.token[i] = s.ids[i]; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        const int rc = llama_decode(ctx, b);
        if (rc != 0) {
            std::fprintf(stderr, "sequence %zu (line %d): decode failed: %d\n", k, s.line, rc);
            return 1;
        }

        rowp.assign(m, nullptr);
        for (int j = 0; j < m; ++j) {
            rowp[j] = llama_get_logits_ith(ctx, s.from - 1 + j);
            if (!rowp[j]) {
                std::fprintf(stderr, "sequence %zu (line %d): no logits for row %d\n", k, s.line, s.from - 1 + j);
                return 1;
            }
        }
        lp.assign(m, 0.0);
        am.assign(m, -1);
        bad.assign(m, 0);
        score_rows(rowp, s.ids.data() + s.from, n_vocab, threads, lp, am, bad);
        const double secs = std::chrono::duration<double>(clk::now() - t0).count();

        long long nonfinite = 0;
        for (int j = 0; j < m; ++j) nonfinite += bad[j];

        if (s.rows) {
            const std::string rp = out + "/" + rows_name(k);
            FILE * rf = std::fopen(rp.c_str(), "wb");
            if (!rf) { std::fprintf(stderr, "cannot write %s\n", rp.c_str()); return 1; }
            for (int i = 0; i < n; ++i) {
                const float * row = llama_get_logits_ith(ctx, i);
                if (!row || std::fwrite(row, sizeof(float), (size_t) n_vocab, rf) != (size_t) n_vocab) {
                    std::fprintf(stderr, "writing %s failed at row %d\n", rp.c_str(), i);
                    return 1;
                }
            }
            if (std::fclose(rf) != 0) { std::fprintf(stderr, "closing %s failed\n", rp.c_str()); return 1; }
        }
        llama_batch_free(b);

        line.clear();
        line += "{\"id\":";
        line += s.id_json;
        line += ",\"n\":" + std::to_string(n) + ",\"from\":" + std::to_string(s.from) + ",\"logp\":[";
        for (int j = 0; j < m; ++j) {
            if (j) line += ',';
            if (std::isfinite(lp[j])) { std::snprintf(num, sizeof(num), "%.17g", lp[j]); line += num; }
            else line += "null";
        }
        line += "],\"argmax\":[";
        for (int j = 0; j < m; ++j) {
            if (j) line += ',';
            line += std::to_string(am[j]);
        }
        line += "],\"nonfinite\":" + std::to_string(nonfinite);
        if (s.rows) line += ",\"rows\":\"" + rows_name(k) + "\"";
        std::snprintf(num, sizeof(num), "%.3f", secs);
        line += ",\"seconds\":";
        line += num;
        line += "}\n";
        if (std::fwrite(line.data(), 1, line.size(), of) != line.size() || std::fflush(of) != 0) {
            std::fprintf(stderr, "writing %s failed\n", out_path.c_str());
            return 1;
        }

        n_tokens_total += n;
        n_scored_total += m;
        nonfinite_total += nonfinite;
        std::printf("seq %zu/%zu: n %d, from %d, %.3f s\n", k + 1, seqs.size(), n, s.from, secs);
        std::fflush(stdout);
    }
    if (std::fclose(of) != 0) { std::fprintf(stderr, "closing %s failed\n", out_path.c_str()); return 1; }
    if (u.amax) std::fclose(u.amax);
    const double run_secs = std::chrono::duration<double>(clk::now() - t_run).count();
    std::printf("done: %zu sequences, %lld tokens, %lld scored positions, %lld non-finite logits, n_vocab %d, ctx %d, "
                "batch %d, ubatch %d, gpu %d, experts-cpu %d, fa %s, unfused %d, calls %d, %.1f s\n",
                seqs.size(), n_tokens_total, n_scored_total, nonfinite_total, n_vocab, n_ctx, n_batch, ubatch, gpu,
                experts_cpu, fa.c_str(), unfused, u.call + 1, run_secs);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
