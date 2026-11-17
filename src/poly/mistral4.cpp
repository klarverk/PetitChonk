// src/poly/mistral4.cpp - the layers of Mistral Small 4.  The reference for the math is llama.cpp's deepseek2
// graph (src/models/deepseek2.cpp, c7518dbe), which its mistral4 model runs, and the YaRN factors of its
// llama-context.cpp.
#include "mistral4.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time
constexpr float kPi = 3.14159265358979323846f;

// ggml_rope_ext's arguments and ggml_rope_yarn_corr_dims(), as deepseek4.cpp's make_rope() has them.
Rope make_rope(float base, int n_rot, float freq_scale, float ext_factor, int n_ctx_orig, float beta_fast,
               float beta_slow) {
    Rope r;
    r.theta_scale = powf(base, -2.0f / (float) n_rot);
    r.freq_scale = freq_scale;
    r.ext_factor = ext_factor;
    r.attn_factor = ext_factor == 0.0f ? 1.0f : 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));
    const auto corr = [&](float rot) {
        return (float) n_rot * logf((float) n_ctx_orig / (rot * 2 * kPi)) / (2 * logf(base));
    };
    r.corr0 = std::max(0.0f, floorf(corr(beta_fast)));
    r.corr1 = std::min((float) (n_rot - 1), ceilf(corr(beta_slow)));
    return r;
}

// Rows of `width` floats from one buffer to another, `from` and `to` floats apart.
void rows2d(float* dst, size_t to, const float* src, size_t from, size_t width, size_t rows, void* stream) {
    const size_t f = sizeof(float);
    ck(cudaMemcpy2DAsync(dst, to * f, src, from * f, width * f, rows, cudaMemcpyDeviceToDevice, cs(stream)),
       "mistral4 rows");
}

}  // namespace

void Mistral4::begin_pass() { blocks_.assign(layers_.size(), Block{}); }

void Mistral4::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    auto num = [&](const char* key, float fallback) {
        const MetaValue* v = find(key);
        return v ? (float) v->num() : fallback;
    };
    n_head_ = shape_i("attention.head_count");
    need(shape_i("attention.head_count_kv") == 1, "attention.head_count_kv is not 1 (a latent row a position)");
    q_rank_ = shape_i("attention.q_lora_rank");
    kv_rank_ = shape_i("attention.kv_lora_rank");
    n_rot_ = shape_i("rope.dimension_count");
    k_mla_ = shape_i("attention.key_length_mla");
    v_mla_ = shape_i("attention.value_length_mla");
    nope_ = k_mla_ - n_rot_;
    row_ = kv_rank_ + n_rot_;
    need(n_head_ > 0 && n_head_ <= 256, "bad attention.head_count");
    need(shape_i("attention.key_length") == row_, "attention.key_length is not kv_lora_rank + rope.dimension_count");
    need(shape_i("attention.value_length") == kv_rank_, "attention.value_length is not kv_lora_rank");
    need(nope_ > 0 && n_rot_ % 2 == 0 && kv_rank_ % 32 == 0 && q_rank_ % 32 == 0 && v_mla_ > 0, "bad MLA widths");
    const MetaValue* dense = find("leading_dense_block_count");
    need(!dense || dense->u == 0, "leading_dense_block_count is not 0");
    // the reference's deepseek2 takes a file without the key as softmax (llama.cpp's 1)
    const MetaValue* gating = find("expert_gating_func");
    need(!gating || gating->u == 1, "the expert gate is not a softmax");
    const MetaValue* groups = find("expert_group_count");
    need(!groups || groups->u <= 1, "expert groups");
    need(shape_i("expert_shared_count") == 1, "expert_shared_count is not 1");
    // a temperature on the queries by the position would be applied by the reference; this backend has none
    need(!find("attention.temperature_scale"), "attention.temperature_scale");
    const MetaValue* norm = find("expert_weights_norm");
    w_norm_ = norm && norm->u != 0;
    w_scale_ = num("expert_weights_scale", 0.f);

    // YaRN, as llama-context.cpp and deepseek2.cpp take it.  The converter stores 0.1 * mscale_all_dim as the log
    // multiplier, and the reference divides it back (deepseek2.cpp:34-37).  The context's attn_factor cancels the
    // turn's own magnitude, and the graph puts mscale squared on the softmax's scale instead.
    const MetaValue* type = find("rope.scaling.type");
    need(type && type->s == "yarn", "rope.scaling.type is not yarn");
    const float base = shape_f("rope.freq_base"), factor = shape_f("rope.scaling.factor");
    need(factor > 1.f, "bad rope.scaling.factor");
    const int n_orig = shape_i("rope.scaling.original_context_length");
    const float log_mul = num("rope.scaling.yarn_log_multiplier", 0.f) / 0.1f;
    const auto m = [](float s, float k) { return s <= 1.f ? 1.f : 0.1f * k * logf(s) + 1.f; };
    float yarn_attn = log_mul != 0.f ? m(factor, 1.f) / m(factor, log_mul) : m(factor, 1.f);
    yarn_attn *= 1.f / (1.f + 0.1f * logf(factor));
    yarn_attn *= num("rope.scaling.attn_factor", 1.f);
    const float mscale = yarn_attn * (1.f + 0.1f * logf(factor)) * (1.f + 0.1f * log_mul * logf(factor));
    kq_scale_ = mscale * mscale / std::sqrt((float) k_mla_);
    rope_ = make_rope(base, n_rot_, 1.f / factor, 1.f, n_orig, num("rope.scaling.yarn_beta_fast", 32.f),
                      num("rope.scaling.yarn_beta_slow", 1.f));
    rope_.attn_factor = yarn_attn;
    if (opt_.dump)
        std::fprintf(stderr, "mistral4: kq_scale=%.7f rope attn_factor=%.7f corr=[%g, %g] theta_scale=%.7f\n",
                     kq_scale_, rope_.attn_factor, rope_.corr0, rope_.corr1, rope_.theta_scale);

    // no limit on a SwiGLU's inputs (llama-model.cpp: swiglu_clamp_exp is 0 for this model)
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
    gelu_ = false;
    // the routed experts' down input can leave a half's range (the reference rescales it, llama-graph.cpp:2584):
    // a prompt batch's products and a few tokens' both go through guard_half_sums(), which is exact
    half_sum_guard_ = true;
    guard_expert_down_ = true;
    // STRATA_MISTRAL4_GUARD=0, for the guard's own test only: both guards off, to show what they protect from
    if (const char* v = std::getenv("STRATA_MISTRAL4_GUARD"); v && std::strcmp(v, "0") == 0) {
        half_sum_guard_ = guard_expert_down_ = false;
        std::fprintf(stderr, "mistral4: STRATA_MISTRAL4_GUARD=0: the down input's guards are off (a test only)\n");
    }
}

void Mistral4::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    L.fixed = false;   // the attention takes its position as an argument, so its launches are not recorded
    B.q_a = mat("attn_q_a.weight", n_embd_, q_rank_);
    B.q_a_norm = f32("attn_q_a_norm.weight", q_rank_);
    B.q_b = mat("attn_q_b.weight", q_rank_, (int64_t) n_head_ * k_mla_);
    B.kv_a = mat("attn_kv_a_mqa.weight", n_embd_, row_);
    B.kv_a_norm = f32("attn_kv_a_norm.weight", kv_rank_);
    // the per-head projections as floats: Q8_0's blocks turn into floats exactly, and the grouped products take them
    B.k_b = f32("attn_k_b.weight", nope_, (int64_t) kv_rank_ * n_head_);
    B.v_b = f32("attn_v_b.weight", kv_rank_, (int64_t) v_mla_ * n_head_);
    B.wo = mat("attn_output.weight", (int64_t) n_head_ * v_mla_, n_embd_);
    load_moe(il, true);
}

// A row a position in every layer: the latent, then the turned key.  The values are the latent: no cache of their own.
void Mistral4::alloc_caches() {
    for (int il = 0; il < (int) blocks_.size(); ++il)
        if (runs(il)) blocks_[(size_t) il].kv = (uint16_t*) kv_alloc((size_t) row_ * sizeof(uint16_t), 1);
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

// Per-token buffers hold rows_ tokens, row after row.
void Mistral4::assign_scratch() {
    const size_t n = (size_t) rows_, h = (size_t) n_head_;
    t1_ = fbuf(n * q_rank_);
    qr_ = fbuf(n * q_rank_);
    q_ = fbuf(n * h * k_mla_);
    qn_ = fbuf(n * h * nope_);
    qabs_ = fbuf(n * h * kv_rank_);
    qc_ = fbuf(n * h * row_);
    kv_ = fbuf(n * row_);
    lat_ = fbuf(n * kv_rank_);
    kc_ = fbuf(n * row_);
    att_ = fbuf(n * h * row_);
    o_ = fbuf(n * h * kv_rank_);
    vo_ = fbuf(n * h * v_mla_);
    score_n_ = (size_t) chunk_ * n_head_ * opt_.max_context;
    score_ = fbuf(score_n_);
}

int Mistral4::widest() const { return std::max(q_rank_, n_head_ * v_mla_); }

size_t Mistral4::card_row_bytes() const {
    size_t layers = 0;
    for (const Block& B : blocks_)
        if (B.kv) ++layers;
    return layers * (size_t) row_ * sizeof(uint16_t);
}

void Mistral4::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t row = (size_t) row_ * sizeof(uint16_t);
    for (const Block& B : blocks_) {
        if (!B.kv || n == 0) continue;
        ck(cudaMemcpy(dst, (const uint8_t*) B.kv + (size_t) first * row, (size_t) n * row, cudaMemcpyDeviceToHost),
           "save rows");
        dst += (size_t) n * row;
    }
}

void Mistral4::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t row = (size_t) row_ * sizeof(uint16_t);
    for (const Block& B : blocks_) {
        if (!B.kv || n == 0) continue;
        ck(cudaMemcpy((uint8_t*) B.kv + (size_t) first * row, src + (size_t) (first - from) * row, (size_t) n * row,
                      cudaMemcpyHostToDevice),
           "load rows");
        src += (size_t) count * row;
    }
}

// The attention, as the reference's MLA path: the query's nope part through k_b into the latent's space, its turned
// part beside it; one key row a position that every head reads, [normed latent | turned key], and its latent the
// value; each head's output from the latent through v_b.  The dumps carry the reference's names where there is one.
void Mistral4::attn(Block& B, int il, int64_t pos, int n) {
    const size_t h = (size_t) n_head_, rows = (size_t) n * h;
    lin(B.q_a, xn_, t1_, n, xq_);
    rms_norm(t1_, fp(B.q_a_norm), qr_, q_rank_, n, rms_eps_, st_);
    lin(B.q_b, qr_, q_, n);
    dbg("q", il, q_, (int) rows * k_mla_);
    dsv4_rope(q_, q_, k_mla_, n_head_, nope_, n_rot_, rope_, pos, 1, n, false, 0.f, false, st_);
    dbg("q_pe", il, q_, (int) rows * k_mla_);   // the whole query, its pe values turned

    lin(B.kv_a, xn_, kv_, n, xq_);
    dbg("kv_cmpr_pe", il, kv_, n * row_);
    // kc_: the key turned into its last n_rot_ values, the latent normed into its first kv_rank_
    dsv4_rope(kv_, kc_, row_, 1, kv_rank_, n_rot_, rope_, pos, 1, n, false, 0.f, false, st_);
    rows2d(lat_, kv_rank_, kv_, row_, kv_rank_, n, st_);
    rms_norm(lat_, fp(B.kv_a_norm), kc_, kv_rank_, n, rms_eps_, st_, row_);
    dbg("Kcur", il, kc_, n * row_);

    rows2d(qn_, nope_, q_, k_mla_, nope_, rows, st_);
    lin(B.k_b, qn_, qabs_, n, nullptr, kv_rank_);
    dbg("q_nope_absorbed", il, qabs_, (int) rows * kv_rank_);
    rows2d(qc_, row_, qabs_, kv_rank_, kv_rank_, rows, st_);
    rows2d(qc_ + kv_rank_, row_, q_ + nope_, k_mla_, n_rot_, rows, st_);
    dbg("Qcur", il, qc_, (int) rows * row_);

    store_f16(kc_, B.kv, row_, pos, n, st_);
    // V is the cache's rows too: a head's first kv_rank_ outputs are the attention over the latents, and the n_rot_
    // after them (over the turned keys) are left
    attend(qc_, B.kv, B.kv, pos, n, score_, chunk_, att_, row_, n_head_, 1, kq_scale_, st_, 0, 0, false, nullptr, 0,
           score_n_);
    rows2d(o_, kv_rank_, att_, row_, kv_rank_, rows, st_);
    dbg("kqv", il, o_, (int) rows * kv_rank_);
    lin(B.v_b, o_, vo_, n, nullptr, v_mla_);
    dbg("kqv_v", il, vo_, (int) rows * v_mla_);
    lin(B.wo, vo_, ao_, n);
    dbg("kqv_out", il, ao_, n * n_embd_);
}

// A layer from its input to its router's copy.  A prompt batch's MoE (moe_rows) does its own routing.  A norm and
// its quantization are one launch, and so are the residual's sum and the next norm.
void Mistral4::layer_head(Layer& L, int il, int n) {
    if (rows_of_) throw std::runtime_error("mistral4: step()'s rows of several sequences are not supported");
    Block& B = blocks_[(size_t) il];
    const int threads = sum_threads(n_embd_);
    dbg(il == 0 ? "inp_embd" : "l_in", il, r_, n * n_embd_);
    kernels::native_add_rms_norm_q8_1(r_, nullptr, fp(L.attn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("attn_norm", il, xn_, n * n_embd_);
    attn(B, il, pos_, n);
    kernels::native_add_rms_norm_q8_1(r_, ao_, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("ffn_inp", il, r_, n * n_embd_);
    dbg("ffn_norm", il, xn_, n * n_embd_);
    if (!batched(L, n)) moe_router(L, n);
}

void Mistral4::layer_tail(Layer&, int il, int n) {
    dbg("ffn_out", il, fo_, n * n_embd_);
    axpy(fo_, 1.f, r_, n * n_embd_, true, st_);
    dbg("l_out", il, r_, n * n_embd_);
}

void Mistral4::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

void Mistral4::router_scores(const Layer& L, int n) {
    Engine::router_scores(L, n);
    dbg("ffn_moe_logits", (int) (&L - layers_.data()), rl_, n * n_expert_);
}

// What ggml's MoE graph does for this model: a softmax over every expert, the n_used likeliest (the first among
// equals), and their probabilities made to sum to one.
void Mistral4::route(int, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    float mx = logits[0];
    for (int e = 1; e < n_expert_; ++e) mx = std::max(mx, logits[e]);
    double all = 0.0;
    for (int e = 0; e < n_expert_; ++e) all += probs[e] = std::exp(logits[e] - mx);
    const float inv = (float) (1.0 / all);
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (probs[e] > probs[best]) best = e;
        ids[k] = best;
        w[k] = probs[best] * inv;
        sum += w[k];
        probs[best] = -1.f;
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f && w_scale_ != 1.f) w[k] *= w_scale_;
    }
}

// The shared expert, ungated: the routed experts' sum starts from its output.  Its down input has no guard, as in
// the reference (it is dumped, for the range checks).
void Mistral4::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_shexp_, clamp_shexp_[(size_t) il], st_);
    dbg("ffn_shexp_in", il, h_, n * n_ff_shexp_);
    lin(L.sh_down, h_, fo_, n);
    dbg("ffn_shexp", il, fo_, n * n_embd_);
}

}  // namespace strata::poly
