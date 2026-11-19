// src/poly/mistral4.hpp - Mistral Small 4 (llama.cpp architecture "mistral4"): the text decoder, 119B with 128
// routed experts a layer.
//
// Every layer attends with multi-head latent attention: low-rank queries, and a position keeps one row of the
// cache, a latent of kv_lora_rank values and a key of n_rot values turned by the position (YaRN), that every head
// reads.  The key's and the value's projections up from the latent are absorbed into each head's query and output,
// so the keys and the values stay latent.  Then 128 routed experts with softmax gates, four a token, beside one
// shared expert.  llama.cpp runs it with its deepseek2 graph (src/models/deepseek2.cpp, c7518dbe), the reference.
#pragma once

#include "engine.hpp"

namespace strata::poly {

class Mistral4 final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override { return 0; }
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    bool checked_dumps() const override { return true; }
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    void router_scores(const Layer& L, int n) override;
    void moe_shared(const Layer& L, int il, int n) override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override { (void) il, (void) pos, (void) dst; }

private:
    /// A layer's attention weights (Layer has its norms and its experts), and its cache.
    struct Block {
        Dev q_a, q_a_norm, q_b, kv_a, kv_a_norm, wo;
        Dev k_b, v_b;              ///< per head, as floats: nope -> latent, and latent -> value
        uint16_t* kv = nullptr;    ///< a row a position, fp16: the latent (kv_rank_ values), then the turned key (n_rot_)
    };

    void attn(Block& B, int il, int64_t pos, int n);

    std::vector<Block> blocks_;   ///< one a layer of layers_
    int n_head_ = 0, q_rank_ = 0, kv_rank_ = 0, n_rot_ = 0;
    int k_mla_ = 0, v_mla_ = 0;   ///< a head's query (nope and turned part) and its value
    int nope_ = 0;                ///< a head's query values that are not turned
    int row_ = 0;                 ///< a cache row: kv_rank_ + n_rot_, which is also an absorbed query's width
    bool w_norm_ = false;         ///< the picked experts' weights are made to sum to one
    float w_scale_ = 0;           ///< and then scaled by this, when it is not 0 or 1
    float kq_scale_ = 0;          ///< the softmax's scale: YaRN's mscale squared over sqrt(k_mla_), as the reference's
    Rope rope_;
    int chunk_ = 1;               ///< tokens the attention takes at a time

    // device scratch, rows_ tokens each
    float *t1_ = nullptr, *qr_ = nullptr, *q_ = nullptr, *qn_ = nullptr, *qabs_ = nullptr, *qc_ = nullptr;
    float *kv_ = nullptr, *lat_ = nullptr, *kc_ = nullptr, *att_ = nullptr, *o_ = nullptr, *vo_ = nullptr;
    float* score_ = nullptr;
    size_t score_n_ = 0;   ///< the floats of score_
};

}  // namespace strata::poly
