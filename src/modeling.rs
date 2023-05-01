use super::lazy::LazyTensor;

use dfdx::{
    data::Arange,
    shapes::*,
    tensor::{AsArray, AutoDevice, Tensor, TriangleTensor, ZerosTensor},
    tensor_ops::*,
};

pub const VOCAB: usize = 32_000;
const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 11008;
const NUM_HEADS: usize = 32;
pub const NUM_LAYERS: usize = 32;
const HEAD_DIM: usize = HIDDEN / NUM_HEADS;
const HEAD_DIM_OVER_2: usize = HEAD_DIM / 2;

pub type Dev = AutoDevice;
pub type E = half::f16;

#[derive(Debug)]
pub struct Cache<Batch: Dim, Seq: Dim> {
    key: Tensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>), E, Dev>,
    val: Tensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>), E, Dev>,
}

#[derive(Debug)]
pub struct RMSNorm {
    pub weight: LazyTensor<(Const<HIDDEN>,), E>,
    pub variance_epsilon: f64,
}

impl RMSNorm {
    fn forward<Batch: Dim, Seq: Dim>(
        &self,
        x: Tensor<(Batch, Seq, Const<HIDDEN>), E, Dev>,
    ) -> Tensor<(Batch, Seq, Const<HIDDEN>), E, Dev> {
        let x_f32 = x.to_dtype::<f32>();
        let variance = x_f32.clone().square().mean::<_, Axis<2>>();
        let inv_std = (variance + self.variance_epsilon as f32).sqrt().recip();
        let x_f32 = inv_std.broadcast_like(&x_f32) * x_f32;
        self.weight.get_on(x_f32.device()).broadcast_like(&x_f32) * x_f32.to_dtype::<E>()
    }
}

#[derive(Debug)]
pub struct RotaryEmbedding {
    pub inv_freq: LazyTensor<Rank1<HEAD_DIM_OVER_2>, f32>,
}

impl RotaryEmbedding {
    fn forward<Batch: Dim, Seq: Dim>(
        &self,
        q: Tensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>), E, Dev>,
        k: Tensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>), E, Dev>,
        offset: usize,
    ) -> (
        Tensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>), E, Dev>,
        Tensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>), E, Dev>,
    ) {
        let (sin, cos) = self.get_sincos(q.device(), offset, q.shape().2);
        let sin = sin.broadcast_like(&q);
        let cos = cos.broadcast_like(&q);
        let q_embed = (q.clone() * cos.clone()) + (Self::rotate_half(q) * sin.clone());
        let k_embed = (k.clone() * cos) + (Self::rotate_half(k) * sin);
        (q_embed, k_embed)
    }

    fn get_sincos<Seq: Dim>(
        &self,
        device: &Dev,
        offset: usize,
        seq: Seq,
    ) -> (
        Tensor<(Seq, Const<HEAD_DIM>), E, Dev>,
        Tensor<(Seq, Const<HEAD_DIM>), E, Dev>,
    ) {
        let inv_freq = self.inv_freq.get_on(device);
        let t = device.arange(seq) + offset as f32;
        let freqs = t.matmul(inv_freq);
        let freqs = freqs.realize::<(Seq, usize)>();
        let emb = (freqs.clone(), freqs).concat_along(Axis::<1>);
        let emb_sin = emb.clone().sin();
        let emb_cos = emb.cos();
        (
            emb_sin.to_dtype::<E>().realize(),
            emb_cos.to_dtype::<E>().realize(),
        )
    }

    fn rotate_half<Batch: Dim, Seq: Dim>(
        x: Tensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>), E, Dev>,
    ) -> Tensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>), E, Dev> {
        let x1 = x.clone().slice((.., .., .., ..HEAD_DIM_OVER_2));
        let x2 = x.slice((.., .., .., HEAD_DIM_OVER_2..));
        (-x2, x1).concat_along(Axis::<3>).realize()
    }
}

#[derive(Debug)]
pub struct Attention {
    pub q_proj: LazyTensor<Rank2<HIDDEN, HIDDEN>, E>,
    pub k_proj: LazyTensor<Rank2<HIDDEN, HIDDEN>, E>,
    pub v_proj: LazyTensor<Rank2<HIDDEN, HIDDEN>, E>,
    pub out_proj: LazyTensor<Rank2<HIDDEN, HIDDEN>, E>,
    pub rotary_embed: RotaryEmbedding,
}

impl Attention {
    fn forward<Batch: Dim, CurSeq: Dim, PastSeq: Dim, TotSeq: Dim>(
        &self,
        x: Tensor<(Batch, CurSeq, Const<HIDDEN>), E, Dev>,
        attn_mask: Tensor<(CurSeq, TotSeq), E, Dev>,
        cache: Option<Cache<Batch, PastSeq>>,
    ) -> (
        Tensor<(Batch, CurSeq, Const<HIDDEN>), E, Dev>,
        Cache<Batch, TotSeq>,
    ) {
        let (batch, cur_seq, _) = *x.shape();
        let past_seq = cache
            .as_ref()
            .map(|c| c.key.shape().2.size())
            .unwrap_or_default();
        if cache.is_some() {
            assert_eq!(cur_seq.size(), 1);
        }
        let bsnh = (batch, cur_seq.size(), Const::<NUM_HEADS>, Const::<HEAD_DIM>);
        type Tr12 = Axes4<0, 2, 1, 3>;

        let q = {
            let q_proj = self.q_proj.get_on(x.device());
            let q = x.clone().matmul(q_proj.permute());
            q.reshape_like(&bsnh).unwrap().permute::<_, Tr12>()
        };

        let k = {
            let k_proj = self.k_proj.get_on(x.device());
            let k = x.clone().matmul(k_proj.permute());
            k.reshape_like(&bsnh).unwrap().permute::<_, Tr12>()
        };

        let v = {
            let v_proj = self.v_proj.get_on(x.device());
            let v = x.matmul(v_proj.permute());
            v.reshape_like(&bsnh).unwrap().permute::<_, Tr12>()
        };

        let (q, k) = self.rotary_embed.forward(q, k, past_seq);

        let q = q.realize::<(_, _, CurSeq, _)>();

        let (k, v) = if let Some(cache) = cache {
            let k = (cache.key.realize::<(_, _, usize, _)>(), k).concat_along(Axis::<2>);
            let v = (cache.val.realize::<(_, _, usize, _)>(), v).concat_along(Axis::<2>);
            (k.realize(), v.realize())
        } else {
            (k.realize(), v.realize())
        };

        let cache = Cache {
            key: k.clone(),
            val: v.clone(),
        };

        let inv_head_scale = (HEAD_DIM as f64).sqrt().recip() as f32;
        let attn_weights = q.matmul(k.permute()) * inv_head_scale;
        let attn_weights = attn_mask.broadcast_like(&attn_weights) + attn_weights;
        let attn_weights = attn_weights
            .to_dtype::<f32>()
            .softmax::<Axis<3>>()
            .to_dtype::<E>();

        let attn_output = attn_weights.matmul(v);
        let attn_output = attn_output
            .permute::<_, Tr12>()
            .reshape_like(&(batch, cur_seq, Const::<HIDDEN>))
            .unwrap();

        let out_proj = self.out_proj.get_on(attn_output.device());
        (attn_output.matmul(out_proj.permute()), cache)
    }
}

#[derive(Debug)]
pub struct MLP {
    pub gate_proj: LazyTensor<Rank2<INTERMEDIATE, HIDDEN>, E>,
    pub down_proj: LazyTensor<Rank2<HIDDEN, INTERMEDIATE>, E>,
    pub up_proj: LazyTensor<Rank2<INTERMEDIATE, HIDDEN>, E>,
}

impl MLP {
    fn forward<Batch: Dim, Seq: Dim>(
        &self,
        x: Tensor<(Batch, Seq, Const<HIDDEN>), E, Dev>,
    ) -> Tensor<(Batch, Seq, Const<HIDDEN>), E, Dev> {
        let up = {
            let up_proj = self.up_proj.get_on(x.device());
            x.clone().matmul(up_proj.permute())
        };
        let gate = {
            let gate_proj = self.gate_proj.get_on(x.device());
            x.matmul(gate_proj.permute())
        };
        let silu = up * (gate.clone() * gate.sigmoid());
        let down_proj = self.down_proj.get_on(silu.device());
        silu.matmul(down_proj.permute())
    }
}

#[derive(Debug)]
pub struct DecoderLayer {
    pub self_attn: Attention,
    pub mlp: MLP,
    pub input_layer_norm: RMSNorm,
    pub post_attention_layer_norm: RMSNorm,
}

impl DecoderLayer {
    fn forward<Batch: Dim, CurSeq: Dim, PastSeq: Dim, TotSeq: Dim>(
        &self,
        x: Tensor<(Batch, CurSeq, Const<HIDDEN>), E, Dev>,
        attn_mask: Tensor<(CurSeq, TotSeq), E, Dev>,
        cache: Option<Cache<Batch, PastSeq>>,
    ) -> (
        Tensor<(Batch, CurSeq, Const<HIDDEN>), E, Dev>,
        Cache<Batch, TotSeq>,
    ) {
        let residual = x.clone();
        let x = self.input_layer_norm.forward(x);
        let (x, cache) = self.self_attn.forward(x, attn_mask, cache);
        let x = residual + x;
        let residual = x.clone();
        let x = self.post_attention_layer_norm.forward(x);
        let x = self.mlp.forward(x);
        (x + residual, cache)
    }
}

#[derive(Debug)]
pub struct Llama {
    pub embed_tokens: LazyTensor<Rank2<VOCAB, HIDDEN>, E>,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
}

impl Llama {
    fn forward<Batch: Dim, CurSeq: Dim, PastSeq: Dim, TotSeq: Dim>(
        &self,
        input_ids: Tensor<(Batch, CurSeq), usize, Dev>,
        cache: Option<Vec<Cache<Batch, PastSeq>>>,
    ) -> (
        Tensor<(Batch, CurSeq, Const<HIDDEN>), E, Dev>,
        Vec<Cache<Batch, TotSeq>>,
    ) {
        let cur_seq = input_ids.shape().1;
        let past_seq = cache
            .as_ref()
            .map(|c| c[0].key.shape().2.size())
            .unwrap_or(0);
        let tot_seq = cur_seq.size() + past_seq;
        let dev = input_ids.device().clone();

        let mut attn_mask = dev
            .zeros_like(&(cur_seq, tot_seq))
            .realize::<(CurSeq, TotSeq)>();

        if cur_seq.size() > 1 {
            let causal_mask = dev.upper_tri_like(&(cur_seq, cur_seq.size()), E::MIN, 1);
            let causal_mask = if past_seq == 0 {
                causal_mask
            } else {
                (causal_mask, dev.zeros_like(&(cur_seq, past_seq))).concat_along(Axis::<1>)
            };
            attn_mask = causal_mask.realize();
        }

        let mut hidden_states = {
            let embed_tokens = self.embed_tokens.get_on(input_ids.device());
            embed_tokens.gather(input_ids)
        };
        let mut new_caches = Vec::with_capacity(self.layers.len());
        match cache {
            Some(cache) => {
                assert_eq!(cache.len(), self.layers.len());
                for (layer, cache_i) in self.layers.iter().zip(cache.into_iter()) {
                    let out = layer.forward(hidden_states, attn_mask.clone(), Some(cache_i));
                    hidden_states = out.0;
                    new_caches.push(out.1);
                }
            }
            None => {
                for layer in self.layers.iter() {
                    let out = layer.forward::<Batch, CurSeq, usize, TotSeq>(
                        hidden_states,
                        attn_mask.clone(),
                        None,
                    );
                    hidden_states = out.0;
                    new_caches.push(out.1);
                }
            }
        }
        (self.norm.forward(hidden_states), new_caches)
    }
}

#[derive(Debug)]
pub struct LlamaForCausalLM {
    pub llama: Llama,
    pub lm_head: LazyTensor<Rank2<VOCAB, HIDDEN>, E>,
}

impl LlamaForCausalLM {
    pub fn forward<Batch: Dim, CurSeq: Dim, PastSeq: Dim, TotSeq: Dim>(
        &self,
        input_ids: Tensor<(Batch, CurSeq), usize, Dev>,
        cache: Option<Vec<Cache<Batch, PastSeq>>>,
    ) -> (
        Tensor<(Batch, CurSeq, Const<VOCAB>), E, Dev>,
        Vec<Cache<Batch, TotSeq>>,
    ) {
        let (hidden_states, cache) = self.llama.forward(input_ids, cache);
        let lm_head = self.lm_head.get_on(hidden_states.device());
        (hidden_states.matmul(lm_head.permute()), cache)
    }
}
