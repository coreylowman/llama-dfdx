#![allow(clippy::type_complexity)]

use super::lazy::LazyTensor;

use dfdx::{
    data::Arange,
    dtypes::*,
    shapes::*,
    tensor::{AutoDevice, Tensor, TriangleTensor, ZerosTensor},
    tensor_ops::*,
};

pub const VOCAB: usize = 32_000;
pub const HEAD_DIM: usize = 128;
pub const HEAD_DIM_OVER_2: usize = 64;

pub trait LlamaModel: Clone {
    type Hidden: ConstDim;
    type Intermediate: ConstDim;
    type NumHeads: ConstDim;
    const NUM_LAYERS: usize;
}

#[derive(Debug, Clone, Copy)]
pub struct Llama7b;
impl LlamaModel for Llama7b {
    type Hidden = Const<4096>;
    type Intermediate = Const<11008>;
    type NumHeads = Const<32>;
    const NUM_LAYERS: usize = 32;
}

#[derive(Debug, Clone, Copy)]
pub struct Llama13b;
impl LlamaModel for Llama13b {
    type Hidden = Const<5120>;
    type Intermediate = Const<13824>;
    type NumHeads = Const<40>;
    const NUM_LAYERS: usize = 40;
}

#[derive(Debug, Clone, Copy)]
pub struct Llama65b;
impl LlamaModel for Llama65b {
    type Hidden = Const<8192>;
    type Intermediate = Const<22016>;
    type NumHeads = Const<64>;
    const NUM_LAYERS: usize = 80;
}

pub type Dev = AutoDevice;

#[derive(Debug, Clone)]
pub struct Cache<Batch: Dim, Seq: Dim, M: LlamaModel> {
    pub key: Tensor<(Batch, M::NumHeads, Seq, Const<HEAD_DIM>), f16, Dev>,
    pub val: Tensor<(Batch, M::NumHeads, Seq, Const<HEAD_DIM>), f16, Dev>,
}

#[derive(Debug)]
pub struct RMSNorm<M: LlamaModel> {
    pub weight: LazyTensor<(M::Hidden,), f16>,
    pub variance_epsilon: f64,
}

impl<M: LlamaModel> RMSNorm<M> {
    fn forward<Batch: Dim, Seq: Dim>(
        &mut self,
        x: Tensor<(Batch, Seq, M::Hidden), f16, Dev>,
    ) -> Tensor<(Batch, Seq, M::Hidden), f16, Dev> {
        let x_f32 = x.to_dtype::<f32>();
        let var_f32 = x_f32.clone().square().mean::<_, Axis<2>>();
        let inv_std_f32 = (var_f32 + self.variance_epsilon as f32).sqrt().recip();
        let x_f32 = inv_std_f32.broadcast_like(&x_f32) * x_f32;
        self.weight.get_on(x_f32.device()).broadcast_like(&x_f32) * x_f32.to_dtype::<f16>()
    }
}

#[derive(Debug)]
pub struct RotaryEmbedding {
    pub inv_freq: LazyTensor<Rank1<HEAD_DIM_OVER_2>, f32>,
}

impl RotaryEmbedding {
    fn forward<Batch: Dim, NumHeads: Dim, Seq: Dim>(
        &mut self,
        q: Tensor<(Batch, NumHeads, Seq, Const<HEAD_DIM>), f16, Dev>,
        k: Tensor<(Batch, NumHeads, Seq, Const<HEAD_DIM>), f16, Dev>,
        offset: usize,
    ) -> (
        Tensor<(Batch, NumHeads, Seq, Const<HEAD_DIM>), f16, Dev>,
        Tensor<(Batch, NumHeads, Seq, Const<HEAD_DIM>), f16, Dev>,
    ) {
        let (sin, cos) = self.get_sincos(q.device(), offset, q.shape().2);
        let sin = sin.broadcast_like(&q);
        let cos = cos.broadcast_like(&q);
        let q_embed = (Self::rotate_half(q.clone()) * sin.clone()) + (q * cos.clone());
        let k_embed = (Self::rotate_half(k.clone()) * sin) + (k * cos);
        (q_embed, k_embed)
    }

    fn get_sincos<Seq: Dim>(
        &mut self,
        device: &Dev,
        offset: usize,
        seq: Seq,
    ) -> (
        Tensor<(Seq, Const<HEAD_DIM>), f16, Dev>,
        Tensor<(Seq, Const<HEAD_DIM>), f16, Dev>,
    ) {
        let inv_freq = self.inv_freq.get_on(device);
        let t = device.arange(seq) + offset as f32;
        let freqs = t.matmul(inv_freq);
        let freqs = freqs.realize::<(Seq, usize)>();
        let emb = (freqs.clone(), freqs).concat_along(Axis::<1>);
        let emb_sin = emb.clone().sin();
        let emb_cos = emb.cos();
        (
            emb_sin.to_dtype::<f16>().realize(),
            emb_cos.to_dtype::<f16>().realize(),
        )
    }

    fn rotate_half<Batch: Dim, NumHeads: Dim, Seq: Dim>(
        x: Tensor<(Batch, NumHeads, Seq, Const<HEAD_DIM>), f16, Dev>,
    ) -> Tensor<(Batch, NumHeads, Seq, Const<HEAD_DIM>), f16, Dev> {
        let x1 = x.clone().slice((.., .., .., ..HEAD_DIM_OVER_2));
        let x2 = x.slice((.., .., .., HEAD_DIM_OVER_2..));
        (-x2, x1).concat_along(Axis::<3>).realize()
    }
}

#[derive(Debug)]
pub struct Attention<M: LlamaModel> {
    pub q_proj: LazyTensor<(M::Hidden, M::Hidden), f16>,
    pub k_proj: LazyTensor<(M::Hidden, M::Hidden), f16>,
    pub v_proj: LazyTensor<(M::Hidden, M::Hidden), f16>,
    pub o_proj: LazyTensor<(M::Hidden, M::Hidden), f16>,
    pub rotary_embed: RotaryEmbedding,
}

impl<M: LlamaModel> Attention<M> {
    fn forward<Batch: Dim, CurSeq: Dim, PastSeq: Dim, TotSeq: Dim>(
        &mut self,
        x: Tensor<(Batch, CurSeq, M::Hidden), f16, Dev>,
        attn_mask: Tensor<(CurSeq, TotSeq), f16, Dev>,
        cache: Option<Cache<Batch, PastSeq, M>>,
    ) -> (
        Tensor<(Batch, CurSeq, M::Hidden), f16, Dev>,
        Cache<Batch, TotSeq, M>,
    ) {
        let shape = *x.shape();
        let num_heads = M::NumHeads::default();
        let past_seq = cache.as_ref().map(|c| c.key.shape().2.size()).unwrap_or(0);
        let bsnh = (shape.0, shape.1.size(), num_heads, Const::<HEAD_DIM>);
        type Tr12 = Axes4<0, 2, 1, 3>;

        let q = {
            let q_proj = self.q_proj.get_on(x.device());
            let q = x.clone().matmul(q_proj.permute());
            q.reshape_like(&bsnh).permute::<_, Tr12>()
        };

        let k = {
            let k_proj = self.k_proj.get_on(x.device());
            let k = x.clone().matmul(k_proj.permute());
            k.reshape_like(&bsnh).permute::<_, Tr12>()
        };

        let v = {
            let v_proj = self.v_proj.get_on(x.device());
            let v = x.matmul(v_proj.permute());
            v.reshape_like(&bsnh).permute::<_, Tr12>()
        };

        let (q, k) = self.rotary_embed.forward(q, k, past_seq);
        let q = q.realize::<(_, _, CurSeq, _)>();

        let (k, v) = if let Some(cache) = cache {
            // for concat we need to make the dimension usize if we don't want to use nightly
            let k = (cache.key.realize::<(_, _, usize, _)>(), k).concat_along(Axis::<2>);
            let v = (cache.val.realize::<(_, _, usize, _)>(), v).concat_along(Axis::<2>);
            (k.realize(), v.realize())
        } else {
            (k.realize(), v.realize())
        };

        // save the concat'd key and value for the next forward pass
        // NOTE: k has already had RoPe applied
        let cache = Cache {
            key: k.clone(),
            val: v.clone(),
        };

        let inv_head_scale = (HEAD_DIM as f64).sqrt().recip() as f32;
        let w = q.matmul(k.permute()) * inv_head_scale;
        let w = attn_mask.broadcast_like(&w) + w;
        let w = w.to_dtype::<f32>().softmax::<Axis<3>>().to_dtype::<f16>();

        let o = w.matmul(v);
        let o = o.permute::<_, Tr12>().reshape_like(&shape);

        let out_proj = self.o_proj.get_on(o.device());
        (o.matmul(out_proj.permute()), cache)
    }
}

#[derive(Debug)]
pub struct Mlp<M: LlamaModel> {
    pub gate_proj: LazyTensor<(M::Intermediate, M::Hidden), f16>,
    pub down_proj: LazyTensor<(M::Hidden, M::Intermediate), f16>,
    pub up_proj: LazyTensor<(M::Intermediate, M::Hidden), f16>,
}

impl<M: LlamaModel> Mlp<M> {
    fn forward<Batch: Dim, Seq: Dim>(
        &mut self,
        x: Tensor<(Batch, Seq, M::Hidden), f16, Dev>,
    ) -> Tensor<(Batch, Seq, M::Hidden), f16, Dev> {
        let gate = {
            let gate_proj = self.gate_proj.get_on(x.device());
            let gate = x.clone().matmul(gate_proj.permute());
            gate.clone().sigmoid() * gate
        };
        let up = {
            let up_proj = self.up_proj.get_on(x.device());
            let up = x.matmul(up_proj.permute());
            up * gate
        };
        let down_proj = self.down_proj.get_on(up.device());
        up.matmul(down_proj.permute())
    }
}

#[derive(Debug)]
pub struct DecoderLayer<M: LlamaModel> {
    pub self_attn: Attention<M>,
    pub mlp: Mlp<M>,
    pub input_layer_norm: RMSNorm<M>,
    pub post_attention_layer_norm: RMSNorm<M>,
}

impl<M: LlamaModel> DecoderLayer<M> {
    fn forward<Batch: Dim, CurSeq: Dim, PastSeq: Dim, TotSeq: Dim>(
        &mut self,
        x: Tensor<(Batch, CurSeq, M::Hidden), f16, Dev>,
        attn_mask: Tensor<(CurSeq, TotSeq), f16, Dev>,
        cache: Option<Cache<Batch, PastSeq, M>>,
    ) -> (
        Tensor<(Batch, CurSeq, M::Hidden), f16, Dev>,
        Cache<Batch, TotSeq, M>,
    ) {
        let residual = x.clone();
        let x = self.input_layer_norm.forward(x);
        let (x, cache) = self.self_attn.forward(x, attn_mask, cache);
        let x = x + residual;
        let residual = x.clone();
        let x = self.post_attention_layer_norm.forward(x);
        let x = self.mlp.forward(x);
        (x + residual, cache)
    }
}

#[derive(Debug)]
pub struct Llama<M: LlamaModel> {
    pub embed_tokens: LazyTensor<(Const<VOCAB>, M::Hidden), f16>,
    pub layers: Vec<DecoderLayer<M>>,
    pub norm: RMSNorm<M>,
}

impl<M: LlamaModel> Llama<M> {
    fn forward<Batch: Dim, CurSeq: Dim, PastSeq: Dim, TotSeq: Dim>(
        &mut self,
        input_ids: Tensor<(Batch, CurSeq), usize, Dev>,
        cache: Option<Vec<Cache<Batch, PastSeq, M>>>,
    ) -> (
        Tensor<(Batch, CurSeq, M::Hidden), f16, Dev>,
        Vec<Cache<Batch, TotSeq, M>>,
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
            let causal_mask = dev.upper_tri_like(&(cur_seq, cur_seq.size()), f16::MIN, 1);
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
        let cache = cache
            .map(|cs| cs.into_iter().map(Some).collect())
            .unwrap_or_else(|| vec![None; self.layers.len()]);
        assert_eq!(cache.len(), self.layers.len());
        for (layer_i, cache_i) in self.layers.iter_mut().zip(cache.into_iter()) {
            let out = layer_i.forward(hidden_states, attn_mask.clone(), cache_i);
            hidden_states = out.0;
            new_caches.push(out.1);
        }
        (self.norm.forward(hidden_states), new_caches)
    }
}

#[derive(Debug)]
pub struct LlamaForCausalLM<M: LlamaModel> {
    pub llama: Llama<M>,
    pub lm_head: LazyTensor<(Const<VOCAB>, M::Hidden), f16>,
}

impl<M: LlamaModel> LlamaForCausalLM<M> {
    pub fn forward<Batch: Dim, CurSeq: Dim, PastSeq: Dim, TotSeq: Dim>(
        &mut self,
        input_ids: Tensor<(Batch, CurSeq), usize, Dev>,
        cache: Option<Vec<Cache<Batch, PastSeq, M>>>,
    ) -> (
        Tensor<(Batch, CurSeq, Const<VOCAB>), f16, Dev>,
        Vec<Cache<Batch, TotSeq, M>>,
    ) {
        let (hidden_states, cache) = self.llama.forward(input_ids, cache);
        let lm_head = self.lm_head.get_on(hidden_states.device());
        (hidden_states.matmul(lm_head.permute()), cache)
    }
}
