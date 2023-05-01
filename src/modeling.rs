use super::lazy::LazyTensor;

use dfdx::{data::Arange, shapes::*, tensor::Tensor, tensor_ops::*};

pub const VOCAB: usize = 32_000;
const HIDDEN: usize = 4096;
const INTERMEDIATE: usize = 11008;
const NUM_HEADS: usize = 32;
pub const NUM_LAYERS: usize = 32;
const HEAD_DIM: usize = HIDDEN / NUM_HEADS;
const HEAD_DIM_OVER_2: usize = HEAD_DIM / 2;

pub type E = half::f16;

#[derive(Debug)]
pub struct Cache<Batch: Dim, D: Device<E>> {
    key: Tensor<(Batch, Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, D>,
    val: Tensor<(Batch, Const<NUM_HEADS>, usize, Const<HEAD_DIM>), E, D>,
}

#[derive(Debug)]
pub struct RMSNorm {
    pub weight: LazyTensor<(Const<HIDDEN>,), E>,
    pub variance_epsilon: f64,
}

impl RMSNorm {
    fn forward<Batch: Dim, Seq: Dim, D: Device<E> + Device<f32>>(
        &self,
        x: Tensor<(Batch, Seq, Const<HIDDEN>), E, D>,
    ) -> Tensor<(Batch, Seq, Const<HIDDEN>), E, D> {
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
    fn forward<Batch: Dim, Seq: Dim, D: Device<E> + Device<f32>>(
        &self,
        q: Tensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>), E, D>,
        k: Tensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>), E, D>,
    ) -> (
        Tensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>), E, D>,
        Tensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>), E, D>,
    ) {
        let (sin, cos) = self.get_sincos(q.device(), q.shape().2);
        let sin = sin.broadcast_like(&q);
        let cos = cos.broadcast_like(&q);
        let q_embed = (q.clone() * cos.clone()) + (Self::rotate_half(q) * sin.clone());
        let k_embed = (k.clone() * cos) + (Self::rotate_half(k) * sin);
        (q_embed, k_embed)
    }

    fn get_sincos<Seq: Dim, D: Device<f32> + Device<E> + Arange<f32>>(
        &self,
        device: &D,
        seq: Seq,
    ) -> (
        Tensor<(Seq, Const<HEAD_DIM>), E, D>,
        Tensor<(Seq, Const<HEAD_DIM>), E, D>,
    ) {
        let inv_freq = self.inv_freq.get_on(device);
        let t = device.arange(seq);
        let freqs = t.matmul(inv_freq);
        let freqs = freqs.realize::<(usize, usize)>().unwrap();
        let emb = (freqs.clone(), freqs).concat_along(Axis::<1>);
        let emb_sin = emb.clone().sin();
        let emb_cos = emb.cos();
        (
            emb_sin.to_dtype::<E>().realize().unwrap(),
            emb_cos.to_dtype::<E>().realize().unwrap(),
        )
    }

    fn rotate_half<Batch: Dim, Seq: Dim, D: Device<E>>(
        x: Tensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>), E, D>,
    ) -> Tensor<(Batch, Const<NUM_HEADS>, Seq, Const<HEAD_DIM>), E, D> {
        let x1 = x.clone().slice((.., .., .., ..HEAD_DIM_OVER_2));
        let x2 = x.slice((.., .., .., HEAD_DIM_OVER_2..));
        (-x2, x1).concat_along(Axis::<3>).realize().unwrap()
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
    fn forward<Batch: Dim, Seq: Dim, D: Device<E> + Device<f32>>(
        &self,
        x: Tensor<(Batch, Seq, Const<HIDDEN>), E, D>,
        cache: Option<Cache<Batch, D>>,
    ) -> (Tensor<(Batch, Seq, Const<HIDDEN>), E, D>, Cache<Batch, D>) {
        let (batch, seq, _) = *x.shape();
        let bsnh = (batch, seq.size(), Const::<NUM_HEADS>, Const::<HEAD_DIM>);
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

        let mut v = {
            let v_proj = self.v_proj.get_on(x.device());
            let v = x.matmul(v_proj.permute());
            v.reshape_like(&bsnh).unwrap().permute::<_, Tr12>()
        };

        let (q, mut k) = self.rotary_embed.forward(q, k);

        if let Some(cache) = cache {
            k = (cache.key, k).concat_along(Axis::<2>);
            v = (cache.val, v).concat_along(Axis::<2>);
        }

        let cache = Cache {
            key: k.clone(),
            val: v.clone(),
        };

        let inv_head_scale = (HEAD_DIM as f64).sqrt().recip() as f32;
        let attn_weights = q.matmul(k.permute()) * inv_head_scale;
        let attn_weights = attn_weights
            .to_dtype::<f32>()
            .softmax::<Axis<3>>()
            .to_dtype::<E>();

        let attn_output = attn_weights.matmul(v);
        let attn_output = attn_output
            .permute::<_, Tr12>()
            .reshape_like(&(batch, seq, Const::<HIDDEN>))
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
    fn forward<Batch: Dim, Seq: Dim, D: Device<E>>(
        &self,
        x: Tensor<(Batch, Seq, Const<HIDDEN>), E, D>,
    ) -> Tensor<(Batch, Seq, Const<HIDDEN>), E, D> {
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
    fn forward<Batch: Dim, Seq: Dim, D: Device<E> + Device<f32>>(
        &self,
        x: Tensor<(Batch, Seq, Const<HIDDEN>), E, D>,
        cache: Option<Cache<Batch, D>>,
    ) -> (Tensor<(Batch, Seq, Const<HIDDEN>), E, D>, Cache<Batch, D>) {
        let residual = x.clone();
        let x = self.input_layer_norm.forward(x);
        let (x, cache) = self.self_attn.forward(x, cache);
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
    fn forward<Batch: Dim, Seq: Dim, D: Device<E> + Device<f32>>(
        &self,
        input_ids: Tensor<(Batch, Seq), usize, D>,
        cache: Option<Vec<Cache<Batch, D>>>,
    ) -> (
        Tensor<(Batch, Seq, Const<HIDDEN>), E, D>,
        Vec<Cache<Batch, D>>,
    ) {
        let mut hidden_states = {
            let embed_tokens = self.embed_tokens.get_on(input_ids.device());
            embed_tokens.gather(input_ids)
        };
        let mut caches = Vec::with_capacity(self.layers.len());
        match cache {
            Some(cache) => {
                assert_eq!(cache.len(), self.layers.len());
                for (layer, cache_i) in self.layers.iter().zip(cache.into_iter()) {
                    let out = layer.forward(hidden_states, Some(cache_i));
                    hidden_states = out.0;
                    caches.push(out.1);
                }
            }
            None => {
                for layer in self.layers.iter() {
                    let out = layer.forward(hidden_states, None);
                    hidden_states = out.0;
                    caches.push(out.1);
                }
            }
        }
        (self.norm.forward(hidden_states), caches)
    }
}

#[derive(Debug)]
pub struct LlamaForCausalLM {
    pub llama: Llama,
    pub lm_head: LazyTensor<Rank2<VOCAB, HIDDEN>, E>,
}

impl LlamaForCausalLM {
    pub fn forward<Batch: Dim, Seq: Dim, D: Device<E> + Device<f32>>(
        &self,
        input_ids: Tensor<(Batch, Seq), usize, D>,
        cache: Option<Vec<Cache<Batch, D>>>,
    ) -> (
        Tensor<(Batch, Seq, Const<VOCAB>), E, D>,
        Vec<Cache<Batch, D>>,
    ) {
        let (hidden_states, cache) = self.llama.forward(input_ids, cache);
        let lm_head = self.lm_head.get_on(hidden_states.device());
        (hidden_states.matmul(lm_head.permute()), cache)
    }
}
