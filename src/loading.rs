use super::lazy::LazyTensor;
use super::modeling;
use std::path::Path;

macro_rules! disk_tensor {
    ($Root:expr, $($Part:expr),+) => {{
        let path = $Root $(.join($Part))+;
        assert!(path.is_file(), "{:?} is not a file", path);
        LazyTensor::Disk {
            path,
            shape: Default::default(),
        }
    }};
}

pub fn load_on_disk<P: AsRef<Path>>(root: P) -> modeling::LlamaForCausalLM {
    let variance_epsilon = 1e-6;
    let root = root.as_ref();
    let model = root.join("model");
    let layers = (0..modeling::NUM_LAYERS)
        .map(|i| model.join("layers").join(std::format!("{i}")))
        .map(|layer_root| modeling::DecoderLayer {
            self_attn: modeling::Attention {
                q_proj: disk_tensor!(layer_root, "self_attn", "q_proj", "weight"),
                k_proj: disk_tensor!(layer_root, "self_attn", "k_proj", "weight"),
                v_proj: disk_tensor!(layer_root, "self_attn", "v_proj", "weight"),
                out_proj: disk_tensor!(layer_root, "self_attn", "o_proj", "weight"),
                rotary_embed: modeling::RotaryEmbedding {
                    inv_freq: disk_tensor!(layer_root, "self_attn", "rotary_emb", "inv_freq"),
                },
            },
            mlp: modeling::MLP {
                gate_proj: disk_tensor!(layer_root, "mlp", "gate_proj", "weight"),
                down_proj: disk_tensor!(layer_root, "mlp", "down_proj", "weight"),
                up_proj: disk_tensor!(layer_root, "mlp", "up_proj", "weight"),
            },
            input_layer_norm: modeling::RMSNorm {
                weight: disk_tensor!(layer_root, "input_layernorm", "weight"),
                variance_epsilon,
            },
            post_attention_layer_norm: modeling::RMSNorm {
                weight: disk_tensor!(layer_root, "post_attention_layernorm", "weight"),
                variance_epsilon,
            },
        })
        .collect();
    modeling::LlamaForCausalLM {
        llama: modeling::Llama {
            embed_tokens: disk_tensor!(model, "embed_tokens", "weight"),
            layers,
            norm: modeling::RMSNorm {
                weight: disk_tensor!(model, "norm", "weight"),
                variance_epsilon,
            },
        },
        lm_head: disk_tensor!(root, "lm_head", "weight"),
    }
}
