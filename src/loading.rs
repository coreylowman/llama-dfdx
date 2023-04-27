use super::lazy::LazyTensor;
use super::modeling;
use std::path::Path;

macro_rules! disk_tensor {
    ($Path:expr) => {
        LazyTensor::Disk {
            path: $Path,
            shape: Default::default(),
        }
    };
}

pub fn load_on_disk<P: AsRef<Path>>(root: P) -> modeling::LlamaForCausalLM {
    let variance_epsilon = 1e-6;
    let root = root.as_ref();
    let mut layers = Vec::new();
    for i in 0..modeling::NUM_LAYERS {
        let layer_root = root.join(std::format!("{i}"));
        layers.push(modeling::DecoderLayer {
            self_attn: modeling::Attention {
                q_proj: disk_tensor!(layer_root.join("self_attn").join("q_proj")),
                k_proj: disk_tensor!(layer_root.join("self_attn").join("k_proj")),
                v_proj: disk_tensor!(layer_root.join("self_attn").join("v_proj")),
                out_proj: disk_tensor!(layer_root.join("self_attn").join("out_proj")),
                rotary_embed: modeling::RotaryEmbedding {
                    inv_freq: disk_tensor!(layer_root.join("self_attn").join("rotary_embed")),
                },
            },
            mlp: modeling::MLP {
                gate_proj: disk_tensor!(layer_root.join("mlp").join("gate_proj")),
                down_proj: disk_tensor!(layer_root.join("mlp").join("down_proj")),
                up_proj: disk_tensor!(layer_root.join("mlp").join("up_proj")),
            },
            input_layer_norm: modeling::RMSNorm {
                weight: disk_tensor!(layer_root.join("input_layer_norm").join("weight")),
                variance_epsilon,
            },
            post_attention_layer_norm: modeling::RMSNorm {
                weight: disk_tensor!(layer_root.join("post_attention_layer_norm").join("weight")),
                variance_epsilon,
            },
        });
    }
    modeling::LlamaForCausalLM {
        llama: modeling::Llama {
            embed_tokens: disk_tensor!(root.join("llama").join("embed_tokens")),
            layers,
            norm: modeling::RMSNorm {
                weight: disk_tensor!(root.join("llama").join("norm")),
                variance_epsilon,
            },
        },
        lm_head: disk_tensor!(root.join("lm_head")),
    }
}
