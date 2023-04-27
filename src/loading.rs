use dfdx::tensor::{Cpu, Cuda};

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
    let model = root.join("model");
    let mut layers = Vec::new();
    for i in 0..modeling::NUM_LAYERS {
        let layer_root = model.join("layers").join(std::format!("{i}"));
        layers.push(modeling::DecoderLayer {
            self_attn: modeling::Attention {
                q_proj: disk_tensor!(layer_root.join("self_attn").join("q_proj").join("weight")),
                k_proj: disk_tensor!(layer_root.join("self_attn").join("k_proj").join("weight")),
                v_proj: disk_tensor!(layer_root.join("self_attn").join("v_proj").join("weight")),
                out_proj: disk_tensor!(layer_root
                    .join("self_attn")
                    .join("out_proj")
                    .join("weight")),
                rotary_embed: modeling::RotaryEmbedding {
                    inv_freq: disk_tensor!(layer_root
                        .join("self_attn")
                        .join("rotary_embed")
                        .join("inv_freq")),
                },
            },
            mlp: modeling::MLP {
                gate_proj: disk_tensor!(layer_root.join("mlp").join("gate_proj").join("weight")),
                down_proj: disk_tensor!(layer_root.join("mlp").join("down_proj").join("weight")),
                up_proj: disk_tensor!(layer_root.join("mlp").join("up_proj").join("weight")),
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
            embed_tokens: disk_tensor!(model.join("embed_tokens").join("weight")),
            layers,
            norm: modeling::RMSNorm {
                weight: disk_tensor!(model.join("norm").join("weight")),
                variance_epsilon,
            },
        },
        lm_head: disk_tensor!(root.join("lm_head")),
    }
}

pub fn load_into_cpu(llama: &mut modeling::LlamaForCausalLM, device: &Cpu) {
    todo!();
}

pub fn load_into_cuda(llama: &mut modeling::LlamaForCausalLM, device: &Cuda) {
    todo!();
}
