use dfdx::tensor_ops::Device;

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

impl super::modeling::RMSNorm {
    pub fn maybe_load_on<D: Device<super::modeling::E>>(
        &mut self,
        mut max_bytes: usize,
        device: &D,
    ) -> usize {
        if max_bytes >= self.weight.num_bytes() && self.weight.is_on_disk() {
            self.weight.load_on(device);
            max_bytes -= self.weight.num_bytes();
        }
        max_bytes
    }
}

impl super::modeling::RotaryEmbedding {
    pub fn maybe_load_on<D: Device<f32>>(&mut self, mut max_bytes: usize, device: &D) -> usize {
        if max_bytes >= self.inv_freq.num_bytes() && self.inv_freq.is_on_disk() {
            self.inv_freq.load_on(device);
            max_bytes -= self.inv_freq.num_bytes();
        }
        max_bytes
    }
}

impl super::modeling::Attention {
    pub fn maybe_load_on<D: Device<super::modeling::E> + Device<f32>>(
        &mut self,
        mut max_bytes: usize,
        device: &D,
    ) -> usize {
        if max_bytes >= self.q_proj.num_bytes() && self.q_proj.is_on_disk() {
            self.q_proj.load_on(device);
            max_bytes -= self.q_proj.num_bytes();
        }
        if max_bytes >= self.k_proj.num_bytes() && self.k_proj.is_on_disk() {
            self.k_proj.load_on(device);
            max_bytes -= self.k_proj.num_bytes();
        }
        if max_bytes >= self.v_proj.num_bytes() && self.v_proj.is_on_disk() {
            self.v_proj.load_on(device);
            max_bytes -= self.v_proj.num_bytes();
        }
        if max_bytes >= self.out_proj.num_bytes() && self.out_proj.is_on_disk() {
            self.out_proj.load_on(device);
            max_bytes -= self.out_proj.num_bytes();
        }
        self.rotary_embed.maybe_load_on(max_bytes, device)
    }
}

impl super::modeling::MLP {
    pub fn maybe_load_on<D: Device<super::modeling::E>>(
        &mut self,
        mut max_bytes: usize,
        device: &D,
    ) -> usize {
        if max_bytes >= self.gate_proj.num_bytes() && self.gate_proj.is_on_disk() {
            self.gate_proj.load_on(device);
            max_bytes -= self.gate_proj.num_bytes();
        }
        if max_bytes >= self.down_proj.num_bytes() && self.down_proj.is_on_disk() {
            self.down_proj.load_on(device);
            max_bytes -= self.down_proj.num_bytes();
        }
        if max_bytes >= self.up_proj.num_bytes() && self.up_proj.is_on_disk() {
            self.up_proj.load_on(device);
            max_bytes -= self.up_proj.num_bytes();
        }
        max_bytes
    }
}

impl super::modeling::DecoderLayer {
    pub fn maybe_load_on<D: Device<super::modeling::E> + Device<f32>>(
        &mut self,
        mut max_bytes: usize,
        device: &D,
    ) -> usize {
        max_bytes = self.self_attn.maybe_load_on(max_bytes, device);
        max_bytes = self.mlp.maybe_load_on(max_bytes, device);
        max_bytes = self.input_layer_norm.maybe_load_on(max_bytes, device);
        max_bytes = self
            .post_attention_layer_norm
            .maybe_load_on(max_bytes, device);
        max_bytes
    }
}

impl super::modeling::Llama {
    pub fn maybe_load_on<D: Device<super::modeling::E> + Device<f32>>(
        &mut self,
        mut max_bytes: usize,
        device: &D,
    ) -> usize {
        if max_bytes >= self.embed_tokens.num_bytes() && self.embed_tokens.is_on_disk() {
            self.embed_tokens.load_on(device);
            max_bytes -= self.embed_tokens.num_bytes();
        }
        for layer in self.layers.iter_mut() {
            max_bytes = layer.maybe_load_on(max_bytes, device);
        }
        self.norm.maybe_load_on(max_bytes, device)
    }
}

impl super::modeling::LlamaForCausalLM {
    pub fn maybe_load_on<D: Device<super::modeling::E> + Device<f32>>(
        &mut self,
        mut max_bytes: usize,
        device: &D,
    ) -> usize {
        if max_bytes >= self.lm_head.num_bytes() && self.lm_head.is_on_disk() {
            self.lm_head.load_on(device);
            max_bytes -= self.lm_head.num_bytes();
        }
        self.llama.maybe_load_on(max_bytes, device)
    }
}
