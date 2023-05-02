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
            move_to_ram: false,
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
                o_proj: disk_tensor!(layer_root, "self_attn", "o_proj", "weight"),
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

macro_rules! maybe_mark {
    ($MaxBytes:tt, $Field:expr) => {
        if $MaxBytes >= $Field.num_bytes() && $Field.is_on_disk() {
            $Field.defer_load();
            $MaxBytes -= $Field.num_bytes();
        }
    };
}

impl super::modeling::RMSNorm {
    pub fn num_bytes(&self) -> usize {
        self.weight.num_bytes()
    }

    pub fn maybe_mark_for_ram(&mut self, mut max_bytes: usize) -> usize {
        maybe_mark!(max_bytes, self.weight);
        max_bytes
    }
}

impl super::modeling::RotaryEmbedding {
    pub fn num_bytes(&self) -> usize {
        self.inv_freq.num_bytes()
    }

    pub fn maybe_mark_for_ram(&mut self, mut max_bytes: usize) -> usize {
        maybe_mark!(max_bytes, self.inv_freq);
        max_bytes
    }
}

impl super::modeling::Attention {
    pub fn num_bytes(&self) -> usize {
        self.q_proj.num_bytes()
            + self.k_proj.num_bytes()
            + self.v_proj.num_bytes()
            + self.o_proj.num_bytes()
            + self.rotary_embed.num_bytes()
    }

    pub fn maybe_mark_for_ram(&mut self, mut max_bytes: usize) -> usize {
        maybe_mark!(max_bytes, self.q_proj);
        maybe_mark!(max_bytes, self.k_proj);
        maybe_mark!(max_bytes, self.v_proj);
        maybe_mark!(max_bytes, self.o_proj);
        self.rotary_embed.maybe_mark_for_ram(max_bytes)
    }
}

impl super::modeling::MLP {
    pub fn num_bytes(&self) -> usize {
        self.gate_proj.num_bytes() + self.down_proj.num_bytes() + self.up_proj.num_bytes()
    }

    pub fn maybe_mark_for_ram(&mut self, mut max_bytes: usize) -> usize {
        maybe_mark!(max_bytes, self.gate_proj);
        maybe_mark!(max_bytes, self.down_proj);
        maybe_mark!(max_bytes, self.up_proj);
        max_bytes
    }
}

impl super::modeling::DecoderLayer {
    pub fn num_bytes(&self) -> usize {
        self.self_attn.num_bytes()
            + self.mlp.num_bytes()
            + self.input_layer_norm.num_bytes()
            + self.post_attention_layer_norm.num_bytes()
    }

    pub fn maybe_mark_for_ram(&mut self, mut max_bytes: usize) -> usize {
        max_bytes = self.self_attn.maybe_mark_for_ram(max_bytes);
        max_bytes = self.mlp.maybe_mark_for_ram(max_bytes);
        max_bytes = self.input_layer_norm.maybe_mark_for_ram(max_bytes);
        max_bytes = self.post_attention_layer_norm.maybe_mark_for_ram(max_bytes);
        max_bytes
    }
}

impl super::modeling::Llama {
    pub fn num_bytes(&self) -> usize {
        self.embed_tokens.num_bytes()
            + self.layers.iter().map(|l| l.num_bytes()).sum::<usize>()
            + self.norm.num_bytes()
    }

    pub fn maybe_mark_for_ram(&mut self, mut max_bytes: usize) -> usize {
        maybe_mark!(max_bytes, self.embed_tokens);
        for layer in self.layers.iter_mut() {
            max_bytes = layer.maybe_mark_for_ram(max_bytes);
        }
        self.norm.maybe_mark_for_ram(max_bytes)
    }
}

impl super::modeling::LlamaForCausalLM {
    pub fn num_bytes(&self) -> usize {
        self.llama.num_bytes() + self.lm_head.num_bytes()
    }

    pub fn deferred_load(&mut self, mut max_bytes: usize) -> usize {
        maybe_mark!(max_bytes, self.lm_head);
        self.llama.maybe_mark_for_ram(max_bytes)
    }
}
