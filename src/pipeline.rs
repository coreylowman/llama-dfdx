use rand::{rngs::StdRng, SeedableRng};
use rust_tokenizers::tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy};

use dfdx::{
    shapes::{Axis, Const, HasShape},
    tensor::*,
    tensor_ops::*,
};

use super::sampling;

#[derive(Debug, Clone, Copy)]
pub struct LlamaConfig {
    pub num_tokens: usize,
    pub disable_cache: bool,
    pub greedy: bool,
    pub top_p: f32,
    pub temperature: f32,
    pub top_k: usize,
}

pub struct LlamaPipeline {
    pub cfg: LlamaConfig,
    pub device: super::modeling::Dev,
    pub llama: super::modeling::LlamaForCausalLM,
    pub tokenizer: SentencePieceBpeTokenizer,
    pub bos_token: usize,
    pub eos_token: usize,
    pub rng: StdRng,
}

impl LlamaPipeline {
    pub fn new(
        model: String,
        ram_limit_for_model: Option<usize>,
        seed: u64,
        cfg: LlamaConfig,
    ) -> Self {
        const MB: usize = 1_000_000;

        let rng = StdRng::seed_from_u64(seed);
        let device = super::modeling::Dev::seed_from_u64(seed);

        let mut llama = super::loading::load_on_disk(model.clone());
        let model_num_bytes = llama.num_bytes();
        println!("Model size: {} MB", model_num_bytes / MB);

        let max_bytes = ram_limit_for_model
            .map(|x| x * MB)
            .unwrap_or(model_num_bytes);
        let unused_bytes = llama.deferred_load(max_bytes);
        println!(
            "{} MB of model parameters will be held in RAM.",
            (max_bytes - unused_bytes) / MB
        );

        let tokenizer =
            SentencePieceBpeTokenizer::from_file(model + "/tokenizer.model", false).unwrap();

        Self {
            cfg,
            device,
            llama,
            tokenizer,
            bos_token: 1,
            eos_token: 2,
            rng,
        }
    }

    pub fn generate<Prompt: Into<String>>(
        &mut self,
        prompt: Prompt,
    ) -> impl Iterator<Item = String> + '_ {
        let prompt = prompt.into();

        let tokenized_input = self.tokenizer.encode_list(
            &[prompt.clone()],
            prompt.len(),
            &TruncationStrategy::LongestFirst,
            0,
        );

        let mut tokens: Vec<usize> = tokenized_input[0]
            .token_ids
            .iter()
            .map(|&x| x as usize)
            .collect();

        // BOS token, since SentencePieceBpeTokenizer doesn't add it
        tokens.insert(0, self.bos_token);

        let mut cache: Option<Vec<super::modeling::Cache<Const<1>, usize>>> = None;

        let cfg = self.cfg;
        (0..cfg.num_tokens)
            .map(move |_| {
                let n_tokens = tokens.len();
                let input_ids = match cache.as_ref() {
                    None => self
                        .device
                        .tensor_from_vec(tokens.clone(), (Const::<1>, n_tokens)),
                    Some(_) => self.device.tensor([[*tokens.last().unwrap()]]).realize(),
                };
                let seq_len = input_ids.shape().1;
                let out = self.llama.forward(input_ids, cache.clone());
                let logits = out.0;
                cache = (!cfg.disable_cache).then_some(out.1);
                let vocab = logits.select(self.device.tensor([seq_len - 1]));
                let new_token = if cfg.greedy {
                    sampling::greedy(vocab.to_dtype::<f32>().as_vec())
                } else {
                    let probs = (vocab.to_dtype::<f32>() / cfg.temperature).softmax::<Axis<1>>();
                    sampling::top_p(probs.as_vec(), cfg.top_p, cfg.top_k, &mut self.rng)
                };

                tokens.push(new_token);

                if new_token == self.eos_token {
                    None
                } else if new_token == 13 {
                    Some("\n".into())
                } else {
                    Some(self.tokenizer.decode(&[new_token as i64], true, false))
                }
            })
            .take_while(Option::is_some)
            .map(Option::unwrap)
    }
}
