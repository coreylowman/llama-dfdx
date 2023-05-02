use std::ops::ControlFlow;

use rand::rngs::StdRng;
use rust_tokenizers::tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy};

use dfdx::{
    shapes::{Axis, Const, HasShape},
    tensor::*,
    tensor_ops::*,
};

use super::sampling;

pub struct LlamaArgs {
    /// Seed for device & sampling RNG.
    pub seed: u64,

    /// Root directory of the **converted** model. Use `convert.py` to convert the PyTorch `.bin` to
    /// the correct format.
    pub model: String,

    /// Number of new tokens to generate for each prompt.
    pub num_tokens: usize,

    /// Maximum number of **megabytes** available
    /// to store *model* weights.
    pub ram_limit_for_model: Option<usize>,

    /// Disable the KV cache. This will slow computations down,
    /// but reduce memory usage.
    pub disable_cache: bool,

    /// Whether to do greedy sampling or top_p sampling.
    pub greedy: bool,

    /// The top probability value when using non-greedy sampling
    pub top_p: f32,

    /// The temperature value when using non-greedy sampling
    pub temperature: f32,

    /// The number of tokens to consider when using non-greedy sampling.
    pub top_k: usize,
}

pub struct LlamaPipeline {
    pub args: LlamaArgs,
    pub device: super::modeling::Dev,
    pub llama: super::modeling::LlamaForCausalLM,
    pub tokenizer: SentencePieceBpeTokenizer,
    pub bos_token: usize,
    pub eos_token: usize,
    pub rng: StdRng,
}

impl LlamaPipeline {
    pub fn new(args: LlamaArgs) -> Self {
        todo!()
    }

    pub fn generate(
        &mut self,
        prompt: String,
        num_tokens: usize,
    ) -> impl Iterator<Item = String> + '_ {
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

        let start = std::time::Instant::now();

        (0..num_tokens)
            .map(|i_token| {
                let n_tokens = tokens.len();
                let input_ids = match cache.as_ref() {
                    None => self
                        .device
                        .tensor_from_vec(tokens.clone(), (Const::<1>, n_tokens)),
                    Some(_) => self.device.tensor([[*tokens.last().unwrap()]]).realize(),
                };
                let seq_len = input_ids.shape().1;
                let out = self.llama.forward(input_ids, cache);
                let logits = out.0;
                cache = (!self.args.disable_cache).then(|| out.1);
                let vocab = logits.select(self.device.tensor([seq_len - 1]));
                let new_token = if self.args.greedy {
                    sampling::greedy(vocab.to_dtype::<f32>().as_vec())
                } else {
                    let probs =
                        (vocab.to_dtype::<f32>() / self.args.temperature).softmax::<Axis<1>>();
                    sampling::top_p(
                        probs.as_vec(),
                        self.args.top_p,
                        self.args.top_k,
                        &mut self.rng,
                    )
                };

                if new_token == self.eos_token {
                    std::ops::ControlFlow::Break(())
                } else if new_token == 13 {
                    std::ops::ControlFlow::Continue("\n".into())
                } else {
                    std::ops::ControlFlow::Continue(self.tokenizer.decode(
                        &[new_token as i64],
                        true,
                        false,
                    ))
                }
            })
            .map_while(|item| match item {
                ControlFlow::Break(()) => None,
                ControlFlow::Continue(item) => Some(item),
            })
    }
}
