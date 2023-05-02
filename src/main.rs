mod lazy;
mod loading;
mod modeling;
mod sampling;

use std::io::Write;

use self::loading::load_on_disk;

use clap::Parser;
use dfdx::{
    shapes::{Axis, Const, HasShape},
    tensor::*,
    tensor_ops::*,
};
use rand::{rngs::StdRng, SeedableRng};
use rust_tokenizers::tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy};

const MB: usize = 1_000_000;

/// Run text generation with the LLaMa 7b model
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct LlamaArgs {
    /// Seed for device & sampling RNG.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Root directory of the **converted** model. Use `convert.py` to convert the PyTorch `.bin` to
    /// the correct format.
    #[arg(short, long, default_value = "llama-7b-hf")]
    model: String,

    /// Number of new tokens to generate for each prompt.
    #[arg(short, long, default_value_t = 512)]
    num_tokens: usize,

    /// Maximum number of **megabytes** available
    /// to store *model* weights.
    #[arg(long, default_value = None)]
    ram_limit_for_model: Option<usize>,

    /// Disable the KV cache. This will slow computations down,
    /// but reduce memory usage.
    #[arg(long, default_value_t = false)]
    disable_cache: bool,

    /// Whether to do greedy sampling or top_p sampling with
    /// top_p = `0.95`, and temperature = `0.8`.
    #[arg(long, default_value_t = false)]
    greedy: bool,

    /// The top probability value when using non-greedy sampling
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// The temperature value when using non-greedy sampling
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// The number of tokens to consider when using non-greedy sampling.
    #[arg(long, default_value_t = modeling::VOCAB)]
    top_k: usize,
}

fn get_prompt_from_cli() -> String {
    let mut user_input = String::new();
    std::print!("Enter prompt > ");
    std::io::stdout().flush().unwrap();
    std::io::stdin().read_line(&mut user_input).unwrap();
    user_input
}

fn main() {
    let args = LlamaArgs::parse();

    let mut rng = StdRng::seed_from_u64(args.seed);
    let dev: modeling::Dev = modeling::Dev::seed_from_u64(args.seed);

    let mut llama = load_on_disk(args.model.clone());
    let model_num_bytes = llama.num_bytes();
    println!("Model size: {} MB", model_num_bytes / MB);

    let max_bytes = args
        .ram_limit_for_model
        .map(|x| x * MB)
        .unwrap_or(model_num_bytes);
    let unused_bytes = llama.deferred_load(max_bytes);
    println!(
        "{} MB of model parameters will be held in RAM.",
        (max_bytes - unused_bytes) / MB
    );

    let tokenizer =
        SentencePieceBpeTokenizer::from_file(args.model + "/tokenizer.model", false).unwrap();

    const BOS_TOKEN: usize = 1;
    const EOS_TOKEN: usize = 2;
    const NEWLINE_TOKEN: usize = 13;

    loop {
        let prompt = get_prompt_from_cli();
        let prompt = prompt.trim_end();
        let tokenized_input = tokenizer.encode_list(
            &[prompt],
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
        tokens.insert(0, BOS_TOKEN);

        let mut cache: Option<Vec<modeling::Cache<Const<1>, usize>>> = None;

        let mut output: String = prompt.into();

        let start = std::time::Instant::now();

        for i_token in 0..args.num_tokens {
            let n_tokens = tokens.len();
            let input_ids = match cache.as_ref() {
                None => dev.tensor_from_vec(tokens.clone(), (Const::<1>, n_tokens)),
                Some(_) => dev.tensor([[*tokens.last().unwrap()]]).realize(),
            };
            let seq_len = input_ids.shape().1;
            let out = llama.forward(input_ids, cache);
            let logits = out.0;
            cache = (!args.disable_cache).then(|| out.1);
            let vocab = logits.select(dev.tensor([seq_len - 1]));
            let new_token = if args.greedy {
                sampling::greedy(vocab.to_dtype::<f32>().as_vec())
            } else {
                let probs = (vocab.to_dtype::<f32>() / args.temperature).softmax::<Axis<1>>();
                sampling::top_p(probs.as_vec(), args.top_p, args.top_k, &mut rng)
            };

            tokens.push(new_token);

            if new_token == EOS_TOKEN {
                break;
            }

            let new_content = if new_token == NEWLINE_TOKEN {
                "\n".into()
            } else {
                tokenizer.decode(&[new_token as i64], true, false)
            };
            output.push_str(&new_content);
            print!("{}", if i_token == 0 { &output } else { &new_content });
            std::io::stdout().flush().unwrap();
        }

        let elapsed = start.elapsed();
        let tokens_per_s = args.num_tokens as f64 / elapsed.as_secs_f64();

        println!(
            "\nGenerated {} tokens in {:.3?}. {tokens_per_s:.3} tokens/s",
            args.num_tokens, elapsed
        );
    }
}
