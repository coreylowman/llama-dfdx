mod lazy;
mod loading;
mod modeling;

use std::io::Write;

use self::loading::load_on_disk;

use clap::Parser;
use dfdx::{
    shapes::{Const, HasShape},
    tensor::*,
    tensor_ops::*,
};
use rust_tokenizers::tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy};

const MB: usize = 1_000_000;

/// Run text generation with the LLaMa 7b model
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct LlamaArgs {
    /// Root directory of the **converted** model. Use `convert.py` to convert the PyTorch `.bin` to
    /// the correct format.
    #[arg(short, long, default_value = "llama-7b-hf")]
    model: String,

    /// Number of new tokens to generate for each prompt.
    #[arg(short, long, default_value_t = 10)]
    num_tokens: usize,

    /// Maximum number of **megabytes** available
    /// to store *model* weights.
    #[arg(long, default_value = None)]
    ram_limit_for_model: Option<usize>,

    /// Disable the KV cache. This will slow computations down,
    /// but reduce memory usage.
    #[arg(long, default_value_t = false)]
    disable_cache: bool,
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

    let dev: modeling::Dev = Default::default();

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

    const BOS_TOKEN: usize = 0;
    const EOS_TOKEN: usize = 1;

    loop {
        let prompt = get_prompt_from_cli();
        let prompt = prompt.trim_end();
        let tokenized_input =
            tokenizer.encode_list(&[prompt], 128, &TruncationStrategy::LongestFirst, 0);

        let mut tokens: Vec<usize> = tokenized_input[0]
            .token_ids
            .iter()
            .map(|&x| x as usize)
            .collect();

        // BOS token, since SentencePieceBpeTokenizer doesn't add it
        tokens.insert(0, BOS_TOKEN);

        let mut cache: Option<Vec<modeling::Cache<Const<1>, usize>>> = None;

        let mut output: String = prompt.into();

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
            let logits = vocab.as_vec();
            let new_token = logits
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .map(|x| x.0)
                .unwrap();
            tokens.push(new_token);
            if new_token == EOS_TOKEN {
                break;
            }

            let new_content = tokenizer.decode(&[new_token as i64], false, false);
            output.push_str(&new_content);
            print!("{}", if i_token == 0 { &output } else { &new_content });
            std::io::stdout().flush().unwrap();
        }
        println!();
    }
}
