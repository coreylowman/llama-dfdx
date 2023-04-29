mod lazy;
mod loading;
mod modeling;

use std::{io::Write, time::Instant};

use self::loading::load_on_disk;

use clap::Parser;
use dfdx::{shapes::Const, tensor::*, tensor_ops::*};
use rust_tokenizers::tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Root directory of the **converted** model. Use `convert.py` to convert the PyTorch `.bin` to
    /// the correct format.
    #[arg(short, long)]
    model: String,

    /// Number of new tokens to generate for each prompt.
    #[arg(short, long, default_value_t = 30)]
    generate: usize,
}

fn get_prompt_from_cli() -> String {
    let mut user_input = String::new();
    std::print!("Enter prompt > ");
    std::io::stdout().flush().unwrap();
    std::io::stdin().read_line(&mut user_input).unwrap();
    user_input
}

fn main() {
    let args = Args::parse();

    let dev: AutoDevice = Default::default();
    let llama = load_on_disk(args.model.clone());
    let tokenizer =
        SentencePieceBpeTokenizer::from_file(args.model + "/tokenizer.model", false).unwrap();

    const BOS_TOKEN: usize = 0;
    const EOS_TOKEN: usize = 0;

    loop {
        let prompt = get_prompt_from_cli();
        let tokenized_input =
            tokenizer.encode_list(&[prompt], 128, &TruncationStrategy::LongestFirst, 0);
        println!("Tokenized: {:?}", tokenized_input);

        let mut tokens: Vec<usize> = tokenized_input[0]
            .token_ids
            .iter()
            .map(|&x| x as usize)
            .collect();

        // BOS token, since SentencePieceBpeTokenizer doesn't add it
        tokens.push(BOS_TOKEN);

        for _ in 0..args.generate {
            let start = Instant::now();
            let n_tokens = tokens.len();
            let input_ids = dev.tensor_from_vec(tokens.clone(), (Const::<1>, n_tokens));
            let logits = llama.forward(input_ids.to_dtype::<usize>());
            let vocab = logits.select(dev.tensor([n_tokens - 1]));
            let logits = vocab.as_vec();
            let new_token = logits
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                .map(|x| x.0)
                .unwrap();
            tokens.push(new_token);
            println!("Generated token {new_token} in {:?}", start.elapsed());
            if new_token == EOS_TOKEN {
                break;
            }
        }

        let tokens_i64: Vec<i64> = tokens.iter().map(|&x| x as i64).collect();
        println!("{:?}", tokenizer.decode(&tokens_i64, false, true));
    }
}
