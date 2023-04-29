mod lazy;
mod loading;
mod modeling;

use std::{io::Write, time::Instant};

use self::loading::load_on_disk;

use clap::Parser;
use dfdx::{shapes::Const, tensor::*, tensor_ops::*};
use rust_tokenizers::tokenizer::{SentencePieceBpeTokenizer, Tokenizer, TruncationStrategy};

/// Run text generation with the LLaMa 7b model
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct LlamaArgs {
    /// Root directory of the **converted** model. Use `convert.py` to convert the PyTorch `.bin` to
    /// the correct format.
    #[arg(short, long)]
    model: String,

    /// Number of new tokens to generate for each prompt.
    #[arg(short, long, default_value_t = 30)]
    generate: usize,

    /// Maximum number of **bytes** available in RAM to store model
    /// weights.
    ///
    /// This can be used in combination with --model-cuda-ram; You
    /// can have model weights loaded in both cpu and cuda.
    ///
    /// Default of 0 means **no** model weights will be loaded
    /// into RAM.
    #[arg(long, default_value_t = 0)]
    model_cpu_ram: usize,

    /// Maximum number of **bytes** available in CUDA memory
    /// to store model weights.
    ///
    /// This can be used in combination with --model-cpu-ram; You
    /// can have model weights loaded in both cpu and cuda.
    ///
    /// Default of 0 means **no** model weights will be loaded
    /// into CUDA.
    #[arg(long, default_value_t = 0)]
    model_cuda_ram: usize,
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

    let dev: AutoDevice = Default::default();

    let mut llama = load_on_disk(args.model.clone());

    #[cfg(feature = "cuda")]
    {
        let cpu: Cpu = Default::default();
        let unused_bytes = llama.maybe_load_on(args.model_cuda_ram, &dev);
        println!(
            "Used {} bytes of CUDA ram",
            args.model_cuda_ram - unused_bytes
        );
        let unused_bytes = llama.maybe_load_on(args.model_cpu_ram, &cpu);
        println!(
            "Used {} bytes of CPU ram",
            args.model_cpu_ram - unused_bytes
        );
    }

    #[cfg(not(feature = "cuda"))]
    {
        let unused_bytes = llama.maybe_load_on(args.model_cpu_ram, &dev);
        println!(
            "Used {} bytes of CPU ram",
            args.model_cpu_ram - unused_bytes
        );
    }

    let tokenizer =
        SentencePieceBpeTokenizer::from_file(args.model + "/tokenizer.model", false).unwrap();

    const BOS_TOKEN: usize = 0;
    const EOS_TOKEN: usize = 0;

    loop {
        let prompt = get_prompt_from_cli();
        let tokenized_input =
            tokenizer.encode_list(&[prompt], 128, &TruncationStrategy::LongestFirst, 0);

        let mut tokens: Vec<usize> = tokenized_input[0]
            .token_ids
            .iter()
            .map(|&x| x as usize)
            .collect();

        // BOS token, since SentencePieceBpeTokenizer doesn't add it
        tokens.insert(0, BOS_TOKEN);

        for _ in 0..args.generate {
            let start = Instant::now();
            let n_tokens = tokens.len();
            let input_ids = dev.tensor_from_vec(tokens.clone(), (Const::<1>, n_tokens));
            let logits = llama.forward(input_ids);
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
        println!("{:?}", tokenizer.decode(&tokens_i64, true, true));
    }
}
