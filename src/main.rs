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
    generate: usize,

    /// Maximum number of **megabytes** available in RAM to store model
    /// weights.
    ///
    /// This can be used in combination with --model-cuda-ram; You
    /// can have model weights loaded in both cpu and cuda.
    ///
    /// Value of 0 means **no** model weights will be loaded
    /// into RAM.
    #[arg(long, default_value_t = 0)]
    model_cpu_mb: usize,

    /// Maximum number of **megabytes** available in CUDA memory
    /// to store model weights.
    ///
    /// This can be used in combination with --model-cpu-ram; You
    /// can have model weights loaded in both cpu and cuda.
    ///
    /// Value of 0 means **no** model weights will be loaded
    /// into CUDA.
    #[arg(long, default_value_t = 13476)]
    model_cuda_mb: usize,

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
    println!("Model size: {} MB", llama.num_bytes() / MB);

    let cpu_bytes = args.model_cpu_mb * MB;

    #[cfg(feature = "cuda")]
    {
        let cuda_bytes = args.model_cuda_mb * MB;
        println!("Loading model weights into CUDA...");
        let unused_bytes = llama.maybe_load_on(cuda_bytes, &dev);
        println!("Used {} MB of CUDA ram", (cuda_bytes - unused_bytes) / MB);
    }

    {
        println!("Loading model weights into CPU...");
        let cpu: Cpu = Default::default();
        let unused_bytes = llama.maybe_load_on(cpu_bytes, &cpu);
        println!("Used {} MB of CPU ram", (cpu_bytes - unused_bytes) / MB);
    }

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

        for i_token in 0..args.generate {
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
