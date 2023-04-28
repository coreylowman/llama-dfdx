mod lazy;
mod loading;
mod modeling;

use std::io::Write;

use dfdx::{shapes::*, tensor::*, tensor_ops::*};
use loading::load_on_disk;
use modeling::LlamaForCausalLM;

use tokenizers::tokenizer::{Result, Tokenizer};

fn get_prompt_from_cli() -> String {
    let mut user_input = String::new();
    std::print!("> ");
    std::io::stdout().flush().unwrap();
    std::io::stdin().read_line(&mut user_input).unwrap();
    user_input
}

fn main() {
    let dev: AutoDevice = Default::default();
    let tokenizer = Tokenizer::from_file("../llama-7b-hf/tokenizer_config.json").unwrap();
    let llama = load_on_disk("../llama-7b-bytes");

    let encoding = tokenizer.encode("Hey there!", false).unwrap();
    let input_ids: Vec<usize> = encoding.get_ids().iter().map(|&x| x as usize).collect();
    let seq_len = input_ids.len();
    let input_ids = dev.tensor_from_vec(input_ids, (Const::<1>, seq_len));

    let logits = llama.forward(input_ids);
    let vocab = logits.select(dev.tensor([9]));
    let logits = vocab.as_vec();
    let new_token = logits
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
        .map(|x| x.0)
        .unwrap();
    println!("{new_token}");
}
