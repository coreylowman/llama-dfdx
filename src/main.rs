mod lazy;
mod loading;
mod modeling;

use std::io::Write;

use dfdx::{shapes::*, tensor::*, tensor_ops::*};
use loading::load_on_disk;
use modeling::LlamaForCausalLM;

fn get_prompt_from_cli() -> String {
    let mut user_input = String::new();
    std::print!("> ");
    std::io::stdout().flush().unwrap();
    std::io::stdin().read_line(&mut user_input).unwrap();
    user_input
}

fn main() {
    println!("Loading tokenizer...");
    println!("Downloading from huggingface...");
    println!("Loading model weights...");

    let root = "../llama-7b-bytes";
    let llama = load_on_disk(root);
    let dev: AutoDevice = Default::default();
    let input_ids: Tensor<Rank2<1, 10>, usize, _> = dev.zeros();
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
