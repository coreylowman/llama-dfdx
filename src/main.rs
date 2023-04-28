mod lazy;
mod loading;
mod modeling;

use dfdx::{tensor::*, tensor_ops::*};
use loading::load_on_disk;

use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of the person to greet
    #[arg(short, long)]
    model_root: String,

    #[arg(short, long, default_value_t = 30)]
    num_tokens: usize,
}

fn main() {
    let args = Args::parse();
    let llama = load_on_disk(args.model_root);
    let dev: AutoDevice = Default::default();
    let input_ids = dev.tensor([[0, 1000, 2000, 3000, 4000]]);
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
