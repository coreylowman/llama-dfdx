mod lazy;
mod loading;
mod modeling;
mod pipeline;
mod sampling;

use std::io::Write;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(ValueEnum, Debug, Clone)]
enum Structure {
    Auto,
    Llama7b,
    Llama13b,
    Llama65b,
}

impl std::fmt::Display for Structure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => f.write_str("auto"),
            Self::Llama7b => f.write_str("llama-7b"),
            Self::Llama13b => f.write_str("llama-13b"),
            Self::Llama65b => f.write_str("llama-65b"),
        }
    }
}

/// Run text generation with the LLaMa 7b model
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Seed for device & sampling RNG.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Root directory of the **converted** model. Use `convert.py` to convert the PyTorch `.bin` to
    /// the correct format.
    #[arg(short, long, default_value = "llama-7b-hf")]
    model: String,

    /// Number of new tokens to generate for each prompt.
    #[arg(short, long, default_value_t = 128)]
    num_tokens: usize,

    /// Maximum number of **megabytes** available
    /// to store *model* weights.
    #[arg(long, default_value = None)]
    ram_limit_for_model: Option<usize>,

    /// Disable the KV cache. This will slow computations down,
    /// but reduce memory usage.
    #[arg(long, default_value_t = false)]
    disable_cache: bool,

    /// Specify to do greedy sampling instead of top-p sampling.
    #[arg(long, default_value_t = false)]
    greedy: bool,

    /// The top probability value when using non-greedy sampling
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// The temperature value when using non-greedy sampling
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,

    /// The number of tokens to consider when using non-greedy sampling.
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// The structure of the model. "auto" will attempt to auto detect
    /// the structure based on the contents of the `--model` directory.
    #[arg(long, default_value_t = Structure::Auto)]
    structure: Structure,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Chat back and forth interactively with the model using
    /// stdio.
    Chat,

    /// Generates text with a prompt from the CLI.
    Generate {
        /// The text to begin generation from.
        prompt: String,
    },

    /// Generates text with a prompt in a file.
    File {
        /// The path to the file containing prompts.
        path: String,
    },
}

fn main() {
    let args = Cli::parse();

    match args.structure {
        Structure::Auto => {
            let num_bins = std::fs::read_dir(&args.model)
                .expect("Model directory does not exist.")
                .filter(|path| path.as_ref().unwrap().path().ends_with(".bin"))
                .count();
            if num_bins == 33 {
                println!("Detected model folder as LLaMa 7b.");
                run::<modeling::Llama7b>(args);
            } else if num_bins == 41 {
                println!("Detected model folder as LLaMa 13b.");
                run::<modeling::Llama13b>(args);
            } else if num_bins == 81 {
                println!("Detected model folder as LLaMa 65b.");
                run::<modeling::Llama65b>(args);
            } else {
                panic!(
                    "Found {num_bins} .bin files in the model directory. Expected 33, 41, or 81."
                );
            }
        }
        Structure::Llama7b => run::<modeling::Llama7b>(args),
        Structure::Llama13b => run::<modeling::Llama13b>(args),
        Structure::Llama65b => run::<modeling::Llama65b>(args),
    }
}

fn run<M: modeling::LlamaModel>(args: Cli) {
    let mut pipeline = pipeline::LlamaPipeline::<M>::new(
        args.model,
        args.ram_limit_for_model,
        args.seed,
        pipeline::LlamaConfig {
            num_tokens: args.num_tokens,
            disable_cache: args.disable_cache,
            greedy: args.greedy,
            top_p: args.top_p,
            temperature: args.temperature,
            top_k: args.top_k,
        },
    );

    match args.command {
        Commands::Generate { prompt } => {
            print!("{prompt}");
            for new_content in pipeline.generate(prompt) {
                print!("{new_content}");
                std::io::stdout().flush().unwrap();
            }
            println!();
        }
        Commands::File { path } => {
            let prompt = std::fs::read_to_string(path).unwrap();
            print!("{prompt}");
            for new_content in pipeline.generate(prompt) {
                print!("{new_content}");
                std::io::stdout().flush().unwrap();
            }
            println!();
        }
        Commands::Chat => {
            let mut conversation = String::new();
            loop {
                let mut prompt = String::new();
                std::print!("> ");
                std::io::stdout().flush().unwrap();
                std::io::stdin().read_line(&mut prompt).unwrap();
                let prompt = prompt.trim_end();

                conversation.push_str(prompt);

                let start = std::time::Instant::now();

                let mut num_tokens_generated = 0;
                for new_content in pipeline.generate(prompt) {
                    print!("{}", &new_content);
                    conversation.push_str(&new_content);
                    std::io::stdout().flush().unwrap();
                    num_tokens_generated += 1;
                }
                println!();

                let elapsed = start.elapsed();
                let elapsed_s = elapsed.as_secs_f64();
                let tokens_per_s = num_tokens_generated as f64 / elapsed_s;
                let ms_per_token = 1000.0 * elapsed_s / num_tokens_generated as f64;

                println!(
                    "\n*Generated {} tokens in {:.3?} ({tokens_per_s:.3} tokens/s, {ms_per_token:.0} ms/token)*",
                    num_tokens_generated, elapsed
                );
            }
        }
    }
}
