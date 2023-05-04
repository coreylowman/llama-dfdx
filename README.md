# LLaMa 7b in rust

[![](https://dcbadge.vercel.app/api/server/AtUhGqBDP5)](https://discord.gg/AtUhGqBDP5)

This repo contains the popular [LLaMa 7b](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
language model, fully implemented in the rust programming language!

Uses [dfdx](https://github.com/coreylowman/dfdx) tensors and CUDA acceleration.

**This runs LLaMa directly in f16, meaning there is no hardware acceleration on CPU.** Using CUDA is heavily recommended.

# How To Run

## (Once) Setting up model weights

### Download model weights
1. Install git lfs. On ubuntu you can run `sudo apt install git-lfs`
2. Activate git lfs with `git lfs install`.
3. Run the following commands to download the model weights in pytorch format (~25 GB):
    a. LLaMa 7b (~25 GB): `git clone https://huggingface.co/decapoda-research/llama-7b-hf`
    b. LLaMa 13b (~75 GB): `git clone https://huggingface.co/decapoda-research/llama-13b-hf`
    c. LLaMa 65b (~XX GB): `git clone https://huggingface.co/decapoda-research/llama-65b-hf`

### Convert the model
1. (Optional) Run `python3.x -m venv <my_env_name>` to create a python virtual environment, where `x` is your prefered python version
2. (Optional, requires 1.) Run `source <my_env_name>\bin\activate` (or `<my_env_name>\Scripts\activate` if on Windows) to activate the environment
3. Run `pip install numpy torch`
4. Run `python convert.py` to convert the model weights to rust understandable format:
    a. LLaMa 7b: `python convert.py`
    b. LLaMa 13b: `python convert.py llama-13b-hf`
    c. LLaMa 65b: `python convert.py llama-65b-hf`

## (Once) Compile

You can compile with normal rust commands:

With cuda:
```bash
cargo build --release -F cuda
```

Without cuda:
```bash
cargo build --release
```

## Run the executable

With default args:
```bash
./target/release/llama-dfdx --model <model-dir> generate "<prompt>"
./target/release/llama-dfdx --model <model-dir> chat
./target/release/llama-dfdx --model <model-dir> file <path to prompt file>
```

To see what commands/custom args you can use:
```bash
./target/release/llama-dfdx --help
```
