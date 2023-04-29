# LLaMa in dfdx

This repo contains the popular [LLaMa 7b](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/)
language model, implemented in [dfdx](https://github.com/coreylowman/dfdx).

# How To Run

## (Once) Setting up model weights

1. Install git lfs. On ubuntu you can run `sudo apt install git-lfs`
2. Activate git lfs with `git lfs install`.
3. Run `bash download.sh` to download the model weights in pytorch format
4. Run `python convert.py` to convert the model weights to rust understandable format

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

```bash
./target/release/llama-dfdx --model llama-7b-hf
```
