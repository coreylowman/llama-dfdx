import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig, LlamaRMSNorm

cfg = LlamaConfig()

device = torch.device("cuda")

token_list = [0, 17166, 263, 4700, 508, 367, 2309, 297, 29871, 29896, 29900, 2560, 6576, 29901, 29871]

while True:
    tokens = torch.tensor(token_list, device=device).unsqueeze(0)
    position_ids = torch.arange(len(token_list), device=device).unsqueeze(0)

    # embedding
    sd_ends = torch.load("./llama-7b-hf/pytorch_model-00033-of-00033.bin")
    embed = torch.nn.Embedding(32000, 4096, padding_idx=0, device=device, dtype=torch.float16)
    embed.weight.data = sd_ends["model.embed_tokens.weight"].to(device)
    del sd_ends
    x = embed(tokens)
    del embed

    # decoder layers
    decoder = LlamaDecoderLayer(cfg).to(device=device)
    for i in range(32):
        sd = torch.load(f"./llama-7b-hf/pytorch_model-{i + 1:05}-of-00033.bin")
        decoder.load_state_dict({k.replace(f"model.layers.{i}.", ""): v for k, v in sd.items()})
        del sd
        x = decoder(x, position_ids=position_ids)[0]

    sd_ends = torch.load("./llama-7b-hf/pytorch_model-00033-of-00033.bin")

    norm = LlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(device)
    norm.weight.data = sd_ends["model.norm.weight"].to(device)
    x = norm(x)
    del norm

    lm_head = torch.nn.Linear(4096, 32000, bias=False, device=device, dtype=torch.float16)
    lm_head.weight.data = sd_ends["lm_head.weight"].to(device)
    x = lm_head(x)
    del lm_head

    del sd_ends

    vocab = x[:, -1, :]
    new_token = vocab.argmax(-1)
    print(new_token)
    token_list.append(new_token.item())