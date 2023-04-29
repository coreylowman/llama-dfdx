import torch
from transformers import LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaConfig, LlamaRMSNorm

cfg = LlamaConfig(bos_token_id=0, eos_token_id=1)
print(cfg)

device = torch.device("cuda")

temperature = 0.0
top_p = 0.95

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

inputs = tokenizer("The capital of Canada is", return_tensors="pt", add_special_tokens=True)
token_list = list(map(int, inputs.input_ids[0]))
print(token_list)

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
    if temperature > 0:
        probs = (vocab / temperature).softmax(-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = (probs_sum - probs_sort) > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        new_token = torch.multinomial(probs_sort, num_samples=1)
        new_token = torch.gather(probs_idx, -1, new_token)
    else:
        new_token = vocab.argmax(-1)
    print(new_token)
    token_list.append(new_token.item())
    print(tokenizer.batch_decode([token_list[1:]]))