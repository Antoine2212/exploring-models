from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = Path("data/cache")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "qwen25_gqa_cache.pt"

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "I'm testing attention heads in a small transformer model.",
    "def add(a, b):\n    return a + b\n",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def to_cpu(x: torch.Tensor | None) -> torch.Tensor | None:
    return x.detach().to("cpu") if x is not None else None


def reshape_last_dim(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    bsz, seq_len, _ = x.shape
    return x.view(bsz, seq_len, n_heads, head_dim)


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=dtype,
    attn_implementation="eager",   # important pour output_attentions
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()

cfg = model.config
layers = model.model.layers

num_layers = cfg.num_hidden_layers
num_q_heads = cfg.num_attention_heads
num_kv_heads = cfg.num_key_value_heads
head_dim = cfg.hidden_size // cfg.num_attention_heads
num_kv_groups = num_q_heads // num_kv_heads

payload: dict[str, Any] = {
    "model_id": MODEL_ID,
    "config": cfg.to_dict(),
    "meta": {
        "num_layers": num_layers,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "num_kv_groups": num_kv_groups,
        "head_dim": head_dim,
        "dtype": str(dtype),
        "device": str(model.device),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
    },
    "prompts": [],
}

with torch.no_grad():
    for prompt_id, text in enumerate(PROMPTS):
        toks = tokenizer(text, return_tensors="pt")
        input_ids = toks["input_ids"].to(model.device)
        attention_mask = toks["attention_mask"].to(model.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        past_key_values = outputs.past_key_values

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        prompt_record: dict[str, Any] = {
            "prompt_id": prompt_id,
            "text": text,
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "tokens": tokens,
            "attentions": [to_cpu(a) for a in attentions],  # [layer][bsz, q_heads, tgt, src]
            "cache": [],
            "layers": [],
        }

        for layer_idx, layer in enumerate(layers):
            h = hidden_states[layer_idx].to(layer.self_attn.q_proj.weight.device)

            q = layer.self_attn.q_proj(h)
            k = layer.self_attn.k_proj(h)
            v = layer.self_attn.v_proj(h)

            q = reshape_last_dim(q, num_q_heads, head_dim)
            k = reshape_last_dim(k, num_kv_heads, head_dim)
            v = reshape_last_dim(v, num_kv_heads, head_dim)

            kv_head_for_q = torch.arange(num_q_heads) // num_kv_groups

            layer_record = {
                "q_pre_rope": q.cpu(),              # [bsz, seq, q_heads, head_dim]
                "k_pre_rope": k.cpu(),              # [bsz, seq, kv_heads, head_dim]
                "v_pre_rope": v.cpu(),              # [bsz, seq, kv_heads, head_dim]
                "kv_head_for_q": kv_head_for_q,     # [q_heads]
            }
            prompt_record["layers"].append(layer_record)

            if past_key_values is not None:
                # DynamicCache moderne : cache stocké dans .layers[idx].keys/.values
                pk = past_key_values.layers[layer_idx].keys   # [bsz, kv_heads, seq, head_dim]
                pv = past_key_values.layers[layer_idx].values
                prompt_record["cache"].append({
                    "k": to_cpu(pk),  # usuellement [bsz, kv_heads, seq, head_dim]
                    "v": to_cpu(pv),
                })
            else:
                prompt_record["cache"].append(None)

        payload["prompts"].append(prompt_record)

torch.save(payload, OUT_PATH)
print(f"Saved cache to: {OUT_PATH}")