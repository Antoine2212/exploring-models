from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT_DIR = Path("data/cache")
OUT_PATH = OUT_DIR / "chatmode" / "qwen25_full_block_cache.pt"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    [{"role": "user", "content": "The quick brown fox jumps over the lazy dog."}],
    [{"role": "user", "content": "I'm testing attention heads in a small transformer model."}],
    [{"role": "user", "content": "def add(a, b):\n    return a + b\n"}],
]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def to_cpu(x: torch.Tensor | None) -> torch.Tensor | None:
    return x.detach().to("cpu") if x is not None else None


def reshape_q(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    bsz, seq_len, _ = x.shape
    return x.view(bsz, seq_len, n_heads, head_dim)


def reshape_kv(x: torch.Tensor, n_kv_heads: int, head_dim: int) -> torch.Tensor:
    bsz, seq_len, _ = x.shape
    return x.view(bsz, seq_len, n_kv_heads, head_dim)


def repeat_kv(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    # x: [bsz, kv_heads, seq, head_dim] -> [bsz, q_heads, seq, head_dim]
    bsz, kv_heads, seq_len, head_dim = x.shape
    if num_groups == 1:
        return x
    x = x[:, :, None, :, :].expand(bsz, kv_heads, num_groups, seq_len, head_dim)
    return x.reshape(bsz, kv_heads * num_groups, seq_len, head_dim)


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    attn_implementation="eager",
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()

cfg = model.config
layers = model.model.layers

num_layers = cfg.num_hidden_layers
num_q_heads = cfg.num_attention_heads
num_kv_heads = cfg.num_key_value_heads
hidden_size = cfg.hidden_size
head_dim = hidden_size // num_q_heads
num_kv_groups = num_q_heads // num_kv_heads

payload: dict[str, Any] = {
    "model_id": MODEL_ID,
    "config": cfg.to_dict(),
    "meta": {
        "num_layers": num_layers,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "num_kv_groups": num_kv_groups,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
        "dtype": str(dtype),
        "device": str(model.device),
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
    },
    "prompts": [],
}

with torch.no_grad():
    for prompt_id, message in enumerate(PROMPTS):
        text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        toks = tokenizer(text, return_tensors="pt")
        input_ids = toks["input_ids"].to(model.device)
        attention_mask = toks["attention_mask"].to(model.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        prompt_record: dict[str, Any] = {
            "prompt_id": prompt_id,
            "text": text,
            "input_ids": input_ids.cpu(),
            "attention_mask": attention_mask.cpu(),
            "tokens": tokens,
            "attentions_hf": [to_cpu(a) for a in attentions],  # [layer][bsz, q_heads, tgt, src]
            "layers": [],
        }

        for layer_idx, layer in enumerate(layers):
            block_in = hidden_states[layer_idx].to(layer.self_attn.q_proj.weight.device)   # [bsz, seq, hidden]
            block_out = hidden_states[layer_idx + 1].to(layer.self_attn.q_proj.weight.device)

            # 1) input RMSNorm
            attn_norm_out = layer.input_layernorm(block_in)  # [bsz, seq, hidden]

            # 2) projections Q/K/V before RoPE
            q_pre = layer.self_attn.q_proj(attn_norm_out)    # [bsz, seq, q_heads*head_dim]
            k_pre = layer.self_attn.k_proj(attn_norm_out)    # [bsz, seq, kv_heads*head_dim]
            v_pre = layer.self_attn.v_proj(attn_norm_out)    # [bsz, seq, kv_heads*head_dim]

            q_pre = reshape_q(q_pre, num_q_heads, head_dim)      # [bsz, seq, q_heads, head_dim]
            k_pre = reshape_kv(k_pre, num_kv_heads, head_dim)    # [bsz, seq, kv_heads, head_dim]
            v_pre = reshape_kv(v_pre, num_kv_heads, head_dim)    # [bsz, seq, kv_heads, head_dim]

            # Transformers attention code uses [bsz, heads, seq, head_dim]
            q_t = q_pre.transpose(1, 2).contiguous()   # [bsz, q_heads, seq, head_dim]
            k_t = k_pre.transpose(1, 2).contiguous()   # [bsz, kv_heads, seq, head_dim]
            v_t = v_pre.transpose(1, 2).contiguous()   # [bsz, kv_heads, seq, head_dim]

            # 3) RoPE
            # On suit l’implémentation HF du module Qwen2Attention
            # position ids
            position_ids = torch.arange(
                input_ids.shape[1], device=block_in.device, dtype=torch.long
            ).unsqueeze(0)

            # RoPE global du modèle, partagé entre couches
            cos, sin = model.model.rotary_emb(block_in, position_ids)

            # application de RoPE sur q/k
            q_post, k_post = apply_rotary_pos_emb(q_t, k_t, cos, sin, unsqueeze_dim=1)

            # 4) expand KV heads -> Q heads (GQA)
            k_rep = repeat_kv(k_post, num_kv_groups)   # [bsz, q_heads, seq, head_dim]
            v_rep = repeat_kv(v_t, num_kv_groups)      # [bsz, q_heads, seq, head_dim]

            # 5) causal mask
            bsz, _, tgt_len, _ = q_post.shape
            src_len = k_rep.shape[2]

            causal = torch.full(
                (tgt_len, src_len),
                fill_value=torch.finfo(q_post.dtype).min,
                device=q_post.device,
                dtype=q_post.dtype,
            )
            causal = torch.triu(causal, diagonal=1)
            causal = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt, src]

            if attention_mask is not None:
                # attention_mask: [bsz, seq], 1 for valid
                am = attention_mask[:, None, None, :].to(q_post.device)
                additive_mask = (1.0 - am.to(q_post.dtype)) * torch.finfo(q_post.dtype).min
                attn_mask_4d = causal + additive_mask
            else:
                attn_mask_4d = causal

            # 6) raw scores + softmax attention
            attn_scores = torch.matmul(q_post, k_rep.transpose(-1, -2)) / math.sqrt(head_dim)
            attn_scores = attn_scores + attn_mask_4d
            attn_weights = torch.softmax(attn_scores, dim=-1)   # [bsz, q_heads, tgt, src]

            # 7) per-head output before o_proj
            head_out = torch.matmul(attn_weights, v_rep)        # [bsz, q_heads, seq, head_dim]

            # 8) merge heads + o_proj
            attn_merge = head_out.transpose(1, 2).contiguous().view(bsz, tgt_len, hidden_size)
            attn_block_out = layer.self_attn.o_proj(attn_merge)  # [bsz, seq, hidden]

            # 9) first residual
            resid1 = block_in + attn_block_out                   # [bsz, seq, hidden]

            # 10) second RMSNorm
            mlp_norm_out = layer.post_attention_layernorm(resid1)

            # 11) MLP
            mlp_out = layer.mlp(mlp_norm_out)                    # [bsz, seq, hidden]

            # 12) second residual / final block output
            resid2 = resid1 + mlp_out                            # [bsz, seq, hidden]

            kv_head_for_q = (torch.arange(num_q_heads) // num_kv_groups).cpu()

            layer_record = {
                # streams full hidden
                "block_input": to_cpu(block_in),                 # [bsz, seq, hidden]
                "input_layernorm_out": to_cpu(attn_norm_out),    # [bsz, seq, hidden]
                "attn_block_out": to_cpu(attn_block_out),        # [bsz, seq, hidden]
                "residual_after_attn": to_cpu(resid1),           # [bsz, seq, hidden]
                "post_attn_layernorm_out": to_cpu(mlp_norm_out), # [bsz, seq, hidden]
                "mlp_out": to_cpu(mlp_out),                      # [bsz, seq, hidden]
                "block_output": to_cpu(resid2),                  # [bsz, seq, hidden]

                # qkv before / after rope
                "q_pre_rope": to_cpu(q_pre),                     # [bsz, seq, q_heads, head_dim]
                "k_pre_rope": to_cpu(k_pre),                     # [bsz, seq, kv_heads, head_dim]
                "v_pre": to_cpu(v_pre),                          # [bsz, seq, kv_heads, head_dim]

                "q_post_rope": to_cpu(q_post.transpose(1, 2)),   # [bsz, seq, q_heads, head_dim]
                "k_post_rope": to_cpu(k_post.transpose(1, 2)),   # [bsz, seq, kv_heads, head_dim]

                # attention internals
                "attn_scores": to_cpu(attn_scores),              # [bsz, q_heads, tgt, src]
                "attn_weights": to_cpu(attn_weights),            # [bsz, q_heads, tgt, src]
                "head_out": to_cpu(head_out.transpose(1, 2)),    # [bsz, seq, q_heads, head_dim]

                # metadata for GQA
                "kv_head_for_q": kv_head_for_q,                  # [q_heads]
            }

            # check consistency with HF hidden_states if desired
            layer_record["checks"] = {
                "max_abs_diff_block_output_vs_hf_hidden_state": float((resid2 - block_out).abs().max().item()),
                "max_abs_diff_attn_weights_vs_hf_attn": float((attn_weights - attentions[layer_idx].to(attn_weights.device)).abs().max().item()),
            }

            prompt_record["layers"].append(layer_record)

        payload["prompts"].append(prompt_record)

torch.save(payload, OUT_PATH)
print(f"Saved full block cache to: {OUT_PATH}")