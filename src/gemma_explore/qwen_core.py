# qwen_core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence
import math
import hashlib

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass
class QwenBundle:
    """Runtime bundle exposing model, tokenizer, and useful config metadata."""

    model: Any
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    model_id: str
    num_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_size: int
    head_dim: int
    num_frequencies: int
    rope_theta: float
    q_group_size: int


@dataclass
class PromptBlock:
    """Contiguous prompt span with text and token boundaries."""

    name: str
    text: str
    token_ids: list[int]
    tokens: list[str]
    start: int
    end: int


@dataclass
class PromptEncoding:
    """Tokenized prompt ready to run through the model."""

    messages: list[dict[str, str]] | None
    text: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    tokens: list[str]
    blocks: list[PromptBlock]


@dataclass
class ScoreOutputs:
    """Global positional/symbolic scores aggregated over block swaps."""

    att_last_col: torch.Tensor
    swaps: list[tuple[int, int]]
    blocks: list[tuple[int, int]]
    pos_scores: torch.Tensor
    sym_scores: torch.Tensor
    mean_scores: torch.Tensor
    tokens: list[str]
    prompt_token_ids: list[int]
    prompt_text: str
    prompt_blocks: list[PromptBlock]
    fix_query: bool = True


@dataclass
class FrequencyScoreOutputs:
    """Frequency-resolved RoPE analysis."""

    full_scores: torch.Tensor
    frequency_scores: torch.Tensor
    frequency_logit_norms: torch.Tensor
    frequency_wavelengths: torch.Tensor
    attention_matrix: torch.Tensor
    swaps: list[tuple[int, int]]
    blocks: list[tuple[int, int]]
    tokens: list[str]
    prompt_token_ids: list[int]
    prompt_text: str
    prompt_blocks: list[PromptBlock]
    fix_query: bool = True


@dataclass
class LayerQKVOutputs:
    """Q/K/V reconstruction for one decoder layer."""

    input_layernorm_out: torch.Tensor
    q_pre_rope: torch.Tensor
    k_pre_rope: torch.Tensor
    v_pre: torch.Tensor
    q_post_rope: torch.Tensor
    k_post_rope: torch.Tensor
    k_repeated: torch.Tensor
    v_repeated: torch.Tensor
    kv_head_for_q: torch.Tensor
    position_ids: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


# ---------------------------------------------------------------------
# Prompt / model helpers
# ---------------------------------------------------------------------


def load_bundle(
    model_name: str = DEFAULT_MODEL,
    device_map: str | dict[str, Any] | None = "auto",
    torch_dtype: str | torch.dtype = "auto",
    trust_remote_code: bool = True,
    attn_implementation: str = "eager",
) -> QwenBundle:
    """Load a Qwen/Qwen2-style model and tokenizer.

    Attention tensors are needed for analysis, so eager attention is preserved by
    default.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    ).eval()

    cfg = model.config
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    num_heads = int(cfg.num_attention_heads)
    num_kv_heads = int(cfg.num_key_value_heads)
    hidden_size = int(cfg.hidden_size)
    head_dim = hidden_size // num_heads

    return QwenBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        model_id=model_name,
        num_layers=int(cfg.num_hidden_layers),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=hidden_size,
        head_dim=head_dim,
        num_frequencies=head_dim // 2,
        rope_theta=float(getattr(cfg, "rope_theta", 1_000_000.0)),
        q_group_size=num_heads // num_kv_heads,
    )


def normalize_input_ids(input_ids: torch.Tensor | Sequence[int]) -> list[int]:
    """Normalize input ids to a plain Python list."""
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 2:
            if input_ids.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got shape {tuple(input_ids.shape)}")
            return input_ids[0].detach().cpu().tolist()
        if input_ids.ndim == 1:
            return input_ids.detach().cpu().tolist()
        raise ValueError(f"Expected 1D or 2D input_ids, got ndim={input_ids.ndim}")
    return list(input_ids)


def encode_prompt(
    bundle: QwenBundle,
    prompt: str | list[dict[str, str]],
    apply_chat_template: bool = True,
    add_generation_prompt: bool = True,
) -> PromptEncoding:
    """Encode a raw prompt or chat message list into model-ready tensors."""
    messages: list[dict[str, str]] | None

    if isinstance(prompt, str):
        if apply_chat_template:
            messages = [{"role": "user", "content": prompt}]
            text = bundle.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            block_name = "chat_template"
        else:
            messages = None
            text = prompt
            block_name = "raw_prompt"
    else:
        messages = [dict(m) for m in prompt]
        text = bundle.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        block_name = "chat_template"

    enc = bundle.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(bundle.device)
    attention_mask = enc["attention_mask"].to(bundle.device)
    token_ids = input_ids[0].tolist()
    tokens = bundle.tokenizer.convert_ids_to_tokens(token_ids)

    blocks = [
        PromptBlock(
            name=block_name,
            text=text,
            token_ids=token_ids,
            tokens=tokens,
            start=0,
            end=len(token_ids),
        )
    ]
    return PromptEncoding(
        messages=messages,
        text=text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokens=tokens,
        blocks=blocks,
    )


def prompt_text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def token_ids_hash(token_ids: Sequence[int]) -> str:
    h = hashlib.sha1()
    h.update(",".join(str(int(x)) for x in token_ids).encode("utf-8"))
    return h.hexdigest()


def print_prompt_with_blocks(
    tokenizer: Any,
    input_ids: torch.Tensor | Sequence[int],
    blocks: Sequence[tuple[int, int]],
    prefix: str = "# ",
    show_tokens: bool = False,
) -> None:
    """Pretty-print prompt spans corresponding to token blocks."""
    ids = normalize_input_ids(input_ids)
    print(f"{prefix}Prompt split into {len(blocks)} block(s):")
    for i, (start, end) in enumerate(blocks):
        if not (0 <= start <= end <= len(ids)):
            raise ValueError(f"Invalid block #{i}: {(start, end)} for seq_len={len(ids)}")
        block_ids = ids[start:end]
        block_text = tokenizer.decode(block_ids, skip_special_tokens=False)
        print(f"{prefix}block {i:02d} [{start}:{end}] {block_text!r}")
        if show_tokens:
            print(f"{prefix}tokens[{i}] = {tokenizer.convert_ids_to_tokens(block_ids)}")


def print_encoding_blocks(
    bundle: QwenBundle,
    encoding: PromptEncoding,
    blocks: Sequence[tuple[int, int]],
    prefix: str = "# ",
    show_tokens: bool = False,
) -> None:
    print_prompt_with_blocks(
        tokenizer=bundle.tokenizer,
        input_ids=encoding.input_ids,
        blocks=blocks,
        prefix=prefix,
        show_tokens=show_tokens,
    )


def inspect_token_span(
    bundle: QwenBundle,
    input_ids: torch.Tensor | Sequence[int],
    start: int,
    end: int,
) -> dict[str, Any]:
    """Return a compact token span description for notebook inspection."""
    ids = normalize_input_ids(input_ids)
    if not (0 <= start <= end <= len(ids)):
        raise ValueError(f"Invalid span {(start, end)} for seq_len={len(ids)}")
    span_ids = ids[start:end]
    return {
        "start": start,
        "end": end,
        "token_ids": span_ids,
        "tokens": bundle.tokenizer.convert_ids_to_tokens(span_ids),
        "text": bundle.tokenizer.decode(span_ids, skip_special_tokens=False),
    }


# ---------------------------------------------------------------------
# Block permutations
# ---------------------------------------------------------------------


def split_into_blocks(n_tokens: int, n_blocks: int) -> list[tuple[int, int]]:
    """Split a sequence into contiguous blocks with nearly equal lengths."""
    if n_tokens <= 0:
        raise ValueError("n_tokens must be positive")
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive")
    n_blocks = min(n_blocks, n_tokens)
    base, rem = divmod(n_tokens, n_blocks)
    out: list[tuple[int, int]] = []
    start = 0
    for b in range(n_blocks):
        length = base + (1 if b < rem else 0)
        out.append((start, start + length))
        start += length
    return out


def all_block_swaps(n_blocks: int) -> list[tuple[int, int]]:
    """List all distinct block swap pairs."""
    if n_blocks < 1:
        return []
    return [(i, j) for i in range(n_blocks - 1) for j in range(i + 1, n_blocks)]


def build_swap_indices(
    seq_len: int,
    blocks: Sequence[tuple[int, int]],
    swaps: Sequence[tuple[int, int]],
) -> torch.Tensor:
    """Build per-swap token permutations.

    If block sizes differ, swap only the common prefix, preserving sequence length.
    """
    col_idx = torch.arange(seq_len).unsqueeze(0).expand(len(swaps), -1).clone()
    for k, (bi, bj) in enumerate(swaps):
        si, ei = blocks[bi]
        sj, ej = blocks[bj]
        li, lj = ei - si, ej - sj
        lmin = min(li, lj)
        if lmin == 0:
            raise ValueError(f"Empty block encountered in swap {(bi, bj)}")
        col_idx[k, si : si + lmin] = torch.arange(sj, sj + lmin)
        col_idx[k, sj : sj + lmin] = torch.arange(si, si + lmin)
    return col_idx


def permute_input_ids(input_ids: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply a positional permutation to a batch-1 input_ids tensor."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"Expected input_ids shape [1, seq], got {tuple(input_ids.shape)}")
    return input_ids[:, perm]


# ---------------------------------------------------------------------
# Attention internals
# ---------------------------------------------------------------------


def get_decoder_layers(bundle: QwenBundle):
    """Return decoder layers from the backbone model."""
    return bundle.model.model.layers


def get_decoder_layer(bundle: QwenBundle, layer_idx: int):
    layers = get_decoder_layers(bundle)
    if not (0 <= layer_idx < len(layers)):
        raise IndexError(f"layer_idx={layer_idx} out of range for {len(layers)} layers")
    return layers[layer_idx]


def to_cpu(x: torch.Tensor | None) -> torch.Tensor | None:
    return None if x is None else x.detach().to("cpu")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Canonical GQA KV repetition.

    Input: [batch, kv_heads, seq, head_dim]
    Output: [batch, q_heads,  seq, head_dim]
    """
    if n_rep <= 0:
        raise ValueError(f"n_rep must be positive, got {n_rep}")
    if n_rep == 1:
        return hidden_states
    bsz, nkv, seq_len, head_dim = hidden_states.shape
    x = hidden_states[:, :, None, :, :].expand(bsz, nkv, n_rep, seq_len, head_dim)
    return x.reshape(bsz, nkv * n_rep, seq_len, head_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """RoPE half rotation on the last dimension."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k.

    q: [batch, q_heads, seq, head_dim]
    k: [batch, kv_heads, seq, head_dim]
    cos/sin: [batch, seq, head_dim]
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def _causal_attention_mask(
    q: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Build additive causal mask in the same dtype/device as q."""
    _, _, tgt_len, _ = q.shape
    causal = torch.full(
        (tgt_len, tgt_len),
        fill_value=torch.finfo(q.dtype).min,
        device=q.device,
        dtype=q.dtype,
    )
    causal = torch.triu(causal, diagonal=1).unsqueeze(0).unsqueeze(0)
    if attention_mask is None:
        return causal
    am = attention_mask[:, None, None, :].to(device=q.device)
    additive = (1.0 - am.to(q.dtype)) * torch.finfo(q.dtype).min
    return causal + additive


def compute_qkv_for_layer(
    bundle: QwenBundle,
    hidden_in: torch.Tensor,
    layer_idx: int,
) -> LayerQKVOutputs:
    """Reconstruct per-layer Q/K/V before and after RoPE.

    hidden_in shape: [batch, seq, hidden_size]
    q_pre_rope, q_post_rope: [batch, seq, q_heads, head_dim]
    k_pre_rope, k_post_rope, v_pre: [batch, seq, kv_heads, head_dim]
    """
    layer = get_decoder_layer(bundle, layer_idx)
    attn = layer.self_attn
    device = attn.q_proj.weight.device
    dtype = attn.q_proj.weight.dtype

    hidden_in = hidden_in.to(device=device, dtype=dtype)
    x = layer.input_layernorm(hidden_in)

    bsz, seq_len, _ = x.shape

    q_pre = attn.q_proj(x).view(bsz, seq_len, bundle.num_heads, bundle.head_dim)
    k_pre = attn.k_proj(x).view(bsz, seq_len, bundle.num_kv_heads, bundle.head_dim)
    v_pre = attn.v_proj(x).view(bsz, seq_len, bundle.num_kv_heads, bundle.head_dim)

    q_t = q_pre.transpose(1, 2).contiguous()
    k_t = k_pre.transpose(1, 2).contiguous()
    v_t = v_pre.transpose(1, 2).contiguous()

    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    cos, sin = bundle.model.model.rotary_emb(x, position_ids)
    q_post_t, k_post_t = apply_rotary(q_t, k_t, cos, sin)

    k_rep = repeat_kv(k_post_t, bundle.q_group_size)
    v_rep = repeat_kv(v_t, bundle.q_group_size)
    kv_head_for_q = (torch.arange(bundle.num_heads, device=device) // bundle.q_group_size).long()

    return LayerQKVOutputs(
        input_layernorm_out=x,
        q_pre_rope=q_pre,
        k_pre_rope=k_pre,
        v_pre=v_pre,
        q_post_rope=q_post_t.transpose(1, 2).contiguous(),
        k_post_rope=k_post_t.transpose(1, 2).contiguous(),
        k_repeated=k_rep,
        v_repeated=v_rep,
        kv_head_for_q=kv_head_for_q,
        position_ids=position_ids,
        cos=cos,
        sin=sin,
    )


def compute_attention_internals_for_layer(
    bundle: QwenBundle,
    hidden_in: torch.Tensor,
    attention_mask: torch.Tensor | None,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    """Reconstruct attention internals for a layer, matching HF layout when possible."""
    layer = get_decoder_layer(bundle, layer_idx)
    qkv = compute_qkv_for_layer(bundle, hidden_in, layer_idx)
    q = qkv.q_post_rope.transpose(1, 2).contiguous()
    k = qkv.k_repeated
    v = qkv.v_repeated

    mask_4d = _causal_attention_mask(q, attention_mask)
    attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(bundle.head_dim)
    attn_scores = attn_scores + mask_4d
    attn_weights = torch.softmax(attn_scores, dim=-1)
    head_out = torch.matmul(attn_weights, v)
    attn_merge = head_out.transpose(1, 2).contiguous().view(hidden_in.shape[0], hidden_in.shape[1], bundle.hidden_size)
    attn_block_out = layer.self_attn.o_proj(attn_merge)
    residual_after_attn = hidden_in.to(attn_block_out.device, dtype=attn_block_out.dtype) + attn_block_out
    post_attn_layernorm_out = layer.post_attention_layernorm(residual_after_attn)
    mlp_out = layer.mlp(post_attn_layernorm_out)
    block_output = residual_after_attn + mlp_out

    return {
        "block_input": hidden_in,
        "input_layernorm_out": qkv.input_layernorm_out,
        "attn_block_out": attn_block_out,
        "residual_after_attn": residual_after_attn,
        "post_attn_layernorm_out": post_attn_layernorm_out,
        "mlp_out": mlp_out,
        "block_output": block_output,
        "q_pre_rope": qkv.q_pre_rope,
        "k_pre_rope": qkv.k_pre_rope,
        "v_pre": qkv.v_pre,
        "q_post_rope": qkv.q_post_rope,
        "k_post_rope": qkv.k_post_rope,
        "attn_scores": attn_scores,
        "attn_weights": attn_weights,
        "head_out": head_out.transpose(1, 2).contiguous(),
        "kv_head_for_q": qkv.kv_head_for_q,
    }


def extract_full_attention_matrices(
    attentions: Sequence[torch.Tensor | None],
    num_layers: int,
    num_heads: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Stack attentions to shape [layers, heads, seq, seq]."""
    if attentions is None or len(attentions) == 0:
        raise RuntimeError("No attentions returned; use eager attention and output_attentions=True.")
    mats = []
    for a in attentions:
        if a is None:
            raise RuntimeError("A layer returned attention=None")
        mats.append(a[0])
    out = torch.stack(mats, dim=0).to(device=device, dtype=torch.float32)
    expected = (num_layers, num_heads, seq_len, seq_len)
    if tuple(out.shape) != expected:
        raise RuntimeError(f"Unexpected full attention shape {tuple(out.shape)} != {expected}")
    return out


def extract_last_token_attentions(
    attentions: Sequence[torch.Tensor | None],
    num_layers: int,
    num_heads: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Extract last-token attention vectors to shape [layers, heads, seq]."""
    if attentions is None or len(attentions) == 0:
        raise RuntimeError("No attentions returned; use eager attention and output_attentions=True.")
    cols = []
    for a in attentions:
        if a is None:
            raise RuntimeError("A layer returned attention=None")
        cols.append(a[0, :, -1, :])
    out = torch.stack(cols, dim=0).to(device=device, dtype=torch.float32)
    expected = (num_layers, num_heads, seq_len)
    if tuple(out.shape) != expected:
        raise RuntimeError(f"Unexpected last-token attention shape {tuple(out.shape)} != {expected}")
    return out


@torch.inference_mode()
def run_model(
    bundle: QwenBundle,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_hidden_states: bool = True,
    output_attentions: bool = True,
):
    """Run the decoder backbone without KV cache."""
    return bundle.model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
        return_dict=True,
        use_cache=False,
    )


# ---------------------------------------------------------------------
# Global scores
# ---------------------------------------------------------------------


def get_scores(
    att_last_col: torch.Tensor,
    swaps: Sequence[tuple[int, int]],
    blocks: Sequence[tuple[int, int]],
    tau: float = 0.1,
) -> torch.Tensor:
    """Compute positional and symbolic scores from swapped-block attentions.

    att_last_col shape: [layers, batch=1, n_swaps+1, heads_like, seq]
    output shape: [layers, heads_like, 1, 2]
    """
    device = att_last_col.device
    nl, ns, _, nh, seq_len = att_last_col.shape
    n_swaps = len(swaps)
    if ns != 1:
        raise ValueError(f"Expected singleton batch axis in att_last_col, got {ns}")
    if n_swaps == 0:
        return torch.zeros((nl, nh, ns, 2), device=device, dtype=att_last_col.dtype)

    m = len(blocks)
    swaps_t = torch.tensor(swaps, device=device)
    base = att_last_col[:, :, 0]
    perms = att_last_col[:, :, 1:]

    token_to_block = torch.empty(seq_len, dtype=torch.long, device=device)
    block_sizes = torch.empty(m, dtype=torch.long, device=device)
    for b, (s, e) in enumerate(blocks):
        token_to_block[s:e] = b
        block_sizes[b] = e - s

    idx = token_to_block.view(1, 1, 1, -1).expand(nl, ns, nh, -1)
    block_sum_base = torch.zeros(nl, ns, nh, m, device=device, dtype=base.dtype)
    block_sum_base.scatter_add_(-1, idx, base)
    block_avg_base = block_sum_base / block_sizes

    perms = perms.permute(0, 1, 3, 2, 4)

    permuted_block_ids = token_to_block.unsqueeze(0).expand(n_swaps, -1).clone()
    bi = swaps_t[:, 0].unsqueeze(1)
    bj = swaps_t[:, 1].unsqueeze(1)
    permuted_block_ids = torch.where(permuted_block_ids == bi, bj, permuted_block_ids)
    permuted_block_ids = torch.where(permuted_block_ids == bj, bi, permuted_block_ids)

    idx_perm = permuted_block_ids.view(1, 1, 1, n_swaps, seq_len).expand(nl, ns, nh, -1, -1)
    block_sum_perm = torch.zeros(nl, ns, nh, n_swaps, m, device=device, dtype=perms.dtype)
    block_sum_perm.scatter_add_(-1, idx_perm, perms)

    perm_sizes = block_sizes.unsqueeze(0).expand(n_swaps, -1).clone()
    bi1, bj1 = swaps_t[:, 0], swaps_t[:, 1]
    tmp = perm_sizes[:, bi1].clone()
    perm_sizes[:, bi1] = perm_sizes[:, bj1]
    perm_sizes[:, bj1] = tmp
    block_avg_perm = block_sum_perm / perm_sizes.view(1, 1, 1, n_swaps, m)

    vij_base = torch.stack([block_avg_base[..., bi1], block_avg_base[..., bj1]], dim=-1)
    swap_range = torch.arange(n_swaps, device=device)
    vij_perm = torch.stack(
        [block_avg_perm[..., swap_range, bj1], block_avg_perm[..., swap_range, bi1]],
        dim=-1,
    )

    deltas = (block_avg_base[..., bi1] - block_avg_base[..., bj1]).abs()
    weights = F.softmax(deltas / tau, dim=-1)

    pos = F.cosine_similarity(vij_perm, vij_base, dim=-1)
    sym = F.cosine_similarity(vij_perm, torch.flip(vij_base, dims=[-1]), dim=-1)

    pos_scores = (weights * pos).sum(dim=-1).permute(0, 2, 1)
    sym_scores = (weights * sym).sum(dim=-1).permute(0, 2, 1)
    return torch.stack([pos_scores, sym_scores], dim=-1)


def _compute_layer_attn_weights(
    bundle: QwenBundle,
    hidden_in: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Compute attention weights [batch, heads, seq, seq] without running MLP or residual."""
    qkv = compute_qkv_for_layer(bundle, hidden_in, layer_idx)
    q = qkv.q_post_rope.transpose(1, 2).contiguous()
    k = qkv.k_repeated
    mask_4d = _causal_attention_mask(q, None)
    attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(bundle.head_dim)
    return torch.softmax(attn_scores + mask_4d, dim=-1)


def _compute_layer_attn_weights_fixed_query(
    bundle: QwenBundle,
    hidden_orig: torch.Tensor,
    hidden_perm: torch.Tensor,
    layer_idx: int,
) -> torch.Tensor:
    """Compute attention weights keeping Q from the original hidden state.

    Q is taken from hidden_orig so the original last token's query is preserved.
    K is taken from hidden_perm so the keys reflect the permuted token positions.
    This isolates whether the last token attends to a *position* (positional) or
    to a specific *symbol* (symbolic) when that symbol is moved by a block swap.

    hidden_orig: [batch, seq, hidden_size] — unperturbed input to the layer
    hidden_perm: [batch, seq, hidden_size] — permuted input to the layer
    Returns: [batch, heads, seq, seq]
    """
    qkv_orig = compute_qkv_for_layer(bundle, hidden_orig, layer_idx)
    qkv_perm = compute_qkv_for_layer(bundle, hidden_perm, layer_idx)
    q = qkv_orig.q_post_rope.transpose(1, 2).contiguous()
    k = qkv_perm.k_repeated
    mask_4d = _causal_attention_mask(q, None)
    attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(bundle.head_dim)
    return torch.softmax(attn_scores + mask_4d, dim=-1)


@torch.inference_mode()
def collect_scores(
    bundle: QwenBundle,
    prompt: str | list[dict[str, str]],
    n_blocks: int = 16,
    tau: float = 0.1,
    verbose_prompt: bool = False,
    verbose_prompt_tokens: bool = False,
    apply_chat_template: bool = True,
    add_generation_prompt: bool = True,
    fix_query: bool = True,
) -> ScoreOutputs:
    """Compute positional/symbolic scores with per-layer hidden-state permutation.

    For each layer, the block swap is applied to that layer's input hidden states
    rather than to the token IDs, isolating each layer's own positional behavior.

    When fix_query=True (default), the last token's query is held fixed at its
    original value while only the permuted hidden states supply the keys. This
    correctly measures whether the last token attends to a *position* or a
    *symbol* even when the last token itself is displaced by a swap.

    When fix_query=False, both Q and K come from the permuted hidden state
    (legacy behaviour).
    """
    encoding = encode_prompt(
        bundle,
        prompt,
        apply_chat_template=apply_chat_template,
        add_generation_prompt=add_generation_prompt,
    )
    seq_len = encoding.input_ids.shape[1]
    blocks = split_into_blocks(seq_len, n_blocks)
    if verbose_prompt:
        print_encoding_blocks(bundle, encoding, blocks, show_tokens=verbose_prompt_tokens)

    swaps = all_block_swaps(len(blocks))
    col_idx = build_swap_indices(seq_len, blocks, swaps).to(bundle.device)
    n_swaps = len(swaps)
    nl = bundle.num_layers
    nh = bundle.num_heads

    base_out = run_model(
        bundle,
        encoding.input_ids,
        torch.ones_like(encoding.input_ids),
        output_hidden_states=True,
        output_attentions=False,
    )
    base_hidden = base_out.hidden_states[:-1]  # input to each decoder layer

    att_last_col = torch.empty(nl, 1, n_swaps + 1, nh, seq_len, device=bundle.device, dtype=torch.float32)

    for swap_idx in range(n_swaps + 1):
        perm = col_idx[swap_idx - 1] if swap_idx > 0 else None
        for layer_idx in range(nl):
            hidden_orig = base_hidden[layer_idx]
            if perm is not None and fix_query:
                w = _compute_layer_attn_weights_fixed_query(
                    bundle, hidden_orig, hidden_orig[:, perm, :], layer_idx
                )
            else:
                hidden = hidden_orig[:, perm, :] if perm is not None else hidden_orig
                w = _compute_layer_attn_weights(bundle, hidden, layer_idx)
            att_last_col[layer_idx, 0, swap_idx] = w[0, :, -1, :].float()

    scores = get_scores(att_last_col, swaps, blocks, tau=tau).detach().cpu()
    return ScoreOutputs(
        att_last_col=att_last_col.detach().cpu(),
        swaps=list(swaps),
        blocks=list(blocks),
        pos_scores=scores[..., 0],
        sym_scores=scores[..., 1],
        mean_scores=scores.mean(dim=2),
        tokens=encoding.tokens,
        prompt_token_ids=encoding.input_ids[0].tolist(),
        prompt_text=encoding.text,
        prompt_blocks=encoding.blocks,
        fix_query=fix_query,
    )


# ---------------------------------------------------------------------
# Frequency analysis
# ---------------------------------------------------------------------


def rope_wavelengths(bundle: QwenBundle) -> torch.Tensor:
    """Return RoPE wavelengths for each frequency pair."""
    idx = torch.arange(bundle.num_frequencies, dtype=torch.float32, device=bundle.device)
    inv_freq = 1.0 / (bundle.rope_theta ** (idx / bundle.num_frequencies))
    return (2.0 * math.pi) / inv_freq


def compute_all_frequency_attentions_and_norms(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve attention contributions by RoPE frequency.

    q, k shapes: [batch, heads, seq, head_dim]
    Returns:
      attention: [batch, heads, num_freq, seq]
      logit_norms: [batch, heads, num_freq]
    """
    half = q.shape[-1] // 2
    q_pairs = torch.stack([q[..., :half], q[..., half:]], dim=-1)
    k_pairs = torch.stack([k[..., :half], k[..., half:]], dim=-1)
    q_last = q_pairs[:, :, -1]
    logits = torch.einsum("bhfd,bhtfd->bhft", q_last, k_pairs) / math.sqrt(head_dim)
    att = torch.softmax(logits, dim=-1)
    norms = torch.linalg.vector_norm(logits.float(), ord=2, dim=-1)
    return att, norms


@torch.inference_mode()
def collect_frequency_scores(
    bundle: QwenBundle,
    prompt: str | list[dict[str, str]],
    n_blocks: int = 16,
    tau: float = 0.1,
    verbose_prompt: bool = False,
    verbose_prompt_tokens: bool = False,
    apply_chat_template: bool = True,
    add_generation_prompt: bool = True,
    fix_query: bool = True,
) -> FrequencyScoreOutputs:
    """Compute full and frequency-resolved scores with per-layer hidden-state permutation.

    For each layer, block swaps are applied to that layer's input hidden states,
    isolating each layer's own positional/symbolic behavior.

    When fix_query=True (default), the last token's query is held fixed at its
    original value while only the permuted hidden states supply the keys.

    When fix_query=False, both Q and K come from the permuted hidden state
    (legacy behaviour).
    """
    encoding = encode_prompt(
        bundle,
        prompt,
        apply_chat_template=apply_chat_template,
        add_generation_prompt=add_generation_prompt,
    )
    seq_len = encoding.input_ids.shape[1]
    blocks = split_into_blocks(seq_len, n_blocks)
    if verbose_prompt:
        print_encoding_blocks(bundle, encoding, blocks, show_tokens=verbose_prompt_tokens)

    swaps = all_block_swaps(len(blocks))
    col_idx = build_swap_indices(seq_len, blocks, swaps).to(bundle.device)

    n_swaps = len(swaps)
    nl = bundle.num_layers
    nh = bundle.num_heads
    nf = bundle.num_frequencies

    # Base pass: capture hidden states and full attention matrix for visualization
    base_out = run_model(
        bundle,
        encoding.input_ids,
        torch.ones_like(encoding.input_ids),
        output_hidden_states=True,
        output_attentions=True,
    )
    base_hidden = base_out.hidden_states[:-1]  # input to each decoder layer
    attention_matrix = extract_full_attention_matrices(
        base_out.attentions, nl, nh, seq_len, bundle.device
    ).detach().cpu()

    full_att_last_col = torch.empty(nl, 1, n_swaps + 1, nh, seq_len, device=bundle.device, dtype=torch.float32)
    freq_logit_norms = torch.empty(nl, n_swaps + 1, nh, nf, device="cpu", dtype=torch.float32)
    freq_att_last_col = torch.empty(nl, 1, n_swaps + 1, nh, nf, seq_len, device=bundle.device, dtype=torch.float32)

    for swap_idx in range(n_swaps + 1):
        perm = col_idx[swap_idx - 1] if swap_idx > 0 else None
        for layer_idx in range(nl):
            hidden_orig = base_hidden[layer_idx]
            if perm is not None and fix_query:
                qkv_orig = compute_qkv_for_layer(bundle, hidden_orig, layer_idx)
                qkv_perm = compute_qkv_for_layer(bundle, hidden_orig[:, perm, :], layer_idx)
                q = qkv_orig.q_post_rope.transpose(1, 2).contiguous()
                k = qkv_perm.k_repeated
            else:
                hidden = hidden_orig[:, perm, :] if perm is not None else hidden_orig
                qkv = compute_qkv_for_layer(bundle, hidden, layer_idx)
                q = qkv.q_post_rope.transpose(1, 2).contiguous()
                k = qkv.k_repeated
            # Full attention for pos/sym scoring
            mask_4d = _causal_attention_mask(q, None)
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(bundle.head_dim)
            attn_weights = torch.softmax(attn_scores + mask_4d, dim=-1)
            full_att_last_col[layer_idx, 0, swap_idx] = attn_weights[0, :, -1, :].float()
            # Frequency-resolved
            att_f, norms_f = compute_all_frequency_attentions_and_norms(q, k, bundle.head_dim)
            freq_att_last_col[layer_idx, 0, swap_idx] = att_f[0].float()
            freq_logit_norms[layer_idx, swap_idx] = norms_f[0].float().cpu()

    full_scores = get_scores(full_att_last_col, swaps, blocks, tau=tau).detach().cpu().squeeze(dim=-2)
    freq_flat = freq_att_last_col.permute(0, 1, 2, 4, 3, 5).reshape(nl, 1, n_swaps + 1, nf * nh, seq_len)
    scores_f = get_scores(freq_flat, swaps, blocks, tau=tau)
    scores_f = scores_f.squeeze(dim=2).reshape(nl, nf, nh, 2).permute(0, 2, 1, 3).contiguous()

    return FrequencyScoreOutputs(
        full_scores=full_scores,
        frequency_scores=scores_f.detach().cpu(),
        frequency_logit_norms=freq_logit_norms,
        frequency_wavelengths=rope_wavelengths(bundle).detach().cpu(),
        attention_matrix=attention_matrix,
        swaps=list(swaps),
        blocks=list(blocks),
        tokens=encoding.tokens,
        prompt_token_ids=encoding.input_ids[0].tolist(),
        prompt_text=encoding.text,
        prompt_blocks=encoding.blocks,
        fix_query=fix_query,
    )