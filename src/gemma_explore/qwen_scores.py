from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class QwenAttentionBundle:
    model: Any
    tokenizer: Any
    device: torch.device
    dtype: torch.dtype
    num_layers: int
    num_heads: int
    num_kv_heads: int
    hidden_size: int
    head_dim: int
    num_frequencies: int


@dataclass
class PromptBlock:
    name: str
    text: str
    token_ids: list[int]
    tokens: list[str]
    start: int
    end: int


@dataclass
class PromptEncoding:
    messages: list[dict[str, str]]
    text: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    tokens: list[str]
    blocks: list[PromptBlock]


@dataclass
class FrequencyScoreOutputs:
    full_scores: torch.Tensor
    frequency_scores: torch.Tensor
    frequency_logit_norms: torch.Tensor
    frequency_wavelengths: torch.Tensor
    swaps: list[tuple[int, int]]
    blocks: list[tuple[int, int]]
    tokens: list[str]
    prompt_text: str
    prompt_blocks: list[PromptBlock]

@dataclass
class ScoreOutputs:
    att_last_col: torch.Tensor
    swaps: list[tuple[int, int]]
    blocks: list[tuple[int, int]]
    pos_scores: torch.Tensor
    sym_scores: torch.Tensor
    mean_scores: torch.Tensor


def load_qwen_bundle(
    model_name: str = DEFAULT_MODEL,
    device_map: str = "auto",
    torch_dtype: str | torch.dtype = "auto",
    trust_remote_code: bool = True,
    attn_implementation: str = "eager",
) -> QwenAttentionBundle:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    model.eval()
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    return QwenAttentionBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        num_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        hidden_size=cfg.hidden_size,
        head_dim=head_dim,
        num_frequencies=head_dim // 2,
    )


def build_messages(prompt: str, system_prompt: str | None = "You are a helpful assistant.") -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def encode_prompt(
    bundle: QwenAttentionBundle,
    prompt: str,
    system_prompt: str | None = "You are a helpful assistant.",
    add_generation_prompt: bool = True,
) -> PromptEncoding:
    messages = build_messages(prompt, system_prompt=system_prompt)
    text = bundle.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    enc = bundle.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(bundle.device)
    attention_mask = enc["attention_mask"].to(bundle.device)
    tokens = bundle.tokenizer.convert_ids_to_tokens(input_ids[0])
    blocks = _build_prompt_blocks(
        bundle,
        prompt=prompt,
        system_prompt=system_prompt,
        add_generation_prompt=add_generation_prompt,
    )
    return PromptEncoding(
        messages=messages,
        text=text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokens=tokens,
        blocks=blocks,
    )

def make_blocks(n: int, m: int) -> list[tuple[int, int]]:
    base, rem = divmod(n, m)
    blocks = []
    start = 0
    for k in range(m):
        length = base + (1 if k < rem else 0)
        blocks.append((start, start + length))
        start += length
    return blocks


def make_swaps(n_blocks: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_blocks - 1) for j in range(i + 1, n_blocks)]


def build_swap_permutations(seq_len: int, blocks: list[tuple[int, int]], swaps: list[tuple[int, int]]) -> torch.Tensor:
    col_idx = torch.arange(seq_len).unsqueeze(0).expand(len(swaps), -1).clone()
    for k, (bi, bj) in enumerate(swaps):
        si, ei = blocks[bi]
        sj, ej = blocks[bj]
        li = ei - si
        lj = ej - sj
        lmin = min(li, lj)
        if lmin == 0:
            raise ValueError(f"Empty block in swap ({bi}, {bj}).")
        col_idx[k, si:si + lmin] = torch.arange(sj, sj + lmin)
        col_idx[k, sj:sj + lmin] = torch.arange(si, si + lmin)
    return col_idx


def get_scores(att_last_col: torch.Tensor, swaps: list[tuple[int, int]], blocks: list[tuple[int, int]], tau: float = 0.1) -> torch.Tensor:
    device = att_last_col.device
    nl, ns, _, nh, seq_len = att_last_col.shape
    n_swaps = len(swaps)
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

    block_sum_base = torch.zeros(nl, ns, nh, m, device=device, dtype=base.dtype)
    idx = token_to_block.view(1, 1, 1, -1).expand(nl, ns, nh, -1)
    block_sum_base.scatter_add_(-1, idx, base)
    block_avg_base = block_sum_base / block_sizes

    perms = perms.permute(0, 1, 3, 2, 4)
    permuted_block_ids = token_to_block.unsqueeze(0).expand(n_swaps, -1).clone()
    bi = swaps_t[:, 0].unsqueeze(1)
    bj = swaps_t[:, 1].unsqueeze(1)
    mask_i = permuted_block_ids == bi
    mask_j = permuted_block_ids == bj
    permuted_block_ids = torch.where(mask_i, bj, permuted_block_ids)
    permuted_block_ids = torch.where(mask_j, bi, permuted_block_ids)

    idx_perm = permuted_block_ids.view(1, 1, 1, n_swaps, seq_len).expand(nl, ns, nh, -1, -1)
    block_sum_perm = torch.zeros(nl, ns, nh, n_swaps, m, device=device, dtype=perms.dtype)
    block_sum_perm.scatter_add_(-1, idx_perm, perms)

    perm_sizes = block_sizes.unsqueeze(0).expand(n_swaps, -1).clone()
    bi1 = swaps_t[:, 0]
    bj1 = swaps_t[:, 1]
    tmp = perm_sizes[:, bi1].clone()
    perm_sizes[:, bi1] = perm_sizes[:, bj1]
    perm_sizes[:, bj1] = tmp
    block_avg_perm = block_sum_perm / perm_sizes.view(1, 1, 1, n_swaps, m)

    vij_base = torch.stack([block_avg_base[..., bi1], block_avg_base[..., bj1]], dim=-1)
    swap_range = torch.arange(n_swaps, device=device)
    vij_perm = torch.stack([block_avg_perm[..., swap_range, bj1], block_avg_perm[..., swap_range, bi1]], dim=-1)

    deltas = (block_avg_base[..., bi1] - block_avg_base[..., bj1]).abs()
    weights = F.softmax(deltas / tau, dim=-1)
    pos = F.cosine_similarity(vij_perm, vij_base, dim=-1)
    sym = F.cosine_similarity(vij_perm, torch.flip(vij_base, dims=[-1]), dim=-1)

    pos_scores = (weights * pos).sum(dim=-1).permute(0, 2, 1)
    sym_scores = (weights * sym).sum(dim=-1).permute(0, 2, 1)
    return torch.stack([pos_scores, sym_scores], dim=-1)


def _repeat_kv(x: torch.Tensor, num_groups: int) -> torch.Tensor:
    if num_groups == 1:
        return x
    bsz, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(bsz, n_kv_heads, num_groups, seq_len, head_dim)
    return x.reshape(bsz, n_kv_heads * num_groups, seq_len, head_dim)


def _get_decoder_layers(bundle: QwenAttentionBundle):
    return bundle.model.model.layers


def _get_hidden_states(encoding: PromptEncoding, bundle: QwenAttentionBundle):
    out = bundle.model.model(
        input_ids=encoding.input_ids,
        attention_mask=encoding.attention_mask,
        output_hidden_states=True,
        output_attentions=True,
        return_dict=True,
        use_cache=False,
    )
    return out.hidden_states[:-1], out


def _compute_qk_for_layer(
    bundle: QwenAttentionBundle,
    hidden_in: torch.Tensor,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer = _get_decoder_layers(bundle)[layer_idx]
    attn = layer.self_attn

    bsz, seq_len, _ = hidden_in.shape
    device = attn.q_proj.weight.device
    dtype = attn.q_proj.weight.dtype

    n_heads = bundle.num_heads
    n_kv_heads = bundle.num_kv_heads
    head_dim = bundle.head_dim
    num_kv_groups = n_heads // n_kv_heads

    hidden_in = hidden_in.to(device=device, dtype=dtype)

    # Qwen2DecoderLayer.forward: input_layernorm avant self_attn
    hidden_states = layer.input_layernorm(hidden_in)

    q = attn.q_proj(hidden_states)
    k = attn.k_proj(hidden_states)

    q = q.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2).contiguous()
    k = k.view(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2).contiguous()

    # Qwen2Model.forward: position_embeddings = model.rotary_emb(hidden_states, position_ids)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cos, sin = bundle.model.model.rotary_emb(hidden_states, position_ids)

    # Même logique que apply_rotary_pos_emb dans modeling_qwen2.py
    q, k = apply_rotary_pos_emb_local(q, k, cos, sin)

    if num_kv_groups > 1:
        k = _repeat_kv(k, num_kv_groups)

    return q, k


def rotate_half_local(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _encode_text_piece(bundle: QwenAttentionBundle, text: str) -> tuple[list[int], list[str]]:
    enc = bundle.tokenizer(text, add_special_tokens=False)
    ids = enc["input_ids"]
    toks = bundle.tokenizer.convert_ids_to_tokens(ids)
    return ids, toks


def _build_prompt_blocks(
    bundle: QwenAttentionBundle,
    prompt: str,
    system_prompt: str | None,
    add_generation_prompt: bool,
) -> list[PromptBlock]:
    blocks: list[PromptBlock] = []
    cursor = 0

    if system_prompt is not None:
        system_text = bundle.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            tokenize=False,
            add_generation_prompt=False,
        )
        ids, toks = _encode_text_piece(bundle, system_text)
        blocks.append(
            PromptBlock(
                name="system",
                text=system_text,
                token_ids=ids,
                tokens=toks,
                start=cursor,
                end=cursor + len(ids),
            )
        )
        cursor += len(ids)

    user_text = bundle.tokenizer.apply_chat_template(
        build_messages(prompt, system_prompt=None),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    ids, toks = _encode_text_piece(bundle, user_text)
    blocks.append(
        PromptBlock(
            name="user+assistant_prefix" if add_generation_prompt else "user",
            text=user_text,
            token_ids=ids,
            tokens=toks,
            start=cursor,
            end=cursor + len(ids),
        )
    )
    cursor += len(ids)

    return blocks


def print_prompt_blocks(encoding: PromptEncoding) -> None:
    print("=" * 100)
    print("FINAL PROMPT")
    print("=" * 100)
    print(encoding.text)
    print()

    print("=" * 100)
    print("PROMPT BLOCKS")
    print("=" * 100)
    for block in encoding.blocks:
        print(f"[{block.name}] tokens {block.start}:{block.end}")
        print("-" * 100)
        print(block.text)
        print()
        print("TOKENS:")
        for i, tok in enumerate(block.tokens, start=block.start):
            print(f"{i:4d} | {tok}")
        print()


def apply_rotary_pos_emb_local(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half_local(q) * sin)
    k_embed = (k * cos) + (rotate_half_local(k) * sin)
    return q_embed, k_embed


def _last_token_attention_from_qk(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    q_last = q[:, :, -1:, :]
    logits = torch.matmul(q_last, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    return torch.softmax(logits, dim=-1).squeeze(-2)


def _frequency_masked_qk(q: torch.Tensor, k: torch.Tensor, freq_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    d = q.shape[-1]
    half = d // 2
    if freq_idx >= half:
        raise IndexError(f"Frequency index {freq_idx} out of range for head_dim={d}")

    q_sub = torch.zeros_like(q)
    k_sub = torch.zeros_like(k)

    q_sub[..., freq_idx] = q[..., freq_idx]
    q_sub[..., half + freq_idx] = q[..., half + freq_idx]

    k_sub[..., freq_idx] = k[..., freq_idx]
    k_sub[..., half + freq_idx] = k[..., half + freq_idx]

    return q_sub, k_sub

def get_frequency_logit_norms(
    bundle: QwenAttentionBundle,
    prompt: str,
    system_prompt: str | None = "You are a helpful assistant.",
    per_frequency_l2: bool = True,
) -> dict[str, Any]:
    encoding = encode_prompt(bundle, prompt, system_prompt=system_prompt)
    hidden_states, _ = _get_hidden_states(encoding, bundle)
    n_freq = bundle.num_frequencies
    seq_len = encoding.input_ids.shape[1]
    norms = torch.zeros(bundle.num_layers, bundle.num_heads, n_freq, device=bundle.device, dtype=torch.float32)
    raw_last_logits = torch.zeros(bundle.num_layers, bundle.num_heads, n_freq, seq_len, device=bundle.device, dtype=torch.float32)

    for layer_idx, hidden_in in enumerate(hidden_states):
        q, k = _compute_qk_for_layer(bundle, hidden_in, layer_idx)
        q_last = q[:, :, -1, :]
        for freq_idx in range(n_freq):
            start = 2 * freq_idx
            end = start + 2
            q_pair = q_last[..., start:end]
            k_pair = k[..., start:end]
            logits = torch.einsum("bhd,bhsd->bhs", q_pair, k_pair) / math.sqrt(bundle.head_dim)
            logits = logits[0].float()
            raw_last_logits[layer_idx, :, freq_idx] = logits
            if per_frequency_l2:
                norms[layer_idx, :, freq_idx] = torch.linalg.vector_norm(logits, ord=2, dim=-1)
            else:
                norms[layer_idx, :, freq_idx] = logits.abs().mean(dim=-1)
    return {
        "tokens": encoding.tokens,
        "text": encoding.text,
        "logit_norms": norms.detach().cpu(),
        "raw_frequency_logits": raw_last_logits.detach().cpu(),
    }


def _permute_input_ids(input_ids: torch.Tensor, col_idx: torch.Tensor) -> torch.Tensor:
    return input_ids[:, col_idx]


def _extract_last_token_attentions(attentions, num_layers: int, num_heads: int, seq_len: int, device: torch.device) -> torch.Tensor:
    if attentions is None or len(attentions) == 0:
        raise RuntimeError(
            "No attentions were returned by the model. This usually happens when the model was loaded "
            "with an attention backend that does not materialize attention weights. Reload with "
            "attn_implementation='eager' and output_attentions=True."
        )
    cols = []
    for a in attentions:
        if a is None:
            raise RuntimeError(
                "A layer returned attention=None. Reload the model with attn_implementation='eager'."
            )
        cols.append(a[0, :, -1, :])
    if len(cols) == 0:
        raise RuntimeError(
            "The attentions tuple is empty. Reload the model with attn_implementation='eager'."
        )
    out = torch.stack(cols, dim=0).to(device=device, dtype=torch.float32)
    if out.shape != (num_layers, num_heads, seq_len):
        raise RuntimeError(
            f"Unexpected attention shape {tuple(out.shape)}; expected {(num_layers, num_heads, seq_len)}."
        )
    return out


def collect_last_token_attentions(
    bundle: QwenAttentionBundle,
    prompt: str,
    system_prompt: str | None = "You are a helpful assistant.",
    n_blocks: int = 16,
    tau: float = 0.1,
    verbose_prompt: bool = True,
) -> ScoreOutputs:
    encoding = encode_prompt(bundle, prompt, system_prompt=system_prompt)
    if verbose_prompt:
        print_prompt_blocks(encoding)
    seq_len = encoding.input_ids.shape[1]
    m = min(n_blocks, seq_len)
    blocks = make_blocks(seq_len, m)
    swaps = make_swaps(m)
    col_idx = build_swap_permutations(seq_len, blocks, swaps)
    n_swaps = len(swaps)
    att_last_col = torch.zeros(bundle.num_layers, 1, n_swaps + 1, bundle.num_heads, seq_len, device=bundle.device, dtype=torch.float32)

    # CORRECTION : Utiliser _get_hidden_states au lieu de bundle.model() directement
    base_hidden_states, base_out = _get_hidden_states(encoding, bundle)
    att_last_col[:, 0, 0] = _extract_last_token_attentions(
        base_out.attentions,
        num_layers=bundle.num_layers,
        num_heads=bundle.num_heads,
        seq_len=seq_len,
        device=bundle.device,
    )

    for k in range(n_swaps):
        perm_ids = _permute_input_ids(encoding.input_ids, col_idx[k].to(bundle.device))
        # CORRECTION : Créer un PromptEncoding pour la variante
        enc_var = PromptEncoding(
            messages=encoding.messages,
            text=encoding.text,
            input_ids=perm_ids,
            attention_mask=torch.ones_like(perm_ids),
            tokens=bundle.tokenizer.convert_ids_to_tokens(perm_ids[0]),
            blocks=encoding.blocks
        )
        # CORRECTION : Utiliser _get_hidden_states
        hidden_states, out = _get_hidden_states(enc_var, bundle)
        att_last_col[:, 0, k + 1] = _extract_last_token_attentions(
            out.attentions,
            num_layers=bundle.num_layers,
            num_heads=bundle.num_heads,
            seq_len=seq_len,
            device=bundle.device,
        )

    scores = get_scores(att_last_col, swaps, blocks, tau=tau)
    return ScoreOutputs(
        att_last_col=att_last_col.detach().cpu(),
        swaps=swaps,
        blocks=blocks,
        pos_scores=scores[..., 0].detach().cpu(),
        sym_scores=scores[..., 1].detach().cpu(),
        mean_scores=scores.mean(dim=2).detach().cpu(),
    )


def collect_frequency_scores(
    bundle: QwenAttentionBundle,
    prompt: str,
    system_prompt: str | None = "You are a helpful assistant.",
    n_blocks: int = 16,
    tau: float = 0.1,
    verbose_prompt: bool = True,
) -> FrequencyScoreOutputs:
    encoding = encode_prompt(bundle, prompt, system_prompt=system_prompt)
    if verbose_prompt:
        print_prompt_blocks(encoding)
    seq_len = encoding.input_ids.shape[1]
    m = min(n_blocks, seq_len)
    blocks = make_blocks(seq_len, m)
    swaps = make_swaps(m)
    col_idx = build_swap_permutations(seq_len, blocks, swaps)
    n_swaps = len(swaps)
    n_freq = bundle.num_frequencies

    full_att_last_col = torch.zeros(
        bundle.num_layers, 1, n_swaps + 1, bundle.num_heads, seq_len,
        device=bundle.device, dtype=torch.float32
    )
    freq_att_last_col = torch.zeros(
        bundle.num_layers, 1, n_swaps + 1, bundle.num_heads, n_freq, seq_len,
        device=bundle.device, dtype=torch.float32
    )
    freq_logit_norms = torch.zeros(
        bundle.num_layers, bundle.num_heads, n_freq,
        device=bundle.device, dtype=torch.float32
    )

    variants = [encoding.input_ids] + [
        _permute_input_ids(encoding.input_ids, col_idx[k].to(bundle.device))
        for k in range(n_swaps)
    ]

    for swap_idx, ids in enumerate(variants):
        enc_var = PromptEncoding(
            messages=encoding.messages,
            text=encoding.text,
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            tokens=bundle.tokenizer.convert_ids_to_tokens(ids[0]),
            blocks=encoding.blocks,
        )
        hidden_states, out = _get_hidden_states(enc_var, bundle)
        full_att_last_col[:, 0, swap_idx] = _extract_last_token_attentions(
            out.attentions,
            num_layers=bundle.num_layers,
            num_heads=bundle.num_heads,
            seq_len=seq_len,
            device=bundle.device,
        )

        for layer_idx, hidden_in in enumerate(hidden_states):
            q, k = _compute_qk_for_layer(bundle, hidden_in, layer_idx)
            for freq_idx in range(n_freq):
                q_sub, k_sub = _frequency_masked_qk(q, k, freq_idx)
                att_sub = _last_token_attention_from_qk(q_sub, k_sub)[0].float()
                freq_att_last_col[layer_idx, 0, swap_idx, :, freq_idx] = att_sub

                if swap_idx == 0:
                    half = bundle.head_dim // 2
                    q_last_pair = torch.stack(
                        [q[0, :, -1, freq_idx], q[0, :, -1, half + freq_idx]],
                        dim=-1,
                    )
                    k_pair = torch.stack(
                        [k[0, :, :, freq_idx], k[0, :, :, half + freq_idx]],
                        dim=-1,
                    )
                    logits_pair = torch.einsum("hd,hsd->hs", q_last_pair, k_pair) / math.sqrt(bundle.head_dim)
                    freq_logit_norms[layer_idx, :, freq_idx] = torch.linalg.vector_norm(
                        logits_pair.float(), ord=2, dim=-1
                    )

    full_scores = get_scores(full_att_last_col, swaps, blocks, tau=tau).detach().cpu()

    frequency_scores = torch.zeros(
        bundle.num_layers, bundle.num_heads, n_freq, 2, dtype=torch.float32
    )
    for freq_idx in range(n_freq):
        scores_f = get_scores(freq_att_last_col[..., freq_idx, :], swaps, blocks, tau=tau)
        frequency_scores[:, :, freq_idx] = scores_f.mean(dim=2).detach().cpu()

    frequency_wavelengths = get_rope_wavelengths(bundle).detach().cpu()

    return FrequencyScoreOutputs(
        full_scores=full_scores,
        frequency_scores=frequency_scores,
        frequency_logit_norms=freq_logit_norms.detach().cpu(),
        frequency_wavelengths=frequency_wavelengths,
        swaps=swaps,
        blocks=blocks,
        tokens=encoding.tokens,
        prompt_text=encoding.text,
        prompt_blocks=encoding.blocks,
    )

def scores_to_dict(mean_scores: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"pos": mean_scores[..., 0], "sym": mean_scores[..., 1]}


def frequency_scores_for_head(
    freq_outputs: FrequencyScoreOutputs,
    layer_idx: int,
    head_idx: int,
) -> dict[str, torch.Tensor]:
    arr = freq_outputs.frequency_scores[layer_idx, head_idx]
    return {
        "pos": arr[:, 0],
        "sym": arr[:, 1],
        "norm": freq_outputs.frequency_logit_norms[layer_idx, head_idx],
        "wavelength": freq_outputs.frequency_wavelengths,
    }

def top_heads(mean_scores: torch.Tensor, score_type: str = "pos", k: int = 10) -> list[tuple[int, int, float]]:
    idx = 0 if score_type == "pos" else 1
    vals = mean_scores[..., idx]
    flat = vals.flatten()
    topv, topi = torch.topk(flat, k=min(k, flat.numel()))
    n_heads = vals.shape[1]
    return [(i // n_heads, i % n_heads, float(v)) for v, i in zip(topv.tolist(), topi.tolist())]


def plot_head_score_heatmap(mean_scores: torch.Tensor, score_type: str = "pos", ax=None, title: str | None = None):
    import matplotlib.pyplot as plt
    idx = 0 if score_type == "pos" else 1
    arr = mean_scores[..., idx].numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(arr, aspect="auto", origin="lower")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title or f"{score_type} score by layer/head")
    plt.colorbar(im, ax=ax)
    return ax

def get_rope_wavelengths(bundle: QwenAttentionBundle) -> torch.Tensor:
    rope_theta = getattr(bundle.model.config, "rope_theta", 10000.0)
    n_freq = bundle.num_frequencies
    idx = torch.arange(n_freq, dtype=torch.float32, device=bundle.device)

    inv_freq = 1.0 / (rope_theta ** (idx / n_freq))
    wavelengths = (2.0 * math.pi) / inv_freq
    return wavelengths

def plot_frequency_scores(
    freq_outputs: FrequencyScoreOutputs,
    layer_idx: int,
    head_idx: int,
    ax=None,
    title: str | None = None,
    annotate_wavelengths: bool = True,
):
    import matplotlib.pyplot as plt

    data = frequency_scores_for_head(freq_outputs, layer_idx, head_idx)
    pos = data["pos"].numpy()
    sym = data["sym"].numpy()
    wavelengths = data["wavelength"].numpy()
    x = list(range(len(pos)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    else:
        fig = ax.figure

    ax.plot(x, pos, marker="o", label="pos score")
    ax.plot(x, sym, marker="s", label="sym score")
    ax.set_xlabel("Frequency index")
    ax.set_ylabel("Score")
    ax.set_title(title or f"Layer {layer_idx}, head {head_idx}")
    ax.legend()
    ax.grid(alpha=0.3)

    if annotate_wavelengths:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_idx = x
        tick_labels = [f"{w:.1f}" for w in wavelengths]
        ax2.set_xticks(tick_idx)
        ax2.set_xticklabels(tick_labels, rotation=45, ha="left")
        ax2.set_xlabel("Wavelength (tokens / cycle)")

    fig.tight_layout()
    return ax

def plot_frequency_logit_norms(freq_outputs: FrequencyScoreOutputs, layer_idx: int, head_idx: int, ax=None, title: str | None = None):
    import matplotlib.pyplot as plt
    y = freq_outputs.frequency_logit_norms[layer_idx, head_idx].numpy()
    x = list(range(len(y)))
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("Frequency index")
    ax.set_ylabel("Logit norm")
    ax.set_title(title or f"Frequency logit norms, layer {layer_idx}, head {head_idx}")
    return ax


def plot_frequency_scores_and_norms(
    freq_outputs: FrequencyScoreOutputs,
    layer_idx: int,
    head_idx: int,
    title: str | None = None,
):
    import matplotlib.pyplot as plt

    data = frequency_scores_for_head(freq_outputs, layer_idx, head_idx)
    pos = data["pos"].numpy()
    sym = data["sym"].numpy()
    norm = data["norm"].numpy()
    wavelengths = data["wavelength"].numpy()
    x = list(range(len(pos)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(x, pos, marker="o", label="pos")
    axes[0].plot(x, sym, marker="s", label="sym")
    axes[0].set_ylabel("Score")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_title(title or f"Layer {layer_idx}, head {head_idx}")

    axes[1].plot(x, norm, marker="^", color="tab:purple", label="logit norm")
    axes[1].set_xlabel("Frequency index")
    axes[1].set_ylabel("Logit norm")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    ax_top = axes[0].twiny()
    ax_top.set_xlim(axes[0].get_xlim())
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"{w:.1f}" for w in wavelengths], rotation=45, ha="left")
    ax_top.set_xlabel("Wavelength (tokens / cycle)")

    fig.tight_layout()
    return axes