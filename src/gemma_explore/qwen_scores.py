from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# -----------------------------------------------------------------------------
# region 0) Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class QwenBundle:
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
    rope_theta: float
    q_group_size: int


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
class ScoreOutputs:
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


@dataclass
class FrequencyScoreOutputs:
    full_scores: torch.Tensor
    frequency_scores: torch.Tensor
    frequency_logit_norms: torch.Tensor
    frequency_wavelengths: torch.Tensor
    swaps: list[tuple[int, int]]
    blocks: list[tuple[int, int]]
    tokens: list[str]
    prompt_token_ids: list[int]
    prompt_text: str
    prompt_blocks: list[PromptBlock]


# -----------------------------------------------------------------------------
# region 1) Model + prompt
# -----------------------------------------------------------------------------


def load_bundle(
    model_name: str = DEFAULT_MODEL,
    device_map: str = "auto",
    torch_dtype: str | torch.dtype = "auto",
    trust_remote_code: bool = True,
    attn_implementation: str = "eager",
) -> QwenBundle:
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
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads

    return QwenBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        num_layers=cfg.num_hidden_layers,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        hidden_size=cfg.hidden_size,
        head_dim=head_dim,
        num_frequencies=head_dim // 2,
        rope_theta=getattr(cfg, "rope_theta", 1000000.0),
        q_group_size=num_heads // num_kv_heads,
    )


def encode_prompt(bundle: QwenBundle, prompt: str, add_generation_prompt: bool = True) -> PromptEncoding:
    messages = [{"role": "user", "content": prompt}]
    text = bundle.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

    enc = bundle.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(bundle.device)
    attention_mask = enc["attention_mask"].to(bundle.device)
    tokens = bundle.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    block_ids = input_ids[0].tolist()
    block_tokens = bundle.tokenizer.convert_ids_to_tokens(block_ids)

    blocks = [
        PromptBlock(
            name="chat_template",
            text=text,
            token_ids=block_ids,
            tokens=block_tokens,
            start=0,
            end=len(block_ids),
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


# endregion


# -----------------------------------------------------------------------------
# region 1bis) Prompt display helpers
# -----------------------------------------------------------------------------


def _normalize_token_ids(input_ids: torch.Tensor | list[int]) -> list[int]:
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 2:
            if input_ids.shape[0] != 1:
                raise ValueError(f"input_ids doit avoir batch_size=1, reçu {tuple(input_ids.shape)}")
            return input_ids[0].detach().cpu().tolist()
        if input_ids.ndim == 1:
            return input_ids.detach().cpu().tolist()
        raise ValueError(f"input_ids doit être de dimension 1 ou 2, reçu ndim={input_ids.ndim}")
    return list(input_ids)


def print_prompt_with_blocks(
    tokenizer,
    input_ids: torch.Tensor | list[int],
    blocks: list[tuple[int, int]],
    prefix: str = "# ",
    show_tokens: bool = False,
) -> None:
    """
    Affiche le prompt réellement envoyé au modèle, découpé bloc par bloc.

    Chaque bloc est imprimé sur une nouvelle ligne avec :
      - son texte décodé via repr(...) pour rendre visibles les coupures,
      - son index de bloc,
      - son intervalle de positions tokenisées [start:end].

    Si show_tokens=True, on affiche aussi les tokens bruts de chaque bloc.
    """
    ids = _normalize_token_ids(input_ids)

    print(f"{prefix}Prompt utilisé par le modèle, découpé en {len(blocks)} blocs :")
    for k, (start, end) in enumerate(blocks):
        if not (0 <= start <= end <= len(ids)):
            raise ValueError(
                f"Bloc invalide #{k}: ({start}, {end}) pour une séquence de longueur {len(ids)}"
            )

        block_ids = ids[start:end]
        block_text = tokenizer.decode(block_ids, skip_special_tokens=False)
        print(f"{repr(block_text)}  # bloc {k}, positions {start}:{end}")

        if show_tokens:
            block_tokens = tokenizer.convert_ids_to_tokens(block_ids)
            print(f"{prefix}tokens[{k}] = {block_tokens}")


def print_encoding_blocks(
    bundle: QwenBundle,
    encoding: PromptEncoding,
    blocks: list[tuple[int, int]],
    show_tokens: bool = False,
    prefix: str = "# ",
) -> None:
    print_prompt_with_blocks(
        tokenizer=bundle.tokenizer,
        input_ids=encoding.input_ids,
        blocks=blocks,
        prefix=prefix,
        show_tokens=show_tokens,
    )


# endregion


# -----------------------------------------------------------------------------
# region 2) Permutations / swaps
# -----------------------------------------------------------------------------


def split_into_blocks(n_tokens: int, n_blocks: int) -> list[tuple[int, int]]:
    n_blocks = min(n_blocks, n_tokens)
    base, rem = divmod(n_tokens, n_blocks)
    out = []
    start = 0
    for b in range(n_blocks):
        length = base + (1 if b < rem else 0)
        out.append((start, start + length))
        start += length
    return out


def all_block_swaps(n_blocks: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_blocks - 1) for j in range(i + 1, n_blocks)]


def build_swap_indices(seq_len: int, blocks: list[tuple[int, int]], swaps: list[tuple[int, int]]) -> torch.Tensor:
    col_idx = torch.arange(seq_len).unsqueeze(0).expand(len(swaps), -1).clone()
    for k, (bi, bj) in enumerate(swaps):
        si, ei = blocks[bi]
        sj, ej = blocks[bj]
        li, lj = ei - si, ej - sj
        if min(li, lj) == 0:
            raise ValueError(f"Empty block in swap ({bi}, {bj}).")
        lmin = min(li, lj)
        col_idx[k, si:si + lmin] = torch.arange(sj, sj + lmin)
        col_idx[k, sj:sj + lmin] = torch.arange(si, si + lmin)
    return col_idx


def permute_input_ids(input_ids: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return input_ids[:, perm]


# endregion


# -----------------------------------------------------------------------------
# region 3) Attention extraction
# -----------------------------------------------------------------------------


def get_decoder_layers(bundle: QwenBundle):
    return bundle.model.model.layers


def extract_last_token_attentions(
    attentions,
    num_layers: int,
    num_heads: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    if attentions is None or len(attentions) == 0:
        raise RuntimeError("No attentions returned; use attn_implementation='eager' and output_attentions=True.")

    cols = []
    for a in attentions:
        if a is None:
            raise RuntimeError("A layer returned attention=None.")
        cols.append(a[0, :, -1, :])

    out = torch.stack(cols, dim=0).to(device=device, dtype=torch.float32)
    if out.shape != (num_layers, num_heads, seq_len):
        raise RuntimeError(f"Unexpected attention shape {tuple(out.shape)}")
    return out


@torch.inference_mode()
def run_model(
    bundle: QwenBundle,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    output_hidden_states: bool = True,
    output_attentions: bool = True,
):
    return bundle.model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        output_attentions=output_attentions,
        return_dict=True,
        use_cache=False,
    )


def get_scores(
    att_last_col: torch.Tensor,
    swaps: list[tuple[int, int]],
    blocks: list[tuple[int, int]],
    tau: float = 0.1,
) -> torch.Tensor:
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
    vij_perm = torch.stack([block_avg_perm[..., swap_range, bj1], block_avg_perm[..., swap_range, bi1]], dim=-1)

    deltas = (block_avg_base[..., bi1] - block_avg_base[..., bj1]).abs()
    weights = F.softmax(deltas / tau, dim=-1)
    pos = F.cosine_similarity(vij_perm, vij_base, dim=-1)
    sym = F.cosine_similarity(vij_perm, torch.flip(vij_base, dims=[-1]), dim=-1)

    pos_scores = (weights * pos).sum(dim=-1).permute(0, 2, 1)
    sym_scores = (weights * sym).sum(dim=-1).permute(0, 2, 1)
    return torch.stack([pos_scores, sym_scores], dim=-1)


@torch.inference_mode()
def collect_scores(
    bundle: QwenBundle,
    prompt: str,
    n_blocks: int = 16,
    tau: float = 0.1,
    verbose_prompt: bool = False,
    verbose_prompt_tokens: bool = False,
) -> ScoreOutputs:
    encoding = encode_prompt(bundle, prompt)
    seq_len = encoding.input_ids.shape[1]
    blocks = split_into_blocks(seq_len, n_blocks)

    if verbose_prompt:
        print_encoding_blocks(
            bundle=bundle,
            encoding=encoding,
            blocks=blocks,
            show_tokens=verbose_prompt_tokens,
        )

    swaps = all_block_swaps(len(blocks))
    col_idx = build_swap_indices(seq_len, blocks, swaps).to(bundle.device)

    att_last_col = torch.empty(
        bundle.num_layers, 1, len(swaps) + 1, bundle.num_heads, seq_len,
        device=bundle.device, dtype=torch.float32
    )

    for swap_idx in range(len(swaps) + 1):
        ids = encoding.input_ids if swap_idx == 0 else permute_input_ids(encoding.input_ids, col_idx[swap_idx - 1])
        mask = torch.ones_like(ids)

        out = run_model(
            bundle,
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=False,
            output_attentions=True,
        )

        att_last_col[:, 0, swap_idx] = extract_last_token_attentions(
            out.attentions,
            bundle.num_layers,
            bundle.num_heads,
            seq_len,
            bundle.device,
        )

    scores = get_scores(att_last_col, swaps, blocks, tau=tau).detach().cpu()
    return ScoreOutputs(
        att_last_col=att_last_col.detach().cpu(),
        swaps=swaps,
        blocks=blocks,
        pos_scores=scores[..., 0],
        sym_scores=scores[..., 1],
        mean_scores=scores.mean(dim=2),
        tokens=encoding.tokens,
        prompt_token_ids=encoding.input_ids[0].tolist(),
        prompt_text=encoding.text,
        prompt_blocks=encoding.blocks,
    )


# endregion


# -----------------------------------------------------------------------------
# region 4) RoPE frequency analysis (optimized)
# -----------------------------------------------------------------------------


def rope_wavelengths(bundle: QwenBundle) -> torch.Tensor:
    idx = torch.arange(bundle.num_frequencies, dtype=torch.float32, device=bundle.device)
    inv_freq = 1.0 / (bundle.rope_theta ** (idx / bundle.num_frequencies))
    return (2.0 * math.pi) / inv_freq


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv_local(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, nkv, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, nkv, n_rep, seq_len, head_dim)
    return hidden_states.reshape(bsz, nkv * n_rep, seq_len, head_dim)


def compute_qk_for_layer(bundle: QwenBundle, hidden_in: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_decoder_layers(bundle)[layer_idx]
    attn = layer.self_attn

    bsz, seq_len, _ = hidden_in.shape
    device = attn.q_proj.weight.device
    dtype = attn.q_proj.weight.dtype

    x = layer.input_layernorm(hidden_in.to(device=device, dtype=dtype))

    q = attn.q_proj(x).view(bsz, seq_len, bundle.num_heads, bundle.head_dim).transpose(1, 2).contiguous()
    k = attn.k_proj(x).view(bsz, seq_len, bundle.num_kv_heads, bundle.head_dim).transpose(1, 2).contiguous()

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cos, sin = bundle.model.model.rotary_emb(x, position_ids)
    q, k = apply_rotary(q, k, cos, sin)

    k = repeat_kv_local(k, bundle.q_group_size)
    return q, k


def compute_all_frequency_attentions_and_norms(
    q: torch.Tensor,
    k: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = q.shape[-1] // 2

    q_pairs = torch.stack([q[..., :half], q[..., half:]], dim=-1)
    k_pairs = torch.stack([k[..., :half], k[..., half:]], dim=-1)

    q_last = q_pairs[:, :, -1]
    logits = torch.einsum("bhfd,bhtfd->bhft", q_last, k_pairs)
    logits = logits / math.sqrt(head_dim)

    att = torch.softmax(logits, dim=-1)
    norms = torch.linalg.vector_norm(logits.float(), ord=2, dim=-1)
    return att, norms


@torch.inference_mode()
def collect_frequency_scores(
    bundle: QwenBundle,
    prompt: str,
    n_blocks: int = 16,
    tau: float = 0.1,
    verbose_prompt: bool = False,
    verbose_prompt_tokens: bool = False,
    store_full_frequency_tensor: bool = True,
    freq_chunk_size: int | None = None,
) -> FrequencyScoreOutputs:
    encoding = encode_prompt(bundle, prompt)
    seq_len = encoding.input_ids.shape[1]
    blocks = split_into_blocks(seq_len, n_blocks)

    if verbose_prompt:
        print_encoding_blocks(
            bundle=bundle,
            encoding=encoding,
            blocks=blocks,
            show_tokens=verbose_prompt_tokens,
        )

    swaps = all_block_swaps(len(blocks))
    col_idx = build_swap_indices(seq_len, blocks, swaps).to(bundle.device)

    n_swaps = len(swaps)
    nl = bundle.num_layers
    nh = bundle.num_heads
    nf = bundle.num_frequencies

    full_att_last_col = torch.empty(
        nl, 1, n_swaps + 1, nh, seq_len,
        device=bundle.device, dtype=torch.float32
    )

    freq_logit_norms = torch.empty(
        nl, nh, nf,
        device="cpu", dtype=torch.float32
    )

    if freq_chunk_size is None:
        freq_chunk_size = nf

    freq_att_last_col = None
    if store_full_frequency_tensor:
        freq_att_last_col = torch.empty(
            nl, 1, n_swaps + 1, nh, nf, seq_len,
            device=bundle.device, dtype=torch.float32
        )

    frequency_scores = torch.empty(
        nl, nh, nf, 2,
        device="cpu", dtype=torch.float32
    )

    for swap_idx in range(n_swaps + 1):
        ids = encoding.input_ids if swap_idx == 0 else permute_input_ids(encoding.input_ids, col_idx[swap_idx - 1])
        mask = torch.ones_like(ids)

        out = run_model(
            bundle,
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            output_attentions=True,
        )

        full_att_last_col[:, 0, swap_idx] = extract_last_token_attentions(
            out.attentions, nl, nh, seq_len, bundle.device
        )

        hidden_states = out.hidden_states[:-1]

        for layer_idx, hidden_in in enumerate(hidden_states):
            q, k = compute_qk_for_layer(bundle, hidden_in, layer_idx)
            att_f, norms_f = compute_all_frequency_attentions_and_norms(q, k, bundle.head_dim)

            if freq_att_last_col is not None:
                freq_att_last_col[layer_idx, 0, swap_idx] = att_f[0].float()

            if swap_idx == 0:
                freq_logit_norms[layer_idx] = norms_f[0].float().cpu()

        del out, hidden_states

    full_scores = get_scores(full_att_last_col, swaps, blocks, tau=tau).detach().cpu()
    full_scores = full_scores.squeeze(dim=-2)

    if freq_att_last_col is not None:
        freq_flat = freq_att_last_col.permute(0, 1, 2, 4, 3, 5).reshape(
            nl, 1, n_swaps + 1, nf * nh, seq_len
        )
        scores_f = get_scores(freq_flat, swaps, blocks, tau=tau)
        scores_f = scores_f.mean(dim=2).reshape(nl, nf, nh, 2).permute(0, 2, 1, 3).contiguous()
        frequency_scores.copy_(scores_f.detach().cpu())
        del freq_att_last_col, freq_flat, scores_f

    else:
        for f0 in range(0, nf, freq_chunk_size):
            f1 = min(f0 + freq_chunk_size, nf)
            chunk = f1 - f0

            freq_chunk_att = torch.empty(
                nl, 1, n_swaps + 1, nh, chunk, seq_len,
                device=bundle.device, dtype=torch.float32
            )

            for swap_idx in range(n_swaps + 1):
                ids = encoding.input_ids if swap_idx == 0 else permute_input_ids(encoding.input_ids, col_idx[swap_idx - 1])
                mask = torch.ones_like(ids)

                out = run_model(
                    bundle,
                    input_ids=ids,
                    attention_mask=mask,
                    output_hidden_states=True,
                    output_attentions=False,
                )
                hidden_states = out.hidden_states[:-1]

                for layer_idx, hidden_in in enumerate(hidden_states):
                    q, k = compute_qk_for_layer(bundle, hidden_in, layer_idx)

                    half = q.shape[-1] // 2
                    q_pairs = torch.stack([q[..., f0:f1], q[..., half + f0:half + f1]], dim=-1)
                    k_pairs = torch.stack([k[..., f0:f1], k[..., half + f0:half + f1]], dim=-1)

                    q_last = q_pairs[:, :, -1]
                    logits = torch.einsum("bhfd,bhtfd->bhft", q_last, k_pairs) / math.sqrt(bundle.head_dim)
                    att = torch.softmax(logits, dim=-1)

                    freq_chunk_att[layer_idx, 0, swap_idx] = att[0].float()

                del out, hidden_states

            chunk_flat = freq_chunk_att.permute(0, 1, 2, 4, 3, 5).reshape(
                nl, 1, n_swaps + 1, chunk * nh, seq_len
            )
            scores_chunk = get_scores(chunk_flat, swaps, blocks, tau=tau)
            scores_chunk = scores_chunk.mean(dim=2).reshape(nl, chunk, nh, 2).permute(0, 2, 1, 3).contiguous()
            frequency_scores[:, :, f0:f1] = scores_chunk.detach().cpu()

            del freq_chunk_att, chunk_flat, scores_chunk

    return FrequencyScoreOutputs(
        full_scores=full_scores,
        frequency_scores=frequency_scores,
        frequency_logit_norms=freq_logit_norms,
        frequency_wavelengths=rope_wavelengths(bundle).detach().cpu(),
        swaps=swaps,
        blocks=blocks,
        tokens=encoding.tokens,
        prompt_token_ids=encoding.input_ids[0].tolist(),
        prompt_text=encoding.text,
        prompt_blocks=encoding.blocks,
    )


# endregion


# -----------------------------------------------------------------------------
# region 5) Plots (matplotlib/seaborn version)
# -----------------------------------------------------------------------------


def _as_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _layer_head_labels(bundle: QwenBundle):
    layers = [f"L{l}" for l in range(bundle.num_layers)]
    heads = [f"H{h}" for h in range(bundle.num_heads)]
    return layers, heads


def _score_title(prefix: str, n_tokens: int | None = None):
    """
    Titre de figure sans le prompt complet.
    On montre juste le nombre de tokens si on le connaît.
    """
    if n_tokens is None:
        return prefix
    return f"{prefix} — {n_tokens} tokens"


def _prepare_score_matrix(arr, expected_shape: tuple[int, int], name: str):
    arr = _as_numpy(arr)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.ndim != 2:
        raise ValueError(
            f"{name} doit avoir la shape [num_layers, num_heads] "
            f"ou [num_layers, num_heads, 1], reçu {arr.shape}"
        )

    if arr.shape != expected_shape:
        raise ValueError(
            f"{name} a la mauvaise shape: attendu {expected_shape}, reçu {arr.shape}"
        )

    return np.asarray(arr, dtype=float)


def _save_figure_with_meta(fig, save_path: str | None, caption: str, description: str):
    if not save_path:
        return

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    with open(str(path) + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "caption": caption,
                "description": description,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def _maybe_print_scores_prompt(
    bundle: QwenBundle,
    scores: ScoreOutputs,
    verbose_prompt: bool,
    verbose_prompt_tokens: bool,
):
    if not verbose_prompt:
        return

    print_prompt_with_blocks(
        tokenizer=bundle.tokenizer,
        input_ids=scores.prompt_token_ids,
        blocks=scores.blocks,
        show_tokens=verbose_prompt_tokens,
    )


def _maybe_print_frequency_prompt(
    bundle: QwenBundle,
    freq: FrequencyScoreOutputs,
    verbose_prompt: bool,
    verbose_prompt_tokens: bool,
):
    if not verbose_prompt:
        return

    print_prompt_with_blocks(
        tokenizer=bundle.tokenizer,
        input_ids=freq.prompt_token_ids,
        blocks=freq.blocks,
        show_tokens=verbose_prompt_tokens,
    )


def plot_pos_sym_heatmaps(
    scores: ScoreOutputs,
    bundle: QwenBundle,
    save_path: str | None = None,
    n_tokens: int | None = None,
    verbose_prompt: bool = False,
    verbose_prompt_tokens: bool = False,
):
    """
    Heatmaps des scores positionnel et symbolique par couche et tête.

    Le titre ne montre pas le prompt, juste un indicateur court (ex: nombre de tokens).
    Le prompt réel est affiché via print() si verbose_prompt=True.
    """
    _maybe_print_scores_prompt(
        bundle=bundle,
        scores=scores,
        verbose_prompt=verbose_prompt,
        verbose_prompt_tokens=verbose_prompt_tokens,
    )

    layers, heads = _layer_head_labels(bundle)
    expected_shape = (bundle.num_layers, bundle.num_heads)

    pos = _prepare_score_matrix(scores.pos_scores, expected_shape, "scores.pos_scores")
    sym = _prepare_score_matrix(scores.sym_scores, expected_shape, "scores.sym_scores")

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    cmap = mpl.colormaps["viridis"]

    sns.heatmap(
        pos,
        xticklabels=heads,
        yticklabels=layers,
        ax=axes[0],
        cmap=cmap,
        cbar_kws={"label": "Score"},
        linewidths=0.25,
        linecolor="white",
    )
    axes[0].set_title("Score positionnel")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    axes[0].invert_yaxis()

    sns.heatmap(
        sym,
        xticklabels=heads,
        yticklabels=layers,
        ax=axes[1],
        cmap=cmap,
        cbar_kws={"label": "Score"},
        linewidths=0.25,
        linecolor="white",
    )
    axes[1].set_title("Score symbolique")
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    axes[1].invert_yaxis()

    if n_tokens is None:
        n_tokens = len(scores.prompt_token_ids)

    fig.suptitle(_score_title("Scores par couche et tête", n_tokens), fontsize=14)

    _save_figure_with_meta(
        fig,
        save_path,
        caption="Heatmaps des scores positionnel et symbolique",
        description="Deux heatmaps montrant les scores par couche et par tête.",
    )
    return fig


def plot_heads_scatter(
    scores: ScoreOutputs,
    bundle: QwenBundle,
    save_path: str | None = None,
    n_tokens: int | None = None,
    verbose_prompt: bool = False,
    verbose_prompt_tokens: bool = False,
):
    """
    Nuage de points des heads dans le plan (score positionnel, score symbolique),
    coloré par couche.

    Le titre ne montre pas le prompt, juste un indicateur court (ex: nombre de tokens).
    Le prompt réel est affiché via print() si verbose_prompt=True.
    """
    _maybe_print_scores_prompt(
        bundle=bundle,
        scores=scores,
        verbose_prompt=verbose_prompt,
        verbose_prompt_tokens=verbose_prompt_tokens,
    )

    expected_shape = (bundle.num_layers, bundle.num_heads)

    pos = _prepare_score_matrix(scores.pos_scores, expected_shape, "scores.pos_scores").reshape(-1)
    sym = _prepare_score_matrix(scores.sym_scores, expected_shape, "scores.sym_scores").reshape(-1)

    layer_ids = np.repeat(np.arange(bundle.num_layers), bundle.num_heads)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)

    scatter = ax.scatter(
        pos,
        sym,
        c=layer_ids,
        cmap=mpl.colormaps["viridis"],
        alpha=0.8,
        s=45,
        edgecolors="black",
        linewidths=0.3,
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Layer")

    ax.set_xlabel("Score positionnel")
    ax.set_ylabel("Score symbolique")

    if n_tokens is None:
        n_tokens = len(scores.prompt_token_ids)
    ax.set_title(_score_title("Heads dans le plan des scores", n_tokens))
    ax.grid(True, linestyle="--", alpha=0.3)

    _save_figure_with_meta(
        fig,
        save_path,
        caption="Nuage de points des heads",
        description="Position des heads dans le plan score positionnel-score symbolique, colorées par couche.",
    )
    return fig


def plot_frequency_analysis(
    freq: FrequencyScoreOutputs,
    layer_idx: int,
    head_idx: int,
    bundle: QwenBundle,
    save_path: str | None = None,
    verbose_prompt: bool = False,
    verbose_prompt_tokens: bool = False,
):
    """
    Analyse fréquentielle pour une tête donnée dans une seule figure :
      - haut : scores positionnel et symbolique par index de fréquence
      - bas  : normes par index de fréquence

    Le titre affiche les vrais scores globaux de la tête étudiée
    (positionnel et symbolique), sans le prompt.
    Le prompt réel est affiché via print() si verbose_prompt=True.
    """
    _maybe_print_frequency_prompt(
        bundle=bundle,
        freq=freq,
        verbose_prompt=verbose_prompt,
        verbose_prompt_tokens=verbose_prompt_tokens,
    )

    wavelengths = np.asarray(_as_numpy(freq.frequency_wavelengths), dtype=float)

    freq_scores = np.asarray(_as_numpy(freq.frequency_scores), dtype=float)
    norm_all = np.asarray(_as_numpy(freq.frequency_logit_norms), dtype=float)
    full_scores = np.asarray(_as_numpy(freq.full_scores), dtype=float)

    if not (0 <= layer_idx < bundle.num_layers):
        raise IndexError(f"layer_idx={layer_idx} hors limites pour {bundle.num_layers} couches")
    if not (0 <= head_idx < bundle.num_heads):
        raise IndexError(f"head_idx={head_idx} hors limites pour {bundle.num_heads} têtes")

    if (
        freq_scores.ndim != 4
        or freq_scores.shape[:2] != (bundle.num_layers, bundle.num_heads)
        or freq_scores.shape[-1] != 2
    ):
        raise ValueError(
            "freq.frequency_scores doit avoir la shape "
            f"[num_layers, num_heads, num_freq, 2], reçu {freq_scores.shape}"
        )

    if norm_all.ndim != 3 or norm_all.shape[:2] != (bundle.num_layers, bundle.num_heads):
        raise ValueError(
            "freq.frequency_logit_norms doit avoir la shape "
            f"[num_layers, num_heads, num_freq], reçu {norm_all.shape}"
        )

    if full_scores.ndim != 3 or full_scores.shape != (bundle.num_layers, bundle.num_heads, 2):
        raise ValueError(
            "freq.full_scores doit avoir la shape "
            f"[num_layers, num_heads, 2], reçu {full_scores.shape}"
        )

    if len(wavelengths) != freq_scores.shape[2] or len(wavelengths) != norm_all.shape[2]:
        raise ValueError(
            "Incohérence entre le nombre de wavelengths et le nombre de fréquences "
            f"({len(wavelengths)} vs {freq_scores.shape[2]} vs {norm_all.shape[2]})"
        )

    head_freq_scores = freq_scores[layer_idx, head_idx]
    norm_mean = norm_all.mean(axis=(0, 1))
    norm_orig = norm_all[layer_idx, head_idx]

    head_full_score = full_scores[layer_idx, head_idx]
    head_pos_score = float(head_full_score[0])
    head_sym_score = float(head_full_score[1])

    x = np.arange(len(wavelengths))

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.0]},
        constrained_layout=True,
    )
    ax1, ax2 = axes

    ax1.plot(
        x, head_freq_scores[:, 0],
        marker="o", markersize=4, linewidth=1.8,
        label="Positionnel"
    )
    ax1.plot(
        x, head_freq_scores[:, 1],
        marker="s", markersize=4, linewidth=1.8,
        label="Symbolique"
    )
    ax1.set_ylabel("Score")
    ax1.set_title("Scores par fréquence")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(
        x, norm_mean,
        marker="o", markersize=4, linewidth=1.8,
        label="Norme moyenne"
    )
    ax2.plot(
        x, norm_orig,
        marker="s", markersize=4, linewidth=1.8,
        label="Norme origine"
    )
    ax2.set_xlabel("Fréquence (index et longueur d'onde)")
    ax2.set_ylabel("Norme")
    ax2.set_title("Normes par fréquence")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="best")

    tick_labels = [f"{i} | λ={w:.2g}" for i, w in enumerate(wavelengths)]
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        tick_labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=8,
    )

    fig.suptitle(
        (
            f"Analyse fréquentielle — Couche {layer_idx}, tête {head_idx} | "
            f"score positionnel={head_pos_score:.4f} | "
            f"score symbolique={head_sym_score:.4f}"
        ),
        fontsize=13,
    )

    _save_figure_with_meta(
        fig,
        save_path,
        caption="Analyse fréquentielle d'une tête",
        description=(
            "Figure unique avec scores positionnel et symbolique par fréquence, "
            "ainsi que les normes par index de fréquence. Le titre affiche les "
            "vrais scores globaux positionnel et symbolique de la tête étudiée, "
            "sans le prompt."
        ),
    )

    return fig, head_pos_score, head_sym_score


# endregion