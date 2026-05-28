# qwen_viz.py
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from gemma_explore.qwen_cache import CacheHandle, get_layer_record, get_prompt_record


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def short_tok(token: str, max_len: int = 18) -> str:
    token = token.replace("Ġ", "▁").replace("Ċ", "\\n")
    return token if len(token) <= max_len else token[: max_len - 1] + "…"


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def _prepare_tokens(tokens: Sequence[str], max_len: int = 18, max_tokens: int | None = None) -> list[str]:
    out = [short_tok(t, max_len=max_len) for t in tokens]
    if max_tokens is not None:
        out = out[:max_tokens]
    return out


def _resolve_prompt(cache_or_prompt: CacheHandle | Mapping[str, Any], prompt_id: int | None = None) -> dict[str, Any]:
    if isinstance(cache_or_prompt, CacheHandle):
        if prompt_id is None:
            raise ValueError("prompt_id is required when passing a CacheHandle")
        return get_prompt_record(cache_or_prompt, prompt_id)
    if prompt_id is not None and "prompts" in cache_or_prompt:
        return get_prompt_record(CacheHandle(dict(cache_or_prompt)), prompt_id)
    return dict(cache_or_prompt)


def _resolve_layer(
    cache_or_prompt_or_layer: CacheHandle | Mapping[str, Any],
    prompt_id: int | None = None,
    layer_idx: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(cache_or_prompt_or_layer, CacheHandle):
        if prompt_id is None or layer_idx is None:
            raise ValueError("prompt_id and layer_idx are required with CacheHandle")
        return get_prompt_record(cache_or_prompt_or_layer, prompt_id), get_layer_record(cache_or_prompt_or_layer, prompt_id, layer_idx)

    obj = dict(cache_or_prompt_or_layer)
    if "layers" in obj and layer_idx is not None:
        return obj, obj["layers"][layer_idx]
    if "block_input" in obj:
        return {}, obj
    raise ValueError("Could not resolve layer record")


def _heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    *,
    title: str,
    cmap: str = "coolwarm",
    center: float | None = 0.0,
    xticklabels: Sequence[str] | bool = False,
    yticklabels: Sequence[str] | bool = False,
    square: bool = False,
    cbar: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    sns.heatmap(
        data,
        ax=ax,
        cmap=cmap,
        center=center,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cbar=cbar,
        square=square,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title, fontsize=11)


def _highlight_cols(
    ax: plt.Axes,
    start: int,
    end: int,
    nrows: int,
    *,
    label: str | None = None,
    color: str = "cyan",
    lw: float = 2.0,
) -> None:
    rect = mpatches.Rectangle(
        (start, 0),
        end - start,
        nrows,
        linewidth=lw,
        edgecolor=color,
        facecolor="none",
        transform=ax.transData,
        clip_on=False,
        label=label,
    )
    ax.add_patch(rect)
    if label:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)


def _score_matrix(scores: Any, expected_shape: tuple[int, int], name: str) -> np.ndarray:
    arr = _as_numpy(scores)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [layers, heads], got {arr.shape}")
    if arr.shape != expected_shape:
        raise ValueError(f"{name} expected {expected_shape}, got {arr.shape}")
    return arr.astype(float)


# ---------------------------------------------------------------------
# Cache visualizations
# ---------------------------------------------------------------------


def plot_all_attention_heads(
    cache: CacheHandle | Mapping[str, Any],
    *,
    prompt_id: int,
    layer_idx: int,
    max_label_len: int = 18,
    figsize_per_panel: float = 5.5,
    ncols: int = 2,
    vmax: float | None = None,
) -> plt.Figure:
    """Plot all Q-head attention heatmaps for one prompt/layer."""
    prompt = _resolve_prompt(cache, prompt_id)
    layer = prompt["layers"][layer_idx]
    tokens = prompt["tokens"]
    labels = _prepare_tokens(tokens, max_len=max_label_len)

    attn = _as_numpy(layer["attn_weights"])[0].astype(float)
    num_heads, _, _ = attn.shape
    kv_map = _as_numpy(layer["kv_head_for_q"]).astype(int).tolist()

    if vmax is None:
        vmax = float(attn.max())

    nrows = math.ceil(num_heads / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * figsize_per_panel, nrows * figsize_per_panel),
        squeeze=False,
    )

    for h in range(nrows * ncols):
        ax = axes[h // ncols][h % ncols]
        if h >= num_heads:
            ax.axis("off")
            continue
        _heatmap(
            ax,
            attn[h],
            title=f"Head q={h} → kv={kv_map[h]}",
            cmap="magma",
            center=None,
            xticklabels=labels,
            yticklabels=labels,
            square=True,
            cbar=False,
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_xlabel("Source token")
        ax.set_ylabel("Target token")
        ax.tick_params(axis="x", rotation=90, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    fig.suptitle(f"Attention heatmaps | prompt={prompt_id} | layer={layer_idx}", y=1.02, fontsize=13)
    plt.tight_layout()
    return fig


def plot_head_block_dynamics(
    cache: CacheHandle | Mapping[str, Any],
    *,
    prompt_id: int,
    layer_idx: int,
    head_idx: int,
    batch_idx: int = 0,
    max_tokens: int | None = 48,
    max_label_len: int = 18,
    figsize: tuple[float, float] | None = None,
    highlight_q_slice: bool = True,
    highlight_kv_slice: bool = True,
) -> plt.Figure:
    """Plot the detailed flat notebook view for one prompt/layer/head."""
    prompt, layer = _resolve_layer(cache, prompt_id=prompt_id, layer_idx=layer_idx)

    tokens_raw = prompt["tokens"]
    T = len(tokens_raw) if max_tokens is None else min(len(tokens_raw), max_tokens)
    labels = _prepare_tokens(tokens_raw[:T], max_len=max_label_len)

    block_input = _as_numpy(layer["block_input"])[batch_idx, :T].astype(float)
    hidden_size = block_input.shape[-1]
    head_dim = int(_as_numpy(layer["head_out"]).shape[-1])
    kv_head_idx = int(_as_numpy(layer["kv_head_for_q"])[head_idx])

    q_col_start = head_idx * head_dim
    q_col_end = q_col_start + head_dim
    kv_col_start = kv_head_idx * head_dim
    kv_col_end = kv_col_start + head_dim

    def get(key: str, sub_head: int | None = None) -> np.ndarray:
        arr = _as_numpy(layer[key])[batch_idx, :T].astype(float)
        if sub_head is not None:
            arr = arr[:, sub_head]
        return arr

    attn_weights = _as_numpy(layer["attn_weights"])[batch_idx, head_idx, :T, :T].astype(float)
    rows = [
        (attn_weights, f"Attention weights — q head {head_idx}", None, "viridis", None),
        (get("block_input"), "Block input", (q_col_start, q_col_end) if highlight_q_slice else None, "coolwarm", 0.0),
        (get("input_layernorm_out"), "Input layernorm out", (q_col_start, q_col_end) if highlight_q_slice else None, "coolwarm", 0.0),
        (get("v_pre", sub_head=kv_head_idx), f"V pre — kv head {kv_head_idx}", (0, head_dim) if highlight_kv_slice else None, "coolwarm", 0.0),
        (get("head_out", sub_head=head_idx), f"Head output — q head {head_idx}", (0, head_dim) if highlight_q_slice else None, "coolwarm", 0.0),
        (get("residual_after_attn"), "Residual after attention", (q_col_start, q_col_end) if highlight_q_slice else None, "coolwarm", 0.0),
        (get("post_attn_layernorm_out"), "Post-attention layernorm out", None, "coolwarm", 0.0),
        (get("mlp_out"), "MLP out", None, "coolwarm", 0.0),
        (get("block_output"), "Block output", None, "coolwarm", 0.0),
    ]

    if figsize is None:
        figsize = (24, 7 * len(rows))

    fig, axes = plt.subplots(len(rows), 1, figsize=figsize, constrained_layout=True)
    fig.suptitle(
        f"Prompt {prompt_id} | Layer {layer_idx} | Q-head {head_idx} (KV head {kv_head_idx}) — {T} tokens",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (data, title, col_range, cmap, center) in zip(axes, rows):
        yticklabels = labels if data.shape[0] == T else False
        xticklabels = labels if data.shape == (T, T) else False
        _heatmap(
            ax,
            data,
            title=title,
            cmap=cmap,
            center=center,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            square=False,
            cbar=True,
        )
        if col_range is not None and data.ndim == 2 and data.shape[1] >= col_range[1]:
            label = f"cols [{col_range[0]}:{col_range[1]}]"
            if data.shape[1] == head_dim:
                label = f"head slice [{col_range[0]}:{col_range[1]}]"
            _highlight_cols(ax, col_range[0], col_range[1], data.shape[0], label=label)
        ax.set_xlabel("dim →" if data.ndim == 2 and data.shape[1] != T else "source token →")
        ax.set_ylabel("tokens →")

    return fig


# ---------------------------------------------------------------------
# Score visualizations
# ---------------------------------------------------------------------


def plot_pos_sym_heatmaps(
    scores: Mapping[str, Any] | Any,
    *,
    num_layers: int,
    num_heads: int,
    title: str = "Scores by layer and head",
) -> plt.Figure:
    """Plot positional and symbolic score heatmaps."""
    pos = _score_matrix(scores["pos_scores"] if isinstance(scores, Mapping) else scores.pos_scores, (num_layers, num_heads), "pos_scores")
    sym = _score_matrix(scores["sym_scores"] if isinstance(scores, Mapping) else scores.sym_scores, (num_layers, num_heads), "sym_scores")

    layers = [f"L{l}" for l in range(num_layers)]
    heads = [f"H{h}" for h in range(num_heads)]

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    _heatmap(
        axes[0],
        pos,
        title="Positional score",
        cmap="viridis",
        center=None,
        xticklabels=heads,
        yticklabels=layers,
        square=False,
        cbar=True,
    )
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    axes[0].invert_yaxis()

    _heatmap(
        axes[1],
        sym,
        title="Symbolic score",
        cmap="viridis",
        center=None,
        xticklabels=heads,
        yticklabels=layers,
        square=False,
        cbar=True,
    )
    axes[1].set_xlabel("Head")
    axes[1].set_ylabel("Layer")
    axes[1].invert_yaxis()

    fig.suptitle(title, fontsize=14)
    return fig


def plot_heads_scatter(
    scores: Mapping[str, Any] | Any,
    *,
    num_layers: int,
    num_heads: int,
    title: str = "Heads in positional/symbolic score plane",
) -> plt.Figure:
    """Scatter plot of heads in the positional vs symbolic plane."""
    pos = _score_matrix(scores["pos_scores"] if isinstance(scores, Mapping) else scores.pos_scores, (num_layers, num_heads), "pos_scores").reshape(-1)
    sym = _score_matrix(scores["sym_scores"] if isinstance(scores, Mapping) else scores.sym_scores, (num_layers, num_heads), "sym_scores").reshape(-1)

    layer_ids = np.repeat(np.arange(num_layers), num_heads)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    scatter = ax.scatter(
        pos,
        sym,
        c=layer_ids,
        cmap=mpl.colormaps["viridis"],
        alpha=0.85,
        s=45,
        edgecolors="black",
        linewidths=0.3,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Layer")
    ax.set_xlabel("Positional score")
    ax.set_ylabel("Symbolic score")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig


# ---------------------------------------------------------------------
# Frequency visualization
# ---------------------------------------------------------------------


def plot_frequency_analysis(
    freq_scores: Mapping[str, Any] | Any,
    *,
    layer_idx: int,
    head_idx: int,
    max_label_len: int = 18,
    max_tokens: int | None = 64,
) -> plt.Figure:
    """Plot frequency-resolved scores, norms, and original full attention."""
    data = freq_scores if isinstance(freq_scores, Mapping) else freq_scores.__dict__
    frequency_scores = _as_numpy(data["frequency_scores"]).astype(float)
    logit_norms = _as_numpy(data["frequency_logit_norms"]).astype(float)
    wavelengths = _as_numpy(data["frequency_wavelengths"]).astype(float)
    full_scores = _as_numpy(data["full_scores"]).astype(float)
    attention_matrix = _as_numpy(data["attention_matrix"]).astype(float)
    tokens = list(data["tokens"])

    if frequency_scores.ndim != 4:
        raise ValueError(f"Expected frequency_scores [layers, heads, freq, 2], got {frequency_scores.shape}")
    if logit_norms.ndim != 4:
        raise ValueError(f"Expected frequency_logit_norms [layers, swaps+1, heads, freq], got {logit_norms.shape}")

    T = len(tokens) if max_tokens is None else min(len(tokens), max_tokens)
    labels = _prepare_tokens(tokens[:T], max_len=max_label_len)
    pos_f = frequency_scores[layer_idx, head_idx, :, 0]
    sym_f = frequency_scores[layer_idx, head_idx, :, 1]
    norms_f = logit_norms[layer_idx, 0, head_idx]
    attn = attention_matrix[layer_idx, head_idx, :T, :T]
    global_pos = float(full_scores[layer_idx, head_idx, 0])
    global_sym = float(full_scores[layer_idx, head_idx, 1])

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3])

    ax0 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(wavelengths))
    ax0.plot(x, pos_f, marker="o", label="Positional")
    ax0.plot(x, sym_f, marker="o", label="Symbolic")
    ax0.axhline(global_pos, linestyle="--", alpha=0.5, label="Global pos")
    ax0.axhline(global_sym, linestyle="--", alpha=0.5, label="Global sym")
    ax0.set_title(f"Per-frequency scores | L{layer_idx} H{head_idx}")
    ax0.set_xlabel("Frequency index")
    ax0.set_ylabel("Score")
    ax0.legend()

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(x, norms_f, marker="o", color="tab:purple")
    ax1.set_title("Per-frequency logit norms")
    ax1.set_xlabel("Frequency index")
    ax1.set_ylabel("L2 norm")

    tick_idx = np.linspace(0, len(wavelengths) - 1, min(8, len(wavelengths)), dtype=int)
    tick_labels = [f"{wavelengths[i]:.2g}" for i in tick_idx]
    ax0.set_xticks(tick_idx, tick_labels, rotation=35, ha="right")
    ax1.set_xticks(tick_idx, tick_labels, rotation=35, ha="right")

    ax2 = fig.add_subplot(gs[1, :])
    _heatmap(
        ax2,
        attn,
        title="Full attention heatmap (original prompt)",
        cmap="magma",
        center=None,
        xticklabels=labels,
        yticklabels=labels,
        square=False,
        cbar=True,
        vmin=0.0,
        vmax=float(attn.max()) if attn.size else None,
    )
    ax2.set_xlabel("Source token")
    ax2.set_ylabel("Target token")
    ax2.tick_params(axis="x", rotation=90, labelsize=8)
    ax2.tick_params(axis="y", rotation=0, labelsize=8)

    fig.suptitle(
        f"Frequency analysis | layer={layer_idx} | head={head_idx} | wavelengths shown on x-ticks",
        fontsize=13,
    )
    return fig