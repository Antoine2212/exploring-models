# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Mechanistic interpretability toolkit for analyzing attention heads in **Qwen2.5-0.5B-Instruct**. The core experiment: split a prompt into token blocks, run all pairwise block swaps, and measure whether each attention head follows token *position* or token *identity* (symbol). Results are exposed via an interactive Panel/Bokeh dashboard.

## Commands

```bash
# Install / sync dependencies
uv sync

# Launch the interactive dashboard
uv run gemma-dashboard

# Run RULER benchmark evaluation
uv run python scripts/ruler_eval_qwen.py

# Load model and run a quick generation sanity check
uv run python tests/test_model.py
```

Tests are scripts rather than a pytest suite — run them directly with `uv run python tests/<file>.py`.

## Architecture

```
src/gemma_explore/
  qwen_core.py        # Central: load_bundle(), collect_scores(), collect_frequency_scores()
  qwen_cache.py       # Disk cache keyed by model ID + prompt hash → data/cache/
  qwen_viz.py         # Standalone matplotlib/seaborn plotting helpers
  dashboard/
    _launcher.py      # Entry point: `gemma-dashboard` CLI script
    app.py            # Panel app wiring — assembles views into tabs
    state.py          # Shared reactive state (AppState) consumed by all views
    prompt_registry.py# Pre-defined example prompts
    widgets.py        # Shared UI widget definitions
    views/            # One Panel component per dashboard tab
```

### Key data flow

1. `load_bundle()` — loads the model + tokenizer into a `QwenBundle` (holds model metadata: layers, heads, GQA config, RoPE params).
2. `collect_scores(bundle, prompt, n_blocks)` → `ScoreResult` — runs all pairwise block swaps and returns `pos_scores` / `sym_scores` of shape `[layers, heads, 1]`.
3. `collect_frequency_scores(bundle, prompt)` → `FrequencyScoreResult` — decomposes scores by RoPE frequency band, shape `[layers, heads, num_freq, 2]`.
4. Results are cached by `qwen_cache.py` to avoid re-running the model.
5. `dashboard/state.py` (`AppState`) holds the latest results reactively; views observe it and re-render on change.

### GQA handling

Qwen2.5-0.5B-Instruct uses Grouped Query Attention (2 KV heads, 14 Q heads). `qwen_core.py` repeats keys before scoring so every Q head gets an independent score. `QwenBundle.q_group_size = num_heads // num_kv_heads = 7`.

### `fix_query` flag

`collect_scores(..., fix_query=True)` (default) holds the last token's query vector fixed across permutations. This correctly isolates positional vs. symbolic sensitivity even when block swaps displace the last token.

---

## Scoring metrics — exact formulas

### Model configuration (Qwen2.5-0.5B-Instruct)

| Symbol | Name | Value |
|--------|------|-------|
| `L` | num_layers | 24 |
| `H` | num_heads (Q) | 14 |
| `H_kv` | num_kv_heads | 2 |
| `d` | hidden_size | 896 |
| `d_h` | head_dim = d/H | 64 |
| `F` | num_frequencies = d_h/2 | 32 |
| `θ` | rope_theta | 1 000 000 |
| `G` | q_group_size = H/H_kv | 7 |

---

### Step 1 — Block partition

A prompt of `T` tokens is split into `m` contiguous blocks (`n_blocks`, default 16). Block `b` covers token indices `[s_b, e_b)` with size `n_b = e_b - s_b`. Blocks are as equal as possible (`base, rem = divmod(T, m)`; the first `rem` blocks get one extra token).

All distinct pairwise swaps are enumerated:

```
S = { (i, j) : 0 ≤ i < j < m },   N_s = m(m-1)/2
```

For swap `k = (i_k, j_k)`: the full segment `[s_{i_k}, e_{j_k})` is rearranged by **block rotation**. Let `L_i = n_{i_k}`, `L_j = n_{j_k}`, `δ = L_j - L_i`. The permutation is:

- Slots `[s_i, s_i + L_j)` ← tokens of block `j_k`
- Slots `[s_i + L_j, s_j + δ)` ← middle tokens `[e_i, s_j)` shifted by `δ`  *(empty if blocks are adjacent)*
- Slots `[s_j + δ, e_j)` ← tokens of block `i_k`

This ensures **all** tokens participate in the swap. The total sequence length is preserved. When `L_i = L_j` this reduces to the standard two-block swap.

---

### Step 2 — Attention collection (`collect_scores`)

The base sequence is run once through the model to capture **per-layer input hidden states** `h_l ∈ ℝ^{T×d}` for each layer `l`.

For each swap `k` and each layer `l`, the Q/K projections are computed as follows.

**`fix_query=True` (default):**
- `Q_l` ← from **unperturbed** `h_l`
- `K_l` ← from **permuted** `h_l[:, perm_k, :]`

This keeps the last token's query fixed so the measurement is: *does the last token's query reach the same position or the same symbol after the swap?*

**`fix_query=False` (legacy):**
- Both `Q_l` and `K_l` from the permuted hidden state.

In both cases, scaled dot-product attention (with causal mask) is computed, and the **last-token attention row** is extracted:

```
a^{(l,h,k)}_t = softmax_t( (Q_l[last] · K_l[t]) / sqrt(d_h) + mask_t )   ∈ [0,1]^T
```

This produces a tensor `att_last_col` of shape `[L, 1, N_s+1, H, T]`  
(index 0 = baseline, indices 1…N_s = one per swap).

---

### Step 3 — Block-averaged attention

For the baseline (`k=0`):

```
avg_base(l, h, b)  =  (1/n_b) Σ_{t ∈ B_b} a^{(l,h,0)}_t
```

For swap `k = (i_k, j_k)`, the block boundaries are **dynamic**: after rotation the new intervals are `B'_{i_k} = [s_i, s_i+L_j)` (size `L_j`) and `B'_{j_k} = [s_j+δ, e_j)` (size `L_i`); middle blocks shift by `δ` but keep their sizes.

```
avg_perm(l, h, k, b)  =  (1/|B'_b|) Σ_{t ∈ B'_b} a^{(l,h,k)}_t
```

Concretely:
- `avg_perm(k, i_k)` = mean attention over `[s_i, s_i+L_j)` — the positions now holding block `j_k`'s tokens
- `avg_perm(k, j_k)` = mean attention over `[s_j+δ, e_j)` — the positions now holding block `i_k`'s tokens

---

### Step 4 — Positional and symbolic scores

For each swap `k = (i_k, j_k)`, form 2D vectors:

```
v_base^k  =  [ avg_base(i_k),  avg_base(j_k) ]   ∈ ℝ²
v_perm^k  =  [ avg_perm(k, j_k), avg_perm(k, i_k) ]   ∈ ℝ²
```

Intuition (layer/head indices dropped for clarity):
- A **positional head** (follows position, ignores token identity) produces `v_perm^k = v_base^k`, so `cos_sim(v_perm^k, v_base^k) = 1`.
- A **symbolic head** (follows token identity, ignores position) produces `v_perm^k = flip(v_base^k)`, so `cos_sim(v_perm^k, flip(v_base^k)) = 1`.

Raw per-swap scores:

```
pos_k  =  cos_sim( v_perm^k,        v_base^k )
sym_k  =  cos_sim( v_perm^k,  flip( v_base^k ) )
```

#### Saliency weighting

Swaps where the baseline attention is similar between both blocks carry little signal. A saliency weight upweights informative swaps:

```
δ_k  =  | avg_base(i_k) − avg_base(j_k) |
w_k  =  softmax_k( δ_k / τ )          τ = 0.1 (default)
```

`softmax` is taken over all `N_s` swaps for a given `(l, h)`.

#### Final aggregated scores

```
pos_score[l, h]  =  Σ_k  w_k · pos_k
sym_score[l, h]  =  Σ_k  w_k · sym_k
```

Output shapes: `pos_scores`, `sym_scores` ∈ `[L, H, 1]` (the singleton dim is the batch axis).

---

### Step 5 — Frequency-resolved scores (`collect_frequency_scores`)

RoPE encodes position using `F = d_h / 2 = 32` frequency pairs. Frequency `f ∈ {0, …, F-1}` uses:

```
θ_f  =  θ^{−f/F}   (inverse frequency)
λ_f  =  2π / θ_f   (wavelength, stored in FrequencyScoreOutputs.frequency_wavelengths)
```

For each swap `k` and layer `l`, after computing Q and K (same `fix_query` logic), the `head_dim = 2F` dimensions are split into `F` pairs:

```
q_f  =  [ Q[…, f],  Q[…, f+F] ]   ∈ ℝ²    for f = 0, …, F-1
k_f  =  [ K[…, f],  K[…, f+F] ]   ∈ ℝ²
```

Frequency-`f` attention logit for the last-token query attending to position `t`:

```
logit_f(t)  =  ( q_f[last] · k_f[t] ) / sqrt(d_h)
```

Note: `d_h`-normalisation is reused even though only 2 dimensions contribute — this is a deliberate choice to keep units comparable with the full-attention scores.

Frequency-`f` attention:

```
att_f(t)  =  softmax_t( logit_f(t) )   ∈ [0,1]^T
```

The same `get_scores` pipeline (Steps 3–4) is then applied to these per-frequency attentions by treating each `(f, h)` pair as an independent "head". The result is:

```
frequency_scores[l, h, f, :]  ∈ ℝ²   →  [pos_score_f,  sym_score_f]
```

Additionally, per-swap logit norms are stored:

```
norm_f  =  ‖ logit_f ‖_2   (over all positions t)
```

stored in `FrequencyScoreOutputs.frequency_logit_norms` of shape `[L, N_s+1, H, F]`.
