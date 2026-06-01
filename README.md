# exploring-models

Mechanistic interpretability toolkit for analyzing attention heads in Qwen2.5-0.5B-Instruct. The core idea: measure whether each attention head tracks **position** (where a token is) or **symbol** (what token it is) by swapping token blocks and observing how last-token attention shifts.

## What it does

- **Positional/symbolic scoring** — splits the prompt into blocks, runs all pairwise block swaps, and computes two scores per head per layer:
  - *positional*: the head follows the position of a token
  - *symbolic*: the head follows the identity of a token
- **Frequency-resolved analysis** — decomposes scores by RoPE frequency band to see which rotation frequencies drive each head's behavior
- **Interactive dashboard** — Panel/Bokeh app with views for head matrices, layer plots, frequency spectra, hidden states, and prompt inspection

## Setup

```bash
uv sync
```

## Usage

### Dashboard

```bash
uv run gemma-dashboard
```

Opens a local web UI. Enter a prompt, run analysis, and explore results across tabs:

| Tab | What you see |
|-----|-------------|
| Head matrix | Per-head pos/sym scores across all layers |
| Layer view | Score evolution by layer |
| Frequency view | RoPE frequency breakdown per head |
| Hidden states | Residual stream visualizations |
| Prompt overview | Token blocks and chat template |

### Python API

```python
from gemma_explore.qwen_core import load_bundle, collect_scores, collect_frequency_scores

bundle = load_bundle()  # loads Qwen/Qwen2.5-0.5B-Instruct

# Positional/symbolic scores
out = collect_scores(bundle, "The cat sat on the mat.", n_blocks=8)
print(out.pos_scores.shape)  # [layers, heads, 1]
print(out.sym_scores.shape)

# Frequency-resolved scores
freq_out = collect_frequency_scores(bundle, "The cat sat on the mat.")
print(freq_out.frequency_scores.shape)  # [layers, heads, num_freq, 2]
```

`fix_query=True` (default) holds the last token's query fixed while permuting keys — this correctly isolates positional vs. symbolic behavior even when the last token is displaced by a swap.

## Project structure

```
src/gemma_explore/
  qwen_core.py          # model loading, block swaps, score computation
  qwen_cache.py         # disk caching for score results
  qwen_viz.py           # standalone plotting utilities
  dashboard/
    app.py              # Panel app wiring
    state.py            # shared app state
    views/              # one file per dashboard tab
scripts/
  ruler_eval_qwen.py    # RULER benchmark evaluation
notebooks/
  explore_heads.ipynb   # exploratory analysis
tests/
  test_model.py
  inspect_model.py
```

## Key concepts

**Block swap protocol** — the prompt is split into `n_blocks` equal-length token blocks. For each pair of blocks `(i, j)`, a permutation swaps their positions. The shift in the last token's attention distribution reveals whether each head is sensitive to position or symbol identity.

**GQA support** — Qwen2.5 uses Grouped Query Attention (8 KV heads, 14 Q heads). Keys are repeated before scoring so every Q head gets its own score.

**Caching** — computed scores are cached to `data/cache/` keyed by model ID and prompt hash to avoid re-running the model.
