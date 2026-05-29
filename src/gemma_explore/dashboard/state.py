from dataclasses import dataclass

from gemma_explore.qwen_core import QwenBundle
from gemma_explore.qwen_cache import (
    CacheHandle,
    build_activation_cache,
    ensure_frequency_scores,
    ensure_scores,
)


@dataclass
class DashboardState:
    bundle: QwenBundle
    cache: CacheHandle | None = None
    active_prompt_id: int = 0
    scores: dict | None = None
    freq_scores: dict | None = None


def run_prompt(
    state: DashboardState,
    prompt_text: str,
    n_blocks: int = 16,
    tau: float = 0.1,
) -> None:
    cache = build_activation_cache(state.bundle, [prompt_text])
    state.cache = cache
    state.scores = ensure_scores(cache, n_blocks=n_blocks, tau=tau)
    state.freq_scores = ensure_frequency_scores(cache, n_blocks=n_blocks, tau=tau)
    state.active_prompt_id = 0


def get_model_dims(state: DashboardState) -> tuple[int, int]:
    num_layers = len(state.bundle.model.model.layers)
    num_heads = state.bundle.model.config.num_attention_heads
    return num_layers, num_heads