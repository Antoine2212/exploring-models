"""Central dashboard state with PNG cache and prompt management."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable

from matplotlib.figure import Figure

from gemma_explore.dashboard.prompt_registry import (
    PromptEntry,
    PromptRegistry,
    make_prompt_entry,
)
from gemma_explore.qwen_cache import (
    CacheHandle,
    build_activation_cache,
    ensure_frequency_scores,
    ensure_scores,
    load_cache,
    save_cache,
)
from gemma_explore.qwen_core import QwenBundle, prompt_text_hash

CACHE_DIR = Path("data/cache")


def png_bytes_to_html(png_bytes: bytes) -> str:
    b64 = base64.b64encode(png_bytes).decode()
    return f'<img src="data:image/png;base64,{b64}" style="width:100%; max-width:1400px;" />'


def _fig_to_png(figure: Figure) -> bytes:
    import matplotlib.pyplot as plt
    buf = BytesIO()
    try:
        figure.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return buf.getvalue()
    finally:
        buf.close()
        plt.close(figure)


@dataclass
class DashboardState:
    bundle: QwenBundle
    cache: CacheHandle | None = None
    scores: dict | None = None
    freq_scores: dict | None = None
    active_prompt_hash: str = ""
    active_apply_chat_template: bool = True
    data_version: int = 0
    selected_layer: int = 0
    selected_head: int = 0
    registry: PromptRegistry = field(default_factory=PromptRegistry)
    _model_png: dict = field(default_factory=dict, init=False, repr=False)
    _layer_png: dict = field(default_factory=dict, init=False, repr=False)
    _freq_png: dict = field(default_factory=dict, init=False, repr=False)
    _hidden_png: dict = field(default_factory=dict, init=False, repr=False)

    def get_model_heatmap_png(self, builder: Callable[[], Figure]) -> bytes:
        key = self.active_prompt_hash + ":hmap"
        if key not in self._model_png:
            self._model_png[key] = _fig_to_png(builder())
        return self._model_png[key]

    def get_model_scatter_png(self, builder: Callable[[], Figure]) -> bytes:
        key = self.active_prompt_hash + ":scat"
        if key not in self._model_png:
            self._model_png[key] = _fig_to_png(builder())
        return self._model_png[key]

    def get_layer_png(self, layer: int, builder: Callable[[], Figure]) -> bytes:
        key = (self.active_prompt_hash, layer)
        if key not in self._layer_png:
            self._layer_png[key] = _fig_to_png(builder())
        return self._layer_png[key]

    def get_freq_png(self, layer: int, head: int, builder: Callable[[], Figure]) -> bytes:
        key = (self.active_prompt_hash, layer, head)
        if key not in self._freq_png:
            self._freq_png[key] = _fig_to_png(builder())
        return self._freq_png[key]

    def get_hidden_png(self, layer: int, head: int, builder: Callable[[], Figure]) -> bytes:
        key = (self.active_prompt_hash, layer, head)
        if key not in self._hidden_png:
            self._hidden_png[key] = _fig_to_png(builder())
        return self._hidden_png[key]


def _cache_path(bundle: QwenBundle, prompt_text: str, apply_chat_template: bool) -> Path:
    model_slug = bundle.model_id.replace("/", "__")
    salt = "|chat" if apply_chat_template else "|raw"
    h = prompt_text_hash(prompt_text + salt)
    return CACHE_DIR / f"{model_slug}__{h}.pt"


def run_prompt(
    state: DashboardState,
    prompt_text: str,
    apply_chat_template: bool = True,
    n_blocks: int = 16,
    tau: float = 0.1,
) -> bool:
    prompt_text = prompt_text.strip()
    if not prompt_text:
        raise ValueError("Prompt text must not be empty.")

    salt = "|chat" if apply_chat_template else "|raw"
    prompt_hash = prompt_text_hash(prompt_text + salt)

    if prompt_hash == state.active_prompt_hash and state.cache is not None:
        return False

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(state.bundle, prompt_text, apply_chat_template)

    if cache_path.exists():
        cache = load_cache(cache_path)
    else:
        cache = build_activation_cache(
            state.bundle,
            [prompt_text],
            apply_chat_template=apply_chat_template,
        )
        save_cache(cache, cache_path)

    scores = ensure_scores(
        cache,
        bundle=state.bundle,
        prompt_id=0,
        n_blocks=n_blocks,
        tau=tau,
        apply_chat_template=apply_chat_template,
    )
    freq_scores = ensure_frequency_scores(
        cache,
        bundle=state.bundle,
        prompt_id=0,
        n_blocks=n_blocks,
        tau=tau,
        apply_chat_template=apply_chat_template,
    )
    save_cache(cache, cache_path)

    entry = make_prompt_entry(
        prompt_text=prompt_text,
        apply_chat_template=apply_chat_template,
        cache_path=cache_path,
        prompt_hash=prompt_hash,
    )
    state.registry.add(entry)
    _install_state(state, cache, scores, freq_scores, prompt_hash, apply_chat_template)
    return True


def load_entry(
    state: DashboardState,
    entry: PromptEntry,
    n_blocks: int = 16,
    tau: float = 0.1,
) -> bool:
    if entry.prompt_hash == state.active_prompt_hash and state.cache is not None:
        return False

    cache = load_cache(entry.cache_path_obj())
    scores = ensure_scores(
        cache,
        bundle=state.bundle,
        prompt_id=0,
        n_blocks=n_blocks,
        tau=tau,
        apply_chat_template=entry.apply_chat_template,
    )
    freq_scores = ensure_frequency_scores(
        cache,
        bundle=state.bundle,
        prompt_id=0,
        n_blocks=n_blocks,
        tau=tau,
        apply_chat_template=entry.apply_chat_template,
    )
    save_cache(cache, entry.cache_path_obj())
    _install_state(state, cache, scores, freq_scores, entry.prompt_hash, entry.apply_chat_template)
    return True


def restore_prompt(
    state: DashboardState,
    prompt_text: str,
    n_blocks: int = 16,
    tau: float = 0.1,
) -> bool:
    """Restore a previously run prompt from the registry by text."""
    for apply_chat in (True, False):
        salt = "|chat" if apply_chat else "|raw"
        h = prompt_text_hash(prompt_text.strip() + salt)
        entry = state.registry.get(h)
        if entry is not None and entry.cache_path_obj().exists():
            return load_entry(state, entry, n_blocks=n_blocks, tau=tau)
    return False


def _install_state(
    state: DashboardState,
    cache: CacheHandle,
    scores: dict,
    freq_scores: dict,
    prompt_hash: str,
    apply_chat_template: bool,
) -> None:
    state.cache = cache
    state.scores = scores
    state.freq_scores = freq_scores
    state.active_prompt_hash = prompt_hash
    state.active_apply_chat_template = apply_chat_template
    state.data_version += 1
