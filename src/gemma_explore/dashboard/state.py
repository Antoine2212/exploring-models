"""Central dashboard state with PNG cache and prompt management."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
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


@dataclass(frozen=True)
class PromptHistoryEntry:
    prompt_text: str
    cache_path: Path
    prompt_hash: str


@dataclass
class DashboardState:
    bundle: QwenBundle
    cache: CacheHandle | None = None
    active_prompt_id: int = 0
    scores: dict | None = None
    freq_scores: dict | None = None
    last_prompt_text: str = ""
    data_version: int = 0
    score_version: int = 0
    freq_version: int = 0
    selected_layer: int = 0
    selected_head: int = 0
    prompt_history: list[PromptHistoryEntry] = field(default_factory=list)
    active_prompt_hash: str = ""
    _prompt_history_by_hash: dict[str, PromptHistoryEntry] = field(default_factory=dict, init=False, repr=False)
    _prompt_history_by_label: dict[str, PromptHistoryEntry] = field(default_factory=dict, init=False, repr=False)
    _head_png_cache: dict[tuple[str, int, int], bytes] = field(default_factory=dict, init=False, repr=False)
    _frequency_png_cache: dict[tuple[str, int, int], bytes] = field(default_factory=dict, init=False, repr=False)
    _score_png_cache: dict[tuple[str, int, int, int, int], bytes] = field(default_factory=dict, init=False, repr=False)
    _layer_png_cache: dict[tuple, bytes] = field(default_factory=dict, init=False, repr=False)
    _model_heatmap_png_cache: dict[tuple, bytes] = field(default_factory=dict, init=False, repr=False)
    _model_scatter_png_cache: dict[tuple, bytes] = field(default_factory=dict, init=False, repr=False)
    _hidden_png_cache: dict[tuple, bytes] = field(default_factory=dict, init=False, repr=False)
    _freq_png_cache: dict[tuple, bytes] = field(default_factory=dict, init=False, repr=False)

    selected_layer: int = 0
    selected_head: int = 0

    _model_png: dict = field(default_factory=dict, init=False, repr=False)
    _layer_png: dict = field(default_factory=dict, init=False, repr=False)
    _freq_png: dict = field(default_factory=dict, init=False, repr=False)
    _hidden_png: dict = field(default_factory=dict, init=False, repr=False)

    def get_model_heatmap_png(self, builder: Callable[[], Figure]) -> bytes:
        key = self.active_prompt_hash + ":hmap"
        if key not in self._model_png:
            self._model_png[key] = fig_to_png_bytes(builder())
        return self._model_png[key]

    def get_model_scatter_png(self, builder: Callable[[], Figure]) -> bytes:
        key = self.active_prompt_hash + ":scat"
        if key not in self._model_png:
            self._model_png[key] = fig_to_png_bytes(builder())
        return self._model_png[key]

    def get_layer_png(self, layer: int, builder: Callable[[], Figure]) -> bytes:
        key = (self.active_prompt_hash, layer)
        if key not in self._layer_png:
            self._layer_png[key] = fig_to_png_bytes(builder())
        return self._layer_png[key]

    def get_freq_png(self, layer: int, head: int, builder: Callable[[], Figure]) -> bytes:
        key = (self.active_prompt_hash, layer, head)
        if key not in self._freq_png:
            self._freq_png[key] = fig_to_png_bytes(builder())
        return self._freq_png[key]

    def get_prompt_entry_by_hash(self, prompt_hash: str) -> PromptHistoryEntry | None:
        """Lookup a prompt-history entry by prompt hash."""
        return self._prompt_history_by_hash.get(prompt_hash)

    def get_prompt_entry_by_label(self, label: str) -> PromptHistoryEntry | None:
        """Lookup a prompt-history entry by display label."""
        self._rebuild_prompt_label_index()
        return self._prompt_history_by_label.get(label)

    def get_cached_head_png(
        self,
        layer_idx: int,
        head_idx: int,
        builder: Callable[[], Figure],
    ) -> bytes:
        """Return cached PNG bytes for a head plot."""
        return self._get_or_build_png(
            self._head_png_cache,
            (self._require_active_prompt_hash(), layer_idx, head_idx),
            builder,
        )

    def get_cached_frequency_png(
        self,
        layer_idx: int,
        head_idx: int,
        builder: Callable[[], Figure],
    ) -> bytes:
        """Return cached PNG bytes for a frequency plot."""
        return self._get_or_build_png(
            self._frequency_png_cache,
            (self._require_active_prompt_hash(), layer_idx, head_idx),
            builder,
        )

    def get_cached_score_png(
        self,
        builder: Callable[[], Figure],
    ) -> bytes:
        """Return cached PNG bytes for a score plot keyed by prompt and version state."""
        key = (
            self._require_active_prompt_hash(),
            self.data_version,
            self.score_version,
            self.freq_version,
            self.active_prompt_id,
        )
        return self._get_or_build_png(self._score_png_cache, key, builder)

    def get_layer_png(self, layer: int, builder: Callable[[], Figure]) -> bytes:
        key = (self._require_active_prompt_hash(), layer)
        return self._get_or_build_png(self._layer_png_cache, key, builder)

    def get_model_heatmap_png(self, builder: Callable[[], Figure]) -> bytes:
        key = (self._require_active_prompt_hash(), self.data_version)
        return self._get_or_build_png(self._model_heatmap_png_cache, key, builder)

    def get_model_scatter_png(self, builder: Callable[[], Figure]) -> bytes:
        key = (self._require_active_prompt_hash(), self.data_version)
        return self._get_or_build_png(self._model_scatter_png_cache, key, builder)

    def get_hidden_png(self, layer: int, head: int, builder: Callable[[], Figure]) -> bytes:
        key = (self._require_active_prompt_hash(), layer, head)
        return self._get_or_build_png(self._hidden_png_cache, key, builder)

    def get_freq_png(self, layer: int, head: int, builder: Callable[[], Figure]) -> bytes:
        key = (self._require_active_prompt_hash(), layer, head)
        return self._get_or_build_png(self._freq_png_cache, key, builder)

    def clear_visualization_caches(self) -> None:
        """Clear all cached visualization PNG bytes."""
        self._head_png_cache.clear()
        self._frequency_png_cache.clear()
        self._score_png_cache.clear()
        self._layer_png_cache.clear()
        self._model_heatmap_png_cache.clear()
        self._model_scatter_png_cache.clear()
        self._hidden_png_cache.clear()
        self._freq_png_cache.clear()

    def evict_prompt_visualization_caches(self, prompt_hash: str) -> None:
        """Remove cached visualization PNGs for a specific prompt."""
        self._evict_prefixed_keys(self._head_png_cache, prompt_hash)
        self._evict_prefixed_keys(self._frequency_png_cache, prompt_hash)
        self._evict_prefixed_keys(self._score_png_cache, prompt_hash)

    def _get_or_build_png(
        self,
        cache: dict[tuple, bytes],
        key: tuple,
        builder: Callable[[], Figure],
    ) -> bytes:
        cached = cache.get(key)
        if cached is not None:
            return cached

        figure = builder()
        png_bytes = self._figure_to_png_bytes(figure)
        cache[key] = png_bytes
        return png_bytes

    @staticmethod
    def _figure_to_png_bytes(figure: Figure) -> bytes:
        buffer = BytesIO()
        try:
            figure.savefig(buffer, format="png", bbox_inches="tight")
            buffer.seek(0)
            return buffer.getvalue()
        finally:
            buffer.close()
            figure.clf()
            figure.canvas.manager = None  # type: ignore[attr-defined]

            import matplotlib.pyplot as plt

            plt.close(figure)

    def _require_active_prompt_hash(self) -> str:
        if not self.active_prompt_hash:
            raise ValueError("No active prompt is loaded.")
        return self.active_prompt_hash

    def _rebuild_prompt_label_index(self) -> None:
        self._prompt_history_by_label = {
            self.get_prompt_history_label(entry): entry for entry in self.prompt_history
        }

    @staticmethod
    def _prompt_preview(prompt_text: str, max_len: int = 72) -> str:
        compact = " ".join(prompt_text.strip().split())
        if len(compact) <= max_len:
            return compact
        return f"{compact[: max_len - 1].rstrip()}…"

    @staticmethod
    def _evict_prefixed_keys(cache: dict[tuple, bytes], prompt_hash: str) -> None:
        keys_to_remove = [key for key in cache if key and key == prompt_hash]
        for key in keys_to_remove:
            cache.pop(key, None)


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
