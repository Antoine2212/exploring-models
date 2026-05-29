from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable

from matplotlib.figure import Figure

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


@dataclass(frozen=True)
class PromptHistoryEntry:
    prompt_text: str
    cache_path: Path
    prompt_hash: str


@dataclass
class DashboardState:
    """Mutable dashboard state for one running app instance."""

    bundle: QwenBundle
    cache: CacheHandle | None = None
    active_prompt_id: int = 0
    scores: dict | None = None
    freq_scores: dict | None = None
    last_prompt_text: str = ""
    data_version: int = 0
    score_version: int = 0
    freq_version: int = 0
    prompt_history: list[PromptHistoryEntry] = field(default_factory=list)
    active_prompt_hash: str = ""
    _prompt_history_by_hash: dict[str, PromptHistoryEntry] = field(default_factory=dict, init=False, repr=False)
    _prompt_history_by_label: dict[str, PromptHistoryEntry] = field(default_factory=dict, init=False, repr=False)
    _head_png_cache: dict[tuple[str, int, int], bytes] = field(default_factory=dict, init=False, repr=False)
    _frequency_png_cache: dict[tuple[str, int, int], bytes] = field(default_factory=dict, init=False, repr=False)
    _score_png_cache: dict[tuple[str, int, int, int, int], bytes] = field(default_factory=dict, init=False, repr=False)

    def set_active_prompt(
        self,
        *,
        prompt_text: str,
        prompt_hash: str,
        cache: CacheHandle,
        scores: dict | None,
        freq_scores: dict | None,
    ) -> None:
        """Install prompt-backed state and bump versions for all dependent views."""
        self.cache = cache
        self.scores = scores
        self.freq_scores = freq_scores
        self.active_prompt_id = 0
        self.last_prompt_text = prompt_text
        self.active_prompt_hash = prompt_hash
        self.data_version += 1
        self.score_version += 1
        self.freq_version += 1

    def add_prompt_history_entry(
        self,
        *,
        prompt_text: str,
        cache_path: Path,
        prompt_hash: str,
    ) -> PromptHistoryEntry:
        """Add a prompt to in-session history if absent, preserving insertion order."""
        existing = self._prompt_history_by_hash.get(prompt_hash)
        if existing is not None:
            return existing

        entry = PromptHistoryEntry(
            prompt_text=prompt_text,
            cache_path=cache_path,
            prompt_hash=prompt_hash,
        )
        self.prompt_history.append(entry)
        self._prompt_history_by_hash[prompt_hash] = entry
        self._rebuild_prompt_label_index()
        return entry

    def restore_prompt_from_history(
        self,
        entry: PromptHistoryEntry,
        *,
        n_blocks: int = 16,
        tau: float = 0.1,
    ) -> None:
        """Restore a previously computed prompt from disk without recomputation."""
        cache = load_cache(entry.cache_path)
        scores = ensure_scores(
            cache,
            bundle=self.bundle,
            prompt_id=0,
            n_blocks=n_blocks,
            tau=tau,
        )
        freq_scores = ensure_frequency_scores(
            cache,
            bundle=self.bundle,
            prompt_id=0,
            n_blocks=n_blocks,
            tau=tau,
        )
        self.set_active_prompt(
            prompt_text=entry.prompt_text,
            prompt_hash=entry.prompt_hash,
            cache=cache,
            scores=scores,
            freq_scores=freq_scores,
        )

    def get_prompt_history_labels(self) -> list[str]:
        """Return readable labels for prompt-history widgets in insertion order."""
        self._rebuild_prompt_label_index()
        return list(self._prompt_history_by_label.keys())

    def get_prompt_history_label(self, entry: PromptHistoryEntry) -> str:
        """Return the display label for a prompt-history entry."""
        index = self.prompt_history.index(entry) + 1
        preview = self._prompt_preview(entry.prompt_text)
        return f"{index:02d}. {preview} [{entry.prompt_hash[:8]}]"

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

    def clear_visualization_caches(self) -> None:
        """Clear all cached visualization PNG bytes."""
        self._head_png_cache.clear()
        self._frequency_png_cache.clear()
        self._score_png_cache.clear()

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


def _cache_path(bundle: QwenBundle, prompt_text: str) -> Path:
    """Return the cache file path for a prompt."""
    model_slug = bundle.model_id.replace("/", "__")
    prompt_hash = prompt_text_hash(prompt_text)
    return CACHE_DIR / f"{model_slug}__{prompt_hash}.pt"


def run_prompt(
    state: DashboardState,
    prompt_text: str,
    n_blocks: int = 16,
    tau: float = 0.1,
) -> bool:
    """Run a prompt and update state, skipping identical reruns."""
    prompt_text = prompt_text.strip()
    if not prompt_text:
        raise ValueError("Prompt text must not be empty.")

    prompt_hash = prompt_text_hash(prompt_text)
    if prompt_hash == state.active_prompt_hash and state.cache is not None:
        return False

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(state.bundle, prompt_text)

    if cache_path.exists():
        cache = load_cache(cache_path)
    else:
        cache = build_activation_cache(state.bundle, [prompt_text])
        save_cache(cache, cache_path)

    scores = ensure_scores(
        cache,
        bundle=state.bundle,
        prompt_id=0,
        n_blocks=n_blocks,
        tau=tau,
    )
    freq_scores = ensure_frequency_scores(
        cache,
        bundle=state.bundle,
        prompt_id=0,
        n_blocks=n_blocks,
        tau=tau,
    )
    save_cache(cache, cache_path)

    state.add_prompt_history_entry(
        prompt_text=prompt_text,
        cache_path=cache_path,
        prompt_hash=prompt_hash,
    )
    state.set_active_prompt(
        prompt_text=prompt_text,
        prompt_hash=prompt_hash,
        cache=cache,
        scores=scores,
        freq_scores=freq_scores,
    )
    return True


def restore_prompt(
    state: DashboardState,
    entry_or_label: PromptHistoryEntry | str,
    n_blocks: int = 16,
    tau: float = 0.1,
) -> bool:
    """Restore a prompt from in-session history without recomputation."""
    entry = (
        entry_or_label
        if isinstance(entry_or_label, PromptHistoryEntry)
        else state.get_prompt_entry_by_label(entry_or_label)
    )
    if entry is None:
        return False

    if entry.prompt_hash == state.active_prompt_hash and state.cache is not None:
        return False

    state.restore_prompt_from_history(entry, n_blocks=n_blocks, tau=tau)
    return True


def get_model_dims(state: DashboardState) -> tuple[int, int]:
    """Return model layer and head counts."""
    num_layers = len(state.bundle.model.model.layers)
    num_heads = state.bundle.model.config.num_attention_heads
    return num_layers, num_heads