"""Persistent JSON-backed registry of previously run prompts."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REGISTRY_PATH = Path("data/cache/prompts_registry.json")


@dataclass
class PromptEntry:
    prompt_text: str
    apply_chat_template: bool
    cache_path: str
    prompt_hash: str
    created_at: str

    @property
    def display_label(self) -> str:
        tag = "[chat]" if self.apply_chat_template else "[raw]"
        preview = " ".join(self.prompt_text.split())[:60]
        if len(self.prompt_text) > 60:
            preview += "…"
        return f"{tag} {preview}"

    def cache_path_obj(self) -> Path:
        return Path(self.cache_path)


class PromptRegistry:
    """Ordered list of prompt entries, persisted to disk as JSON."""

    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        self._path = path
        self._entries: list[PromptEntry] = []
        self._by_hash: dict[str, PromptEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            for item in data:
                try:
                    entry = PromptEntry(**item)
                except TypeError:
                    continue
                if entry.cache_path_obj().exists():
                    self._entries.append(entry)
                    self._by_hash[entry.prompt_hash] = entry
        except Exception:
            pass  # corrupt registry – start fresh

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump([asdict(e) for e in self._entries], f, indent=2)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add(self, entry: PromptEntry) -> None:
        if entry.prompt_hash in self._by_hash:
            return
        self._entries.append(entry)
        self._by_hash[entry.prompt_hash] = entry
        self.save()

    def remove(self, prompt_hash: str) -> None:
        if prompt_hash not in self._by_hash:
            return
        del self._by_hash[prompt_hash]
        self._entries = [e for e in self._entries if e.prompt_hash != prompt_hash]
        self.save()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, prompt_hash: str) -> Optional[PromptEntry]:
        return self._by_hash.get(prompt_hash)

    def get_by_label(self, label: str) -> Optional[PromptEntry]:
        for e in self._entries:
            if e.display_label == label:
                return e
        return None

    @property
    def entries(self) -> list[PromptEntry]:
        return list(self._entries)

    def labels(self) -> list[str]:
        return [e.display_label for e in self._entries]


def make_prompt_entry(
    prompt_text: str,
    apply_chat_template: bool,
    cache_path: Path,
    prompt_hash: str,
) -> PromptEntry:
    return PromptEntry(
        prompt_text=prompt_text,
        apply_chat_template=apply_chat_template,
        cache_path=str(cache_path),
        prompt_hash=prompt_hash,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
