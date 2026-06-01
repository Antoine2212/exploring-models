"""Main dashboard application."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import panel as pn

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gemma_explore.dashboard.head_matrix import HeadMatrix
from gemma_explore.dashboard.prompt_registry import PromptRegistry
from gemma_explore.dashboard.state import DashboardState, load_entry, run_prompt
from gemma_explore.dashboard.views.freq_view import FreqView
from gemma_explore.dashboard.views.hidden_view import HiddenView
from gemma_explore.dashboard.views.layer_view import LayerView
from gemma_explore.dashboard.views.model_view import ModelView
from gemma_explore.qwen_core import QwenBundle, load_bundle


# ------------------------------------------------------------------
# Prompt manager sidebar section
# ------------------------------------------------------------------


class PromptManager:
    """
    Sidebar widget set for prompt management.

    Modes:
      - "select" : pick from history dropdown  → Load button
      - "new"    : text area + chat-template toggle → Run button
    """

    def __init__(self, state: DashboardState, on_load: callable, on_new: callable) -> None:
        self._state = state
        self._on_load = on_load
        self._on_new = on_new

        # ---- History selector ----
        self._history_select = pn.widgets.Select(
            name="History",
            options=self._build_options(),
            sizing_mode="stretch_width",
        )
        self._load_btn = pn.widgets.Button(
            name="Load", button_type="primary", sizing_mode="stretch_width"
        )
        self._delete_btn = pn.widgets.Button(
            name="Delete from registry", button_type="danger", sizing_mode="stretch_width"
        )

        # ---- New prompt ----
        self._prompt_input = pn.widgets.TextAreaInput(
            name="New prompt",
            placeholder="Enter a prompt…",
            rows=4,
            sizing_mode="stretch_width",
        )
        self._chat_toggle = pn.widgets.Checkbox(
            name="Apply chat template",
            value=True,
        )
        self._run_btn = pn.widgets.Button(
            name="Run", button_type="success", sizing_mode="stretch_width"
        )

        # ---- Status ----
        self._status = pn.pane.Markdown(
            "", sizing_mode="stretch_width", margin=(4, 0, 0, 0)
        )

        # ---- Accordion layout ----
        history_col = pn.Column(
            self._history_select,
            pn.Row(self._load_btn, self._delete_btn, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )
        new_col = pn.Column(
            self._prompt_input,
            self._chat_toggle,
            self._run_btn,
            sizing_mode="stretch_width",
        )
        self._accordion = pn.Accordion(
            ("History", history_col),
            ("New prompt", new_col),
            active=[0] if self._state.registry.entries else [1],
            sizing_mode="stretch_width",
        )

        # ---- Wire up ----
        self._load_btn.on_click(self._do_load)
        self._delete_btn.on_click(self._do_delete)
        self._run_btn.on_click(self._do_run)

        self.panel = pn.Column(
            self._accordion,
            self._status,
            sizing_mode="stretch_width",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_options(self) -> list[str]:
        labels = self._state.registry.labels()
        return labels if labels else ["(no prompts yet)"]

    def _set_status(self, msg: str) -> None:
        self._status.object = msg

    def _sync_history(self) -> None:
        labels = self._build_options()
        self._history_select.options = labels
        active_label = next(
            (e.display_label for e in self._state.registry.entries
             if e.prompt_hash == self._state.active_prompt_hash),
            None,
        )
        if active_label and active_label in labels:
            self._history_select.value = active_label

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _do_load(self, _event) -> None:
        label = self._history_select.value
        if not label or label == "(no prompts yet)":
            self._set_status("No prompt selected.")
            return
        entry = self._state.registry.get_by_label(label)
        if entry is None:
            self._set_status(f"Entry not found: {label!r}")
            return
        self._load_btn.disabled = True
        self._set_status("Loading…")
        try:
            self._on_load(entry)
            self._set_status("Loaded from cache.")
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")
        finally:
            self._load_btn.disabled = False

    def _do_delete(self, _event) -> None:
        label = self._history_select.value
        if not label or label == "(no prompts yet)":
            return
        entry = self._state.registry.get_by_label(label)
        if entry is None:
            return
        self._state.registry.remove(entry.prompt_hash)
        self._sync_history()
        self._set_status(f"Deleted from registry: {label[:40]}")

    def _do_run(self, _event) -> None:
        text = self._prompt_input.value.strip()
        if not text:
            self._set_status("Prompt is empty.")
            return
        self._run_btn.disabled = True
        self._run_btn.name = "Running…"
        self._set_status("Running model…")
        try:
            self._on_new(text, self._chat_toggle.value)
            self._sync_history()
            self._set_status("Done.")
        except Exception as exc:
            self._set_status(f"Run failed: {exc}")
        finally:
            self._run_btn.name = "Run"
            self._run_btn.disabled = False


# ------------------------------------------------------------------
# App builder
# ------------------------------------------------------------------


def build_app(bundle: QwenBundle) -> pn.template.base.BasicTemplate:
    registry = PromptRegistry()
    state = DashboardState(bundle=bundle, registry=registry)

    # ---- Views ----
    model_view = ModelView(state)
    layer_view = LayerView(state)
    freq_view = FreqView(state)
    hidden_view = HiddenView(state)

    # ---- Head matrix ----
    def _on_head_select(layer: int, head: int) -> None:
        state.selected_layer = layer
        state.selected_head = head
        layer_view.refresh()
        freq_view.refresh()
        hidden_view.refresh()  # no-op if not active
        _selected_label.object = f"**Selected:** layer {layer} · head {head}"

    head_matrix = HeadMatrix(
        num_layers=bundle.num_layers,
        num_heads=bundle.num_heads,
        on_select=_on_head_select,
    )

    _selected_label = pn.pane.Markdown(
        "**Selected:** layer 0 · head 0",
        sizing_mode="stretch_width",
        margin=(4, 0, 0, 0),
    )

    # ---- Tabs ----
    tabs = pn.Tabs(
        ("Model", model_view.panel),
        ("Layer", layer_view.panel),
        ("Frequency", freq_view.panel),
        ("Hidden states", hidden_view.panel),
        dynamic=False,
        sizing_mode="stretch_both",
        tabs_location="above",
    )

    def _on_tab_change(event) -> None:
        idx = event.new
        if idx == 3:
            hidden_view.activate()
        else:
            hidden_view.deactivate()

    tabs.param.watch(_on_tab_change, "active")

    # ---- Prompt management ----
    def _do_load(entry) -> None:
        load_entry(state, entry)
        _after_prompt_change()

    def _do_new(text: str, apply_chat_template: bool) -> None:
        run_prompt(state, text, apply_chat_template=apply_chat_template)
        _after_prompt_change()

    def _after_prompt_change() -> None:
        # Update head matrix scores
        if state.scores is not None:
            import torch
            pos = state.scores.get("pos_scores")
            sym = state.scores.get("sym_scores")
            if pos is not None and sym is not None:
                import numpy as np
                p = np.array(pos).reshape(bundle.num_layers, bundle.num_heads)
                s_arr = np.array(sym).reshape(bundle.num_layers, bundle.num_heads)
                head_matrix.update_scores(p.flatten().tolist(), s_arr.flatten().tolist())
        # Refresh all views
        model_view.refresh()
        layer_view.refresh()
        freq_view.refresh()
        if hidden_view._active:
            hidden_view.refresh()

    prompt_manager = PromptManager(state, on_load=_do_load, on_new=_do_new)

    # ---- Sidebar ----
    sidebar = pn.Column(
        pn.pane.Markdown("## Qwen Explore", margin=(0, 0, 8, 0)),
        prompt_manager.panel,
        pn.layout.Divider(),
        head_matrix.panel,
        _selected_label,
        width=340,
        min_width=300,
        max_width=380,
        sizing_mode="fixed",
    )

    # Load first registry entry automatically if available
    def _autoload() -> None:
        entries = state.registry.entries
        if entries:
            try:
                load_entry(state, entries[-1])
                _after_prompt_change()
                prompt_manager._sync_history()
                prompt_manager._set_status(f"Auto-loaded: {entries[-1].display_label[:40]}")
            except Exception:
                pass

    pn.state.onload(_autoload)

    template = pn.template.FastListTemplate(
        title="Qwen Explore",
        sidebar=[sidebar],
        main=[
            pn.Column(
                tabs,
                sizing_mode="stretch_both",
                min_height=700,
            )
        ],
        main_layout=None,
        accent_base_color="#1f77b4",
    )
    return template


def launch_dashboard(
    bundle: Optional[QwenBundle] = None,
    port: int = 5006,
    show: bool = True,
) -> None:
    pn.extension("bokeh", "matplotlib")
    if bundle is None:
        bundle = load_bundle()
    app = build_app(bundle)
    pn.serve(app.servable(), port=port, show=show)


if __name__ == "__main__":
    launch_dashboard()
