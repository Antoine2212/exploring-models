"""Tab 4 – deep hidden-state analysis for the selected head.

This is the most expensive figure (9 sub-plots). It is only rendered
when the tab is explicitly activated, and then PNG-cached per (prompt, layer, head).
"""
from __future__ import annotations

import panel as pn

from gemma_explore.dashboard.state import DashboardState, png_bytes_to_html
from gemma_explore.qwen_viz import plot_head_block_dynamics


class HiddenView:
    def __init__(self, state: DashboardState) -> None:
        self._state = state
        self._last_key: tuple = ()
        self._active: bool = False  # only compute when this tab is visible

        self._status = pn.pane.Markdown(
            "Activate this tab to compute the hidden-state analysis.",
            sizing_mode="stretch_width",
            margin=(0, 0, 4, 0),
        )
        self._plot_pane = pn.pane.HTML(
            "", sizing_mode="stretch_width", min_height=400
        )
        self.panel = pn.Column(
            self._status,
            self._plot_pane,
            sizing_mode="stretch_both",
        )

    def activate(self) -> None:
        """Called when the tab becomes visible."""
        self._active = True
        self.refresh()

    def deactivate(self) -> None:
        self._active = False

    def refresh(self) -> None:
        if not self._active:
            return

        s = self._state
        if s.cache is None:
            self._status.object = "Run or select a prompt first."
            self._plot_pane.object = ""
            self._last_key = ()
            return

        layer = s.selected_layer
        head = s.selected_head
        key = (s.active_prompt_hash, layer, head)
        if key == self._last_key:
            return

        self._status.object = f"Rendering hidden-state analysis — layer {layer}, head {head}…"

        png = s.get_hidden_png(
            layer,
            head,
            lambda: plot_head_block_dynamics(
                s.cache,
                prompt_id=0,
                layer_idx=layer,
                head_idx=head,
            ),
        )
        self._plot_pane.object = png_bytes_to_html(png)
        self._status.object = (
            f"Hidden-state analysis — layer {layer}, head {head}."
            " (cached; switching heads is now instant)"
        )
        self._last_key = key
