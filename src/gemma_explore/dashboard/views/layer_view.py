"""Tab 2 – all attention-head heatmaps for the selected layer.

PNG cached per (prompt, layer).
"""
from __future__ import annotations

import panel as pn

from gemma_explore.dashboard.state import DashboardState, png_bytes_to_html
from gemma_explore.qwen_viz import plot_all_attention_heads


class LayerView:
    def __init__(self, state: DashboardState) -> None:
        self._state = state
        self._last_key: tuple = ()

        self._status = pn.pane.Markdown(
            "Run or select a prompt first.",
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

    def refresh(self) -> None:
        s = self._state
        if s.cache is None:
            self._status.object = "Run or select a prompt first."
            self._plot_pane.object = ""
            self._last_key = ()
            return

        layer = s.selected_layer
        key = (s.active_prompt_hash, layer)
        if key == self._last_key:
            return

        png = s.get_layer_png(
            layer,
            lambda: plot_all_attention_heads(
                s.cache,
                prompt_id=0,
                layer_idx=layer,
            ),
        )
        self._plot_pane.object = png_bytes_to_html(png)
        self._status.object = f"All attention heads — layer {layer}."
        self._last_key = key
