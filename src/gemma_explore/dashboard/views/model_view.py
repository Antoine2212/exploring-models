"""Tab 1 – model-level pos/sym heatmaps and scatter.

Both figures are computed once per prompt and served from the PNG cache.
"""
from __future__ import annotations

import panel as pn

from gemma_explore.dashboard.state import DashboardState, png_bytes_to_html
from gemma_explore.qwen_viz import plot_heads_scatter, plot_pos_sym_heatmaps


class ModelView:
    def __init__(self, state: DashboardState) -> None:
        self._state = state
        self._last_version: int = -1

        self._status = pn.pane.Markdown(
            "Run or select a prompt first.",
            sizing_mode="stretch_width",
            margin=(0, 0, 4, 0),
        )
        self._heatmap_pane = pn.pane.HTML(
            "", sizing_mode="stretch_width", min_height=300
        )
        self._scatter_pane = pn.pane.HTML(
            "", sizing_mode="stretch_width", min_height=300
        )
        self.panel = pn.Column(
            self._status,
            self._heatmap_pane,
            self._scatter_pane,
            sizing_mode="stretch_both",
        )

    def refresh(self) -> None:
        s = self._state
        if s.scores is None:
            self._status.object = "Run or select a prompt first."
            self._heatmap_pane.object = ""
            self._scatter_pane.object = ""
            self._last_version = -1
            return

        if s.data_version == self._last_version:
            return

        nl, nh = s.bundle.num_layers, s.bundle.num_heads

        hmap_png = s.get_model_heatmap_png(
            lambda: plot_pos_sym_heatmaps(s.scores, num_layers=nl, num_heads=nh)
        )
        scat_png = s.get_model_scatter_png(
            lambda: plot_heads_scatter(s.scores, num_layers=nl, num_heads=nh)
        )

        self._heatmap_pane.object = png_bytes_to_html(hmap_png)
        self._scatter_pane.object = png_bytes_to_html(scat_png)
        self._status.object = "Model-level scores."
        self._last_version = s.data_version
