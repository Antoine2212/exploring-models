from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import panel as pn

from gemma_explore.dashboard.state import DashboardState, get_model_dims
from gemma_explore.qwen_viz import plot_heads_scatter, plot_pos_sym_heatmaps




@dataclass
class ScoreView:
    """Persistent score-summary view."""

    state: DashboardState

    def __post_init__(self) -> None:
        self._last_version: int | None = None
        self.status = pn.pane.Markdown(
            "Run a prompt first.",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )
        self.heatmap_pane = pn.pane.Matplotlib(
            None,
            tight=True,
            sizing_mode="stretch_width",
            min_height=360,
        )
        self.scatter_pane = pn.pane.Matplotlib(
            None,
            tight=True,
            sizing_mode="stretch_width",
            min_height=360,
        )
        self.panel = pn.Column(
            self.status,
            self.heatmap_pane,
            self.scatter_pane,
            sizing_mode="stretch_both",
            min_height=760,
        )

    def refresh(self) -> None:
        """Refresh score figures when score data changes."""
        if self.state.scores is None:
            self._last_version = None
            self.status.object = "Run a prompt first."
            self.heatmap_pane.object = None
            self.scatter_pane.object = None
            return

        if self.state.score_version == self._last_version:
            return

        num_layers, num_heads = get_model_dims(self.state)

        heatmap_fig = plot_pos_sym_heatmaps(
            self.state.scores,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        self.heatmap_pane.object = heatmap_fig
        plt.close(heatmap_fig)

        scatter_fig = plot_heads_scatter(
            self.state.scores,
            num_layers=num_layers,
            num_heads=num_heads,
        )

        self.scatter_pane.object = scatter_fig
        plt.close(scatter_fig)

        self.status.object = "Score overview."
        self._last_version = self.state.score_version