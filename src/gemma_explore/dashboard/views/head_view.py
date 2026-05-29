from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import panel as pn

from gemma_explore.dashboard.state import DashboardState
from gemma_explore.qwen_viz import plot_head_block_dynamics



@dataclass
class HeadView:
    """Persistent head-detail view."""

    state: DashboardState
    layer_slider: pn.widgets.IntSlider
    head_slider: pn.widgets.IntSlider

    def __post_init__(self) -> None:
        self._last_key: tuple[int, int, int] | None = None
        self.status = pn.pane.Markdown(
            "Run a prompt first.",
            sizing_mode="stretch_width",
            margin=(0, 0, 8, 0),
        )
        self.plot_pane = pn.pane.Matplotlib(
            None,
            tight=True,
            sizing_mode="stretch_width",
            min_height=360,
        )
        self.panel = pn.Column(
            self.status,
            self.plot_pane,
            sizing_mode="stretch_both",
            min_height=420,
        )

    def refresh(self) -> None:
        """Refresh the head plot if needed."""
        if self.state.cache is None:
            self._last_key = None
            self.status.object = "Run a prompt first."
            self.plot_pane.object = None
            return

        key = (
            self.state.data_version,
            int(self.layer_slider.value),
            int(self.head_slider.value),
        )
        if key == self._last_key:
            return

        fig = plot_head_block_dynamics(
            self.state.cache,
            prompt_id=self.state.active_prompt_id,
            layer_idx=key[1],
            head_idx=key[2],
        )


        self.plot_pane.object = fig
        plt.close(fig)

        self.status.object = (
            f"Layer {key[1]}, head {key[2]}."
        )
        self._last_key = key