from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import panel as pn

from gemma_explore.dashboard.state import DashboardState
from gemma_explore.qwen_viz import plot_frequency_analysis




@dataclass
class FrequencyView:
    """Persistent frequency-analysis view."""

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
        """Refresh the frequency plot if needed."""
        if self.state.freq_scores is None:
            self._last_key = None
            self.status.object = "Run a prompt first."
            self.plot_pane.object = None
            return

        key = (
            self.state.freq_version,
            int(self.layer_slider.value),
            int(self.head_slider.value),
        )
        if key == self._last_key:
            return

        fig = plot_frequency_analysis(
            self.state.freq_scores,
            layer_idx=key[1],
            head_idx=key[2],
        )


        self.plot_pane.object = fig
        plt.close(fig)

        self.status.object = (
            f"Frequency analysis for layer {key[1]}, head {key[2]}."
        )
        self._last_key = key