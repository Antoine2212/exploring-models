import matplotlib.pyplot as plt
import panel as pn

from gemma_explore.dashboard.state import DashboardState
from gemma_explore.qwen_viz import plot_frequency_analysis


def make_freq_panel(
    state: DashboardState,
    layer_slider,
    head_slider,
) -> pn.Column:
    @pn.depends(layer=layer_slider.param.value, head=head_slider.param.value)
    def _frequency_view(layer: int, head: int):
        if state.freq_scores is None:
            return pn.pane.Str("Run a prompt first.")

        fig = plot_frequency_analysis(
            state.freq_scores,
            layer_idx=layer,
            head_idx=head,
        )
        pane = pn.pane.Matplotlib(fig, tight=True)
        plt.close(fig)
        return pane

    return pn.Column(_frequency_view)