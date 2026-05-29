import matplotlib.pyplot as plt
import panel as pn

from gemma_explore.dashboard.state import DashboardState
from gemma_explore.qwen_viz import plot_head_block_dynamics


def make_head_panel(state: DashboardState, layer_slider, head_slider) -> pn.Column:
    @pn.depends(layer=layer_slider, head=head_slider)
    def _render_head_plot(layer, head):
        if state.cache is None:
            return pn.pane.Str("Run a prompt first.")

        fig = plot_head_block_dynamics(
            state.cache,
            prompt_id=state.active_prompt_id,
            layer_idx=layer,
            head_idx=head,
        )
        pane = pn.pane.Matplotlib(fig, tight=True)
        plt.close(fig)
        return pane

    return pn.Column(_render_head_plot)