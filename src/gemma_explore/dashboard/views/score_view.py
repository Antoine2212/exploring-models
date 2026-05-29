import matplotlib.pyplot as plt
import panel as pn

from gemma_explore.dashboard.state import DashboardState, get_model_dims
from gemma_explore.qwen_viz import plot_heads_scatter, plot_pos_sym_heatmaps


def make_score_panel(state: DashboardState) -> pn.Column:
    if state.scores is None:
        return pn.Column(pn.pane.Str("Run a prompt first."))

    num_layers, num_heads = get_model_dims(state)

    heatmap_fig = plot_pos_sym_heatmaps(
        state.scores,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    heatmap_pane = pn.pane.Matplotlib(heatmap_fig, tight=True)
    plt.close(heatmap_fig)

    scatter_fig = plot_heads_scatter(
        state.scores,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    scatter_pane = pn.pane.Matplotlib(scatter_fig, tight=True)
    plt.close(scatter_fig)

    return pn.Column(
        heatmap_pane,
        scatter_pane,
    )