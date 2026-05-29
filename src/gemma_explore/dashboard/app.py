from __future__ import annotations

import panel as pn

from gemma_explore.qwen_core import QwenBundle, load_bundle
from gemma_explore.dashboard import state, widgets
import gemma_explore.dashboard.views.freq_view as freq_view
import gemma_explore.dashboard.views.head_view as head_view
import gemma_explore.dashboard.views.score_view as score_view


def build_app(bundle: QwenBundle) -> pn.template.base.BasicTemplate:
    dashboard_state = state.DashboardState(bundle=bundle)

    layer_slider = widgets.make_layer_slider(bundle.num_layers)
    head_slider = widgets.make_head_slider(bundle.num_heads)
    prompt_input = widgets.make_prompt_input()
    run_button = widgets.make_run_button()

    tabs = pn.Tabs(dynamic=False, sizing_mode="stretch_both")

    def rebuild_views() -> None:
        tabs[:] = [
            ("Head detail", head_view.make_head_panel(dashboard_state, layer_slider, head_slider)),
            ("Scores", score_view.make_score_panel(dashboard_state)),
            ("Frequency", freq_view.make_freq_panel(dashboard_state, layer_slider, head_slider)),
        ]

    template = pn.template.FastListTemplate(
        title="Gemma Explore Dashboard",
        sidebar=[
            pn.Column(
                prompt_input,
                run_button,
                layer_slider,
                head_slider,
                width=320,
                sizing_mode="fixed",
            )
        ],
        main=[tabs],
    )

    rebuild_views()

    def _on_run(_: object) -> None:
        state.run_prompt(dashboard_state, prompt_input.value)
        rebuild_views()
        template.param.trigger("name")

    run_button.on_click(_on_run)
    return template


def launch_dashboard(bundle: QwenBundle, port: int = 5006, show: bool = True) -> None:
    pn.extension("matplotlib")
    app = build_app(bundle).servable()
    pn.serve(app, port=port, show=show)


if __name__ == "__main__":
    launch_dashboard(load_bundle())