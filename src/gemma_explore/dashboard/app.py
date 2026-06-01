from __future__ import annotations

from typing import Any, Callable

import numpy as np
import panel as pn
from param.parameterized import Event

from gemma_explore.dashboard import state, widgets
from gemma_explore.dashboard.head_matrix import HeadMatrix
from gemma_explore.dashboard.views.freq_view import FreqView
from gemma_explore.dashboard.views.hidden_view import HiddenView
from gemma_explore.dashboard.views.layer_view import LayerView
from gemma_explore.dashboard.views.model_view import ModelView
from gemma_explore.qwen_core import QwenBundle, load_bundle


HistoryOption = tuple[str, str]


def _coerce_prompt_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()



def _extract_history_options(dashboard_state: state.DashboardState) -> list[HistoryOption]:
    options: list[HistoryOption] = []
    for entry in dashboard_state.registry.entries:
        options.append((entry.display_label, entry.prompt_text))
    return options


class PromptHistoryController:
    def __init__(
        self,
        dashboard_state: state.DashboardState,
        prompt_input: pn.widgets.TextAreaInput,
        status_pane: pn.pane.Markdown,
        refresh_views: Callable[[], None],
    ) -> None:
        self._dashboard_state = dashboard_state
        self._prompt_input = prompt_input
        self._status_pane = status_pane
        self._refresh_views = refresh_views
        self._is_syncing = False

        self.widget = pn.widgets.Select(
            name="Prompt history",
            options={},
            value=None,
            sizing_mode="stretch_width",
        )
        self.widget.param.watch(self._on_history_selected, "value")
        self.sync_from_state(preserve_selection=False)

    def sync_from_state(self, preserve_selection: bool = True) -> None:
        current_value = self.widget.value if preserve_selection else None
        active_prompt = _coerce_prompt_text(getattr(self._dashboard_state, "prompt", ""))
        options_list = _extract_history_options(self._dashboard_state)
        options = {label: prompt for label, prompt in options_list}

        next_value = None
        if current_value in options.values():
            next_value = current_value
        elif active_prompt and active_prompt in options.values():
            next_value = active_prompt

        self._is_syncing = True
        try:
            self.widget.options = options
            self.widget.value = next_value
        finally:
            self._is_syncing = False

    def set_active_prompt(self, prompt_text: str) -> None:
        normalized_prompt = _coerce_prompt_text(prompt_text)
        if normalized_prompt not in set(self.widget.options.values()):
            self.sync_from_state(preserve_selection=True)
            return

        self._is_syncing = True
        try:
            self.widget.value = normalized_prompt
        finally:
            self._is_syncing = False

    def _on_history_selected(self, event: Event) -> None:
        if self._is_syncing:
            return

        prompt_text = _coerce_prompt_text(event.new)
        if not prompt_text:
            return

        current_prompt = _coerce_prompt_text(getattr(self._dashboard_state, "prompt", ""))
        if prompt_text == current_prompt:
            self._prompt_input.value = prompt_text
            return

        try:
            restore_prompt = getattr(self._dashboard_state, "restore_prompt", None)
            if callable(restore_prompt):
                restore_prompt(prompt_text)
            else:
                state.restore_prompt(self._dashboard_state, prompt_text)
            self._prompt_input.value = prompt_text
            self._status_pane.object = "Prompt restored from session history."
            self._refresh_views()
        except Exception as exc:
            self._status_pane.object = f"Prompt restore failed: `{exc}`"
            self.sync_from_state(preserve_selection=False)


def _extract_score_flat(scores: Any, key: str, num_layers: int, num_heads: int) -> list[float]:
    raw = scores[key] if isinstance(scores, dict) else getattr(scores, key)
    arr = np.array(raw, dtype=float).reshape(num_layers, num_heads)
    return arr.reshape(-1).tolist()


def build_app(bundle: QwenBundle) -> pn.template.base.BasicTemplate:
    dashboard_state = state.DashboardState(bundle=bundle)

    prompt_input = widgets.make_prompt_input()
    run_button = widgets.make_run_button()
    status_pane = pn.pane.Markdown(
        "Enter a prompt and click **Run**.",
        sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )

    model_view = ModelView(dashboard_state)
    layer_view = LayerView(dashboard_state)
    freq_view = FreqView(dashboard_state)
    hidden_view = HiddenView(dashboard_state)

    tabs = pn.Tabs(
        ("Model overview", model_view.panel),
        ("Layer heads", layer_view.panel),
        ("Frequency", freq_view.panel),
        ("Hidden states", hidden_view.panel),
        dynamic=True,
        sizing_mode="stretch_both",
        tabs_location="above",
    )

    def _on_head_selected(layer: int, head: int) -> None:
        dashboard_state.selected_layer = layer
        dashboard_state.selected_head = head
        layer_view.refresh()
        freq_view.refresh()
        if tabs.active == 3:
            hidden_view.refresh()

    matrix = HeadMatrix(
        num_layers=bundle.num_layers,
        num_heads=bundle.num_heads,
        on_select=_on_head_selected,
    )

    def _update_matrix_scores() -> None:
        s = dashboard_state
        if s.scores is None:
            return
        nl, nh = bundle.num_layers, bundle.num_heads
        pos_flat = _extract_score_flat(s.scores, "pos_scores", nl, nh)
        sym_flat = _extract_score_flat(s.scores, "sym_scores", nl, nh)
        matrix.update_scores(pos_flat, sym_flat)

    def _refresh_all_views() -> None:
        model_view.refresh()
        layer_view.refresh()
        freq_view.refresh()
        _update_matrix_scores()
        if tabs.active == 3:
            hidden_view.refresh()

    def _on_tab_change(event: Event) -> None:
        if event.new == 3:
            hidden_view.activate()
        else:
            hidden_view.deactivate()

    tabs.param.watch(_on_tab_change, "active")

    history_controller = PromptHistoryController(
        dashboard_state=dashboard_state,
        prompt_input=prompt_input,
        status_pane=status_pane,
        refresh_views=_refresh_all_views,
    )

    def _set_controls_enabled(enabled: bool) -> None:
        prompt_input.disabled = not enabled
        run_button.disabled = not enabled
        history_controller.widget.disabled = not enabled

    def _on_run(_: object) -> None:
        prompt_text = prompt_input.value.strip()
        if not prompt_text:
            status_pane.object = "Please enter a prompt."
            return

        _set_controls_enabled(False)
        run_button.name = "Running..."
        status_pane.object = "Running prompt and updating plots..."

        try:
            changed = state.run_prompt(dashboard_state, prompt_text)
            history_controller.sync_from_state(preserve_selection=True)
            history_controller.set_active_prompt(prompt_text)
            if changed:
                status_pane.object = "Prompt completed."
            else:
                status_pane.object = "Prompt unchanged; reused cached results."
            _refresh_all_views()
        except Exception as exc:
            status_pane.object = f"Run failed: `{exc}`"
        finally:
            run_button.name = "Run"
            _set_controls_enabled(True)

    run_button.on_click(_on_run)

    sidebar = pn.Column(
        prompt_input,
        run_button,
        pn.Spacer(height=8),
        history_controller.widget,
        pn.Spacer(height=8),
        status_pane,
        width=300,
        min_width=280,
        max_width=340,
        sizing_mode="fixed",
    )

    template = pn.template.FastListTemplate(
        title="Gemma Explore Dashboard",
        sidebar=[sidebar],
        main=[
            pn.Column(
                matrix.panel,
                pn.layout.Divider(),
                tabs,
                sizing_mode="stretch_both",
                min_height=900,
            )
        ],
        main_layout=None,
        accent_base_color="#1f77b4",
    )
    return template


def launch_dashboard(bundle: QwenBundle, port: int = 5006, show: bool = True) -> None:
    pn.extension("bokeh")
    app = build_app(bundle).servable()
    pn.serve(app, port=port, show=show)


if __name__ == "__main__":
    launch_dashboard(load_bundle())
