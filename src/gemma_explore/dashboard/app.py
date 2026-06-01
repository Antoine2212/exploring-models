from __future__ import annotations

from typing import Any, Callable

import panel as pn
from param.parameterized import Event

from gemma_explore.dashboard import state, widgets
from gemma_explore.dashboard.views.freq_view import FrequencyView
from gemma_explore.dashboard.views.head_view import HeadView
from gemma_explore.dashboard.views.score_view import ScoreView
from gemma_explore.qwen_core import QwenBundle, load_bundle


HistoryOption = tuple[str, str]


def _coerce_prompt_text(value: Any) -> str:
    """Normalize a prompt-like value to displayable text."""
    if value is None:
        return ""
    return str(value).strip()


def _make_prompt_preview(prompt_text: str, max_len: int = 72) -> str:
    """Create a readable one-line preview label for a prompt."""
    normalized = " ".join(prompt_text.split())
    if len(normalized) <= max_len:
        return normalized
    return f"{normalized[: max_len - 1].rstrip()}…"


def _extract_history_options(dashboard_state: state.DashboardState) -> list[HistoryOption]:
    """Build prompt history options from dashboard state while preserving order."""
    prompts = getattr(dashboard_state, "prompt_history", None)
    if prompts is None:
        prompts = getattr(dashboard_state, "history", None)

    options: list[HistoryOption] = []
    seen_prompts: set[str] = set()
    if not prompts:
        return options

    for item in prompts:
        prompt_text = ""
        if isinstance(item, str):
            prompt_text = item
        else:
            for attr_name in ("prompt", "prompt_text", "text", "raw_prompt"):
                candidate = getattr(item, attr_name, None)
                if candidate:
                    prompt_text = str(candidate)
                    break
            if not prompt_text and isinstance(item, dict):
                for key in ("prompt", "prompt_text", "text", "raw_prompt"):
                    candidate = item.get(key)
                    if candidate:
                        prompt_text = str(candidate)
                        break

        prompt_text = _coerce_prompt_text(prompt_text)
        if not prompt_text or prompt_text in seen_prompts:
            continue

        seen_prompts.add(prompt_text)
        options.append((_make_prompt_preview(prompt_text), prompt_text))

    return options


class PromptHistoryController:
    """Synchronize the prompt history widget with dashboard state."""

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
        """Refresh widget options from state while avoiding recursive callbacks."""
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
        """Select the active prompt in the widget if present."""
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
        """Restore a previous prompt from cache without recomputation."""
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
        except Exception as exc:  # pragma: no cover - UI surface
            self._status_pane.object = f"Prompt restore failed: `{exc}`"
            self.sync_from_state(preserve_selection=False)


def build_app(bundle: QwenBundle) -> pn.template.base.BasicTemplate:
    """Build the dashboard with persistent widgets and views."""
    dashboard_state = state.DashboardState(bundle=bundle)

    layer_slider = widgets.make_layer_slider(bundle.num_layers)
    head_slider = widgets.make_head_slider(bundle.num_heads)
    prompt_input = widgets.make_prompt_input()
    run_button = widgets.make_run_button()
    status_pane = pn.pane.Markdown(
        "Enter a prompt and click **Run**.",
        sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )

    head_view = HeadView(dashboard_state, layer_slider, head_slider)
    score_view = ScoreView(dashboard_state)
    freq_view = FrequencyView(dashboard_state, layer_slider, head_slider)

    tabs = pn.Tabs(
        ("Head detail", head_view.panel),
        ("Scores", score_view.panel),
        ("Frequency", freq_view.panel),
        dynamic=True,
        sizing_mode="stretch_both",
        tabs_location="above",
    )

    def _refresh_all_views() -> None:
        """Refresh all persistent view panes."""
        head_view.refresh()
        score_view.refresh()
        freq_view.refresh()

    history_controller = PromptHistoryController(
        dashboard_state=dashboard_state,
        prompt_input=prompt_input,
        status_pane=status_pane,
        refresh_views=_refresh_all_views,
    )

    def _set_controls_enabled(enabled: bool) -> None:
        """Enable or disable interactive controls."""
        prompt_input.disabled = not enabled
        run_button.disabled = not enabled
        layer_slider.disabled = not enabled
        head_slider.disabled = not enabled
        history_controller.widget.disabled = not enabled

    def _on_slider_change(_: object) -> None:
        """Refresh only slider-dependent views."""
        head_view.refresh()
        freq_view.refresh()

    def _on_run(_: object) -> None:
        """Run the prompt and update existing views in place."""
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
        except Exception as exc:  # pragma: no cover - UI surface
            status_pane.object = f"Run failed: `{exc}`"
        finally:
            run_button.name = "Run"
            _set_controls_enabled(True)

    layer_slider.param.watch(_on_slider_change, "value")
    head_slider.param.watch(_on_slider_change, "value")
    run_button.on_click(_on_run)

    sidebar = pn.Column(
        prompt_input,
        run_button,
        pn.Spacer(height=8),
        history_controller.widget,
        pn.Spacer(height=8),
        layer_slider,
        head_slider,
        pn.Spacer(height=8),
        status_pane,
        width=320,
        min_width=300,
        max_width=360,
        sizing_mode="fixed",
    )

    template = pn.template.FastListTemplate(
        title="Gemma Explore Dashboard",
        sidebar=[sidebar],
        main=[
            pn.Column(
                tabs,
                sizing_mode="stretch_both",
                min_height=700,
            )
        ],
        main_layout=None,
        accent_base_color="#1f77b4",
    )
    return template


def launch_dashboard(bundle: QwenBundle, port: int = 5006, show: bool = True) -> None:
    """Serve the dashboard."""
    pn.extension("matplotlib")
    app = build_app(bundle).servable()
    pn.serve(app, port=port, show=show)


if __name__ == "__main__":
    launch_dashboard(load_bundle())