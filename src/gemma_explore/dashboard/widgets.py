import panel as pn


def make_layer_slider(num_layers: int) -> pn.widgets.IntSlider:
    return pn.widgets.IntSlider(name="Layer", start=0, end=max(0, num_layers - 1), value=0, step=1, width=300)


def make_head_slider(num_heads: int) -> pn.widgets.IntSlider:
    return pn.widgets.IntSlider(name="Head", start=0, end=max(0, num_heads - 1), value=0, step=1, width=300)


def make_prompt_input() -> pn.widgets.TextAreaInput:
    return pn.widgets.TextAreaInput(name="Prompt", placeholder="Enter a prompt…", width=300)


def make_run_button() -> pn.widgets.Button:
    return pn.widgets.Button(name="Run", button_type="primary", width=300)