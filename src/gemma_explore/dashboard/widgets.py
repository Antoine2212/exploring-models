import panel as pn


def make_prompt_input() -> pn.widgets.TextAreaInput:
    return pn.widgets.TextAreaInput(name="Prompt", placeholder="Enter a prompt…", rows=4, sizing_mode="stretch_width")


def make_run_button() -> pn.widgets.Button:
    return pn.widgets.Button(name="Run", button_type="primary", sizing_mode="stretch_width")


def make_chat_toggle() -> pn.widgets.Checkbox:
    return pn.widgets.Checkbox(name="Chat mode (apply_chat_template)", value=True)
