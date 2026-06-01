"""Tab 5 – Prompt overview: raw text and token-block decomposition."""
from __future__ import annotations

import panel as pn

from gemma_explore.dashboard.state import DashboardState


def _render_prompt_html(state: DashboardState) -> str:
    rec = state.cache.get_prompt(0)
    text: str = rec.get("text", "")
    tokens: list[str] = list(rec.get("tokens", []))
    apply_chat = state.active_apply_chat_template

    # Computation blocks from scores (split_into_blocks), not the single semantic PromptBlock
    score_blocks: list[tuple[int, int]] = []
    if state.scores is not None:
        score_blocks = list(state.scores.get("blocks", []))

    mode_badge = (
        '<span style="background:#1f77b4;color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;">chat</span>'
        if apply_chat else
        '<span style="background:#666;color:white;padding:2px 8px;border-radius:4px;font-size:0.85em;">raw</span>'
    )

    lines = [
        f"<h3>Prompt overview {mode_badge}</h3>",
        "<h4>Full text</h4>",
        f'<pre style="background:#f5f5f5;padding:12px;border-radius:6px;white-space:pre-wrap;word-break:break-word;">{_escape(text)}</pre>',
        f"<p><strong>Total tokens:</strong> {len(tokens)}</p>",
    ]

    if score_blocks:
        lines.append(f"<h4>Computation blocks ({len(score_blocks)} blocks)</h4>")
        lines.append('<table style="border-collapse:collapse;width:100%;font-size:0.9em;">')
        lines.append(
            "<tr>"
            '<th style="border:1px solid #ddd;padding:6px 10px;background:#eee;text-align:left;">Block</th>'
            '<th style="border:1px solid #ddd;padding:6px 10px;background:#eee;text-align:right;">Tokens</th>'
            '<th style="border:1px solid #ddd;padding:6px 10px;background:#eee;text-align:right;">Start → End</th>'
            '<th style="border:1px solid #ddd;padding:6px 10px;background:#eee;text-align:left;">Content</th>'
            "</tr>"
        )
        for i, (start, end) in enumerate(score_blocks):
            block_tokens = tokens[start:end]
            n_tok = len(block_tokens)
            preview = _escape(" ".join(_clean_token(str(t)) for t in block_tokens[:20]))
            if n_tok > 20:
                preview += f" … (+{n_tok - 20})"
            bg = "#fff" if i % 2 == 0 else "#fafafa"
            lines.append(
                f'<tr style="background:{bg};">'
                f'<td style="border:1px solid #ddd;padding:5px 10px;">{i}</td>'
                f'<td style="border:1px solid #ddd;padding:5px 10px;text-align:right;">{n_tok}</td>'
                f'<td style="border:1px solid #ddd;padding:5px 10px;text-align:right;">{start} → {end}</td>'
                f'<td style="border:1px solid #ddd;padding:5px 10px;font-family:monospace;">{preview}</td>'
                "</tr>"
            )
        lines.append("</table>")
    else:
        lines.append("<p><em>No computation blocks available — run a prompt first.</em></p>")

    return "\n".join(lines)


def _clean_token(token: str) -> str:
    """Replace tokenizer special glyphs with readable ASCII equivalents."""
    return token.replace("Ġ", "_").replace("Ċ", "\\n")


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class PromptView:
    def __init__(self, state: DashboardState) -> None:
        self._state = state
        self._last_key: str = ""

        self._status = pn.pane.Markdown(
            "Run a prompt to see its overview.",
            sizing_mode="stretch_width",
            margin=(0, 0, 4, 0),
        )
        self._content = pn.pane.HTML("", sizing_mode="stretch_width", min_height=200)
        self.panel = pn.Column(
            self._status,
            self._content,
            sizing_mode="stretch_both",
        )

    def refresh(self) -> None:
        s = self._state
        if s.cache is None:
            self._status.object = "Run or select a prompt first."
            self._content.object = ""
            self._last_key = ""
            return

        key = s.active_prompt_hash
        if key == self._last_key:
            return

        self._content.object = _render_prompt_html(s)
        self._status.object = ""
        self._last_key = key
