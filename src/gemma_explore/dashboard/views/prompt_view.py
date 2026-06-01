"""Tab 5 – Prompt overview: raw text and token-block decomposition."""
from __future__ import annotations

import panel as pn

from gemma_explore.dashboard.state import DashboardState


def _render_prompt_html(state: DashboardState) -> str:
    rec = state.cache.get_prompt(0)
    text: str = rec.get("text", "")
    tokens: list[str] = list(rec.get("tokens", []))
    blocks = list(rec.get("blocks", []))
    apply_chat = state.active_apply_chat_template

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

    if blocks:
        lines.append(f"<h4>Token blocks ({len(blocks)} blocks)</h4>")
        lines.append('<table style="border-collapse:collapse;width:100%;font-size:0.9em;">')
        lines.append(
            "<tr>"
            '<th style="border:1px solid #ddd;padding:6px 10px;background:#eee;text-align:left;">Block</th>'
            '<th style="border:1px solid #ddd;padding:6px 10px;background:#eee;text-align:left;">Name</th>'
            '<th style="border:1px solid #ddd;padding:6px 10px;background:#eee;text-align:right;">Tokens</th>'
            '<th style="border:1px solid #ddd;padding:6px 10px;background:#eee;text-align:right;">Start → End</th>'
            '<th style="border:1px solid #ddd;padding:6px 10px;background:#eee;text-align:left;">Content</th>'
            "</tr>"
        )
        for i, block in enumerate(blocks):
            name = getattr(block, "name", f"block_{i}")
            start = getattr(block, "start", "?")
            end = getattr(block, "end", "?")
            block_tokens = getattr(block, "tokens", [])
            n_tok = len(block_tokens)
            preview = _escape(" ".join(str(t) for t in block_tokens[:20]))
            if len(block_tokens) > 20:
                preview += f" … (+{len(block_tokens) - 20})"
            bg = "#fff" if i % 2 == 0 else "#fafafa"
            lines.append(
                f'<tr style="background:{bg};">'
                f'<td style="border:1px solid #ddd;padding:5px 10px;">{i}</td>'
                f'<td style="border:1px solid #ddd;padding:5px 10px;"><code>{_escape(str(name))}</code></td>'
                f'<td style="border:1px solid #ddd;padding:5px 10px;text-align:right;">{n_tok}</td>'
                f'<td style="border:1px solid #ddd;padding:5px 10px;text-align:right;">{start} → {end}</td>'
                f'<td style="border:1px solid #ddd;padding:5px 10px;font-family:monospace;">{preview}</td>'
                "</tr>"
            )
        lines.append("</table>")
    else:
        lines.append("<p><em>No block decomposition available for this prompt.</em></p>")

    return "\n".join(lines)


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
