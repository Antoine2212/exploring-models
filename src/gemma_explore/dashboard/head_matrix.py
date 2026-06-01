"""Clickable layer×head matrix coloured by pos/sym score."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import panel as pn
from bokeh.models import (
    ColorBar,
    ColumnDataSource,
    FixedTicker,
    HoverTool,
    LinearColorMapper,
    TapTool,
)
from bokeh.palettes import Viridis256
from bokeh.plotting import figure


class HeadMatrix:
    """
    Bokeh heatmap (layers × heads) that acts as a head selector.

    Clicking a cell sets ``selected_layer`` and ``selected_head`` on the
    provided state object and calls the optional ``on_select`` callback.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        on_select: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self._nl = num_layers
        self._nh = num_heads
        self._on_select = on_select
        self._sel_layer = 0
        self._sel_head = 0
        self._source: Optional[ColumnDataSource] = None
        self._sel_source: Optional[ColumnDataSource] = None
        self._mapper: Optional[LinearColorMapper] = None
        self._pane: Optional[pn.pane.Bokeh] = None
        self._score_toggle: Optional[pn.widgets.RadioButtonGroup] = None
        self._pos_flat: list[float] = [0.0] * (num_layers * num_heads)
        self._sym_flat: list[float] = [0.0] * (num_layers * num_heads)
        self._score_mode: str = "pos"
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        nl, nh = self._nl, self._nh

        layers = np.repeat(np.arange(nl), nh).tolist()
        heads = np.tile(np.arange(nh), nl).tolist()
        scores = [0.0] * (nl * nh)
        layer_labels = [str(l) for l in layers]
        head_labels = [str(h) for h in heads]

        self._source = ColumnDataSource(dict(
            x=heads,
            y=layers,
            score=scores,
            layer_label=layer_labels,
            head_label=head_labels,
        ))

        self._mapper = LinearColorMapper(
            palette=Viridis256, low=0.0, high=1.0, nan_color="lightgrey"
        )

        cell_w = max(16, min(28, 400 // nh))
        cell_h = max(12, min(18, 400 // nl))
        pw = nh * cell_w + 80
        ph = nl * cell_h + 60

        p = figure(
            width=pw,
            height=ph,
            x_range=(-0.5, nh - 0.5),
            y_range=(-0.5, nl - 0.5),
            tools="tap",
            toolbar_location=None,
        )
        p.rect(
            x="x",
            y="y",
            width=0.95,
            height=0.95,
            source=self._source,
            fill_color={"field": "score", "transform": self._mapper},
            line_color=None,
        )

        # Selection highlight
        self._sel_source = ColumnDataSource(dict(x=[0], y=[0]))
        p.rect(
            x="x",
            y="y",
            width=0.95,
            height=0.95,
            source=self._sel_source,
            fill_color=None,
            line_color="red",
            line_width=2,
        )

        # Colour bar
        cb = ColorBar(color_mapper=self._mapper, width=10, padding=4)
        p.add_layout(cb, "right")

        # Hover
        p.add_tools(
            HoverTool(
                tooltips=[
                    ("Layer", "@layer_label"),
                    ("Head", "@head_label"),
                    ("Score", "@score{0.000}"),
                ]
            )
        )

        # Axis ticks – show every other layer to avoid crowding
        step = max(1, nl // 8)
        p.yaxis.ticker = FixedTicker(ticks=list(range(0, nl, step)))
        p.xaxis.ticker = FixedTicker(ticks=list(range(nh)))
        p.xaxis.axis_label = "Head"
        p.yaxis.axis_label = "Layer"
        p.axis.axis_label_text_font_size = "10px"
        p.axis.major_label_text_font_size = "9px"

        self._source.selected.on_change("indices", self._on_tap)

        self._pane = pn.pane.Bokeh(p, sizing_mode="stretch_width")

        self._score_toggle = pn.widgets.RadioButtonGroup(
            options=["pos", "sym"],
            value="pos",
            button_type="default",
            sizing_mode="stretch_width",
        )
        self._score_toggle.param.watch(self._on_toggle, "value")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_tap(self, attr: str, old: list, new: list) -> None:
        if not new:
            return
        idx = new[0]
        layer = int(self._source.data["y"][idx])
        head = int(self._source.data["x"][idx])
        # Clear Bokeh's own selection highlight; we draw ours manually
        self._source.selected.indices = []
        self._sel_source.data = dict(x=[head], y=[layer])
        self._sel_layer = layer
        self._sel_head = head
        if self._on_select is not None:
            self._on_select(layer, head)

    def _on_toggle(self, event) -> None:
        self._score_mode = event.new
        self._refresh_colors()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_scores(self, pos_flat: list[float], sym_flat: list[float]) -> None:
        """Replace score data (flat list, len = layers*heads, row-major)."""
        self._pos_flat = pos_flat
        self._sym_flat = sym_flat
        self._refresh_colors()

    def set_selected(self, layer: int, head: int) -> None:
        self._sel_layer = layer
        self._sel_head = head
        self._sel_source.data = dict(x=[head], y=[layer])

    def _refresh_colors(self) -> None:
        flat = self._pos_flat if self._score_mode == "pos" else self._sym_flat
        arr = np.array(flat, dtype=float)
        vmin, vmax = float(arr.min()), float(arr.max())
        span = max(vmax - vmin, 1e-6)
        self._mapper.low = vmin
        self._mapper.high = vmin + span
        self._source.data["score"] = flat

    @property
    def panel(self) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown("**Head selector** — click to select, colour = score", margin=(0, 0, 2, 0)),
            self._score_toggle,
            self._pane,
            sizing_mode="stretch_width",
        )
