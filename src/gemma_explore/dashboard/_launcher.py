"""Entry point for `uv run gemma-dashboard`."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src layout is on sys.path regardless of how the script is invoked.
_src = Path(__file__).resolve().parents[3]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gemma_explore.dashboard.app import launch_dashboard  # noqa: E402
from gemma_explore.qwen_core import load_bundle  # noqa: E402


def main() -> None:
    launch_dashboard(load_bundle())


if __name__ == "__main__":
    main()
