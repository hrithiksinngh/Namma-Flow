"""Project-wide logger factory."""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to stderr; level comes from ``NAMMA_FLOW_LOG_LEVEL``."""
    global _CONFIGURED
    if not _CONFIGURED:
        level_name = os.environ.get("NAMMA_FLOW_LOG_LEVEL", "INFO").upper()
        root = logging.getLogger("namma_flow")
        if not root.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
            root.addHandler(handler)
        root.setLevel(getattr(logging, level_name, logging.INFO))
        root.propagate = False
        _CONFIGURED = True
    short = name.removeprefix("src.")
    return logging.getLogger(f"namma_flow.{short}")
