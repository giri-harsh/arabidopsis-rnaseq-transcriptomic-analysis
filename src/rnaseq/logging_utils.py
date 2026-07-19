"""
logging_utils.py
=================

Centralized logging configuration.

Every module in the pipeline calls ``get_logger(__name__)`` rather than
configuring its own handlers. This keeps log formatting consistent and lets
``main.py`` decide once, at startup, where logs go (console + a run-specific
file under ``outputs/logs/``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def setup_logging(
    logs_dir: Optional[Path] = None,
    level: int = logging.INFO,
    log_filename: str = "pipeline.log",
) -> logging.Logger:
    """
    Configure the root ``rnaseq`` logger once per process.

    Parameters
    ----------
    logs_dir:
        Directory to write ``log_filename`` into. If ``None``, only console
        logging is configured (useful for unit tests / notebooks).
    level:
        Logging level for both handlers.
    log_filename:
        Name of the log file written under ``logs_dir``.

    Returns
    -------
    The configured root logger for the ``rnaseq`` namespace.
    """
    global _CONFIGURED

    root_logger = logging.getLogger("rnaseq")
    root_logger.setLevel(level)

    if _CONFIGURED:
        return root_logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)

    if logs_dir is not None:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs_dir / log_filename, mode="a")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)

    root_logger.propagate = False
    _CONFIGURED = True
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger nested under the ``rnaseq`` namespace.

    If ``setup_logging`` has not yet been called, this returns a logger
    with no handlers attached; Python's logging module will fall back to
    ``logging.lastResort`` (stderr) so nothing is silently dropped, but for
    proper file logging call ``setup_logging`` first (``main.py`` does this
    automatically).
    """
    if not name.startswith("rnaseq"):
        name = f"rnaseq.{name}"
    return logging.getLogger(name)
