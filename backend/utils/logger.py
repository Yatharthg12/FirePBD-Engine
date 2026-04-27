"""
FirePBD Engine — Structured Logger
====================================
Provides a consistent, coloured logger for all backend modules.
"""
import logging
import os
import sys

from backend.config import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL

# Avoid ANSI escape spam on standard Windows terminals unless VT processing is
# explicitly available.
_USE_COLOUR = bool(
    hasattr(sys.stdout, "isatty")
    and sys.stdout.isatty()
    and (os.name != "nt" or os.getenv("WT_SESSION") or os.getenv("ANSICON"))
)

_COLOURS = {
    "DEBUG": "\033[36m",
    "INFO": "\033[32m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[35m",
} if _USE_COLOUR else {}
_RESET = "\033[0m"


class _ColouredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        colour = _COLOURS.get(record.levelname, "")
        if colour:
            record.levelname = f"{colour}{record.levelname}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


class _SafeStreamHandler(logging.StreamHandler):
    """Replace characters unsupported by the active terminal encoding."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            encoding = getattr(stream, "encoding", None) or "utf-8"
            msg = msg.encode(encoding, errors="replace").decode(
                encoding, errors="replace"
            )
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger configured with safe console output.
    Call once per module: `logger = get_logger(__name__)`
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    handler = _SafeStreamHandler(sys.stdout)
    handler.setFormatter(_ColouredFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False

    return logger
