"""
Configure logging with colored console output and file logging,
adapted for multi-GPU with Lightning’s rank_zero_only decorator.
"""

import logging
import sys
import traceback
from typing import Callable

from colorama import Fore, Style, init
from lightning.pytorch.utilities import rank_zero_only

init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds ANSI colors based on log level.

    DEBUG: cyan, INFO: green, WARNING: yellow, ERROR: red, CRITICAL: magenta+bright.
    """

    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        """
        Apply base formatting and wrap result in level-specific ANSI codes.

        Args:
            record: the LogRecord to format.

        Returns:
            Colored log message string.
        """
        message = super().format(record)
        color = self.COLORS.get(record.levelname, "")
        return f"{color}{message}{Style.RESET_ALL}"


def configure_logging(log_file: str = "app.log") -> None:
    """
    Set up global logging configuration:
    - FileHandler writing to `log_file`
    - StreamHandler with colored output
    - All log methods wrapped with rank_zero_only for Lightning multi-GPU safety
    - Global exception hook that logs uncaught exceptions

    Args:
        log_file: filename for persistent logs.
    """
    # Handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        ColoredFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, console_handler], force=True)

    # Wrap logging methods to run only on rank zero in distributed setups
    root_logger = logging.getLogger()
    for method in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(root_logger, method, rank_zero_only(getattr(root_logger, method)))  # type: ignore

    def _exception_handler(exc_type, exc_value, exc_traceback) -> None:
        """
        Global exception hook that prints and logs uncaught exceptions.

        Writes the traceback in red to console and appends it to the log file.
        """
        tb = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(f"{Fore.RED}{tb}{Style.RESET_ALL}")
        with open(log_file, "a") as f:
            f.write(tb)

    sys.excepthook = _exception_handler
