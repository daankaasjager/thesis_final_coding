import logging
import sys
import traceback

from colorama import Fore, Style, init
from lightning.pytorch.utilities import rank_zero_only

init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
    }

    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, '')}{log_message}{Style.RESET_ALL}"


def configure_logging():
    colored_formatter = ColoredFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler("app.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(colored_formatter)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])

    # Wrap logging methods with rank_zero_only for multi-GPU support.
    # https://github.com/kuleshov-group/mdlm/tree/bbc8fb61077a3ca38eab2423d0f81a5484754f51
    logger = logging.getLogger()
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    def exception_handler(exc_type, exc_value, exc_traceback):
        formatted_exception = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        print(Fore.RED + formatted_exception + Style.RESET_ALL)
        with open("app.log", "a") as log_file:
            log_file.write(formatted_exception)

    sys.excepthook = exception_handler
