import logging
import re
import sys
from pathlib import Path
import datetime

# The logger to use
logger = None
file_logger = None

LOG_DIR = Path("log")


def red(string):
    """Adds ANSI escape codes to print a string in red around the string."""
    return "\033[38;5;196m" + str(string) + "\033[0m"


def green(string):
    """Adds ANSI escape codes to print a string in green around the string."""
    return "\033[38;5;46m" + str(string) + "\033[0m"


def yellow(string):
    """Adds ANSI escape codes to print a string in yellow around the string."""
    return "\033[38;5;226m" + str(string) + "\033[0m"


def blue(string):
    """Adds ANSI escape codes to print a string in blue around the string."""
    return "\033[38;5;36m" + str(string) + "\033[0m"


def orange(string):
    """Adds ANSI escape codes to print a string in orange around the string."""
    return "\033[38;5;214m" + str(string) + "\033[0m"


class CustomFormatter(logging.Formatter):
    """Custom formatter to use different format strings for different log levels"""

    FORMATS = {
        logging.INFO: logging.Formatter(
            ("".join([blue("%(levelname)s"), " - %(message)s"]))
        ),
        logging.WARNING: logging.Formatter(
            ("".join([orange("%(levelname)s"), " - %(message)s"]))
        ),
        logging.ERROR: logging.Formatter(
            ("".join([red("%(levelname)s"), " - %(message)s"]))
        ),
        logging.DEBUG: logging.Formatter(
            ("".join([yellow("%(levelname)s"), " - %(message)s"]))
        ),
        "DEFAULT": logging.Formatter(
            ("".join([yellow("%(levelname)s"), " - %(message)s"]))
        ),
    }

    def format(self, record):
        formatter = self.FORMATS.get(record.levelno, self.FORMATS["DEFAULT"])
        return formatter.format(record)


def configure_console_logger(level="INFO"):
    """
    Configures a simple console logger with the givel level.
    A usecase is to change the formatting of the default handler of the root logger
    """
    assert level in [
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ], f"Invalid log level: {level}"

    # Create a logger
    new_logger = logging.getLogger("my_logger")
    new_logger.setLevel(level)

    formatter = CustomFormatter()

    # stdout stream handler if no handler is configured
    if not new_logger.hasHandlers():
        console = logging.StreamHandler(stream=sys.stdout)
        console.setFormatter(formatter)
        console.setLevel(level)
        new_logger.addHandler(console)

    return new_logger


def configure_file_logger(level="INFO"):
    """
    Configures a simple console logger with the givel level.
    A usecase is to change the formatting of the default handler of the root logger
    """
    assert level in [
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ], f"Invalid log level: {level}"

    # Create a logger
    new_logger = logging.getLogger("file_logger")
    new_logger.setLevel("DEBUG")

    file_log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Set the date format
    date_format = "%Y-%m-%d %H:%M:%S"

    formatter = logging.Formatter(file_log_format, date_format)

    # stdout stream handler if no handler is configured
    if not new_logger.hasHandlers():
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Add file handler
        file_handler = logging.FileHandler(
            Path(LOG_DIR, f"{datetime_str}.log"), mode="a"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel("DEBUG")
        new_logger.addHandler(file_handler)

    return new_logger


def remove_color_escape_codes(text):
    """
    Removes ANSI color escape codes from the given string.
    """

    # ANSI escape code pattern (e.g., \x1b[31m for red)
    escape_code_pattern = re.compile(r"\x1b\[[0-9;]*m")

    return escape_code_pattern.sub("", text)


def succes(message):
    """Prints a message to the console in green."""
    message = str(message)
    logger.info(green(message))
    file_logger.info(remove_color_escape_codes(message))


def warning(message, *args, **kwargs):
    """Prints a message with log level warning."""
    message = str(message)
    logger.warning(message, *args, **kwargs)
    file_logger.warning(remove_color_escape_codes(message), *args, **kwargs)


def error(message, *args, **kwargs):
    """Prints a message with log level error."""
    message = str(message)
    logger.error(message, *args, **kwargs)
    file_logger.error(remove_color_escape_codes(message), *args, **kwargs)


def debug(message, *args, **kwargs):
    """Prints a message with log level debug."""
    message = str(message)
    logger.debug(message, *args, **kwargs)
    file_logger.debug(remove_color_escape_codes(message), *args, **kwargs)


def info(message, *args, **kwargs):
    """Prints a message with log level info."""
    message = str(message)
    logger.info(message, *args, **kwargs)
    file_logger.info(remove_color_escape_codes(message), *args, **kwargs)


def critical(message, *args, **kwargs):
    """Prints a message with log level critical."""
    message = str(message)
    logger.critical(message, *args, **kwargs)
    file_logger.critical(message, *args, **kwargs)


def set_level(level):
    """Sets the log level of the logger."""
    logger.setLevel(level)
    file_logger.setLevel(level)


def set_file_logger_directory(directory):
    """Sets the log level of the logger."""
    global LOG_DIR, file_logger
    LOG_DIR = directory
    # Remove all handlers from the file logger
    for handler in file_logger.handlers:
        file_logger.removeHandler(handler)

    # Add file handler
    file_logger = configure_file_logger(level="DEBUG")


logger = configure_console_logger(level="DEBUG")
file_logger = configure_file_logger(level="DEBUG")
