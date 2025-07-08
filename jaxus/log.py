import datetime
import logging
import os
import re
import sys
from pathlib import Path
from rich.console import Console

console = Console(log_time=False)
# The logger to use
logger = None

LOG_DIR = Path("log")


# Check if the environment variable JAXUS_NO_COLORS is set
if os.environ.get("JAXUS_NO_COLORS", "0") == "1":
    print("Disabling colors for logging.")

    # Disable colors
    def red(string):
        """Dummy function to disable colors."""
        return string

    def green(string):
        """Dummy function to disable colors."""
        return string

    def yellow(string):
        """Dummy function to disable colors."""
        return string

    def blue(string):
        """Dummy function to disable colors."""
        return string

    def orange(string):
        """Dummy function to disable colors."""
        return string

else:

    def red(string):
        """Adds ANSI escape codes to print a string in red around the string."""
        return "[red bold]" + str(string) + "[/red bold]"

    def green(string):
        """Adds ANSI escape codes to print a string in green around the string."""
        return "[green bold]" + str(string) + "[/green bold]"

    def yellow(string):
        """Adds ANSI escape codes to print a string in yellow around the string."""
        return "[yellow bold]" + str(string) + "[/yellow bold]"

    def blue(string):
        """Adds ANSI escape codes to print a string in blue around the string."""
        return "[blue bold]" + str(string) + "[/blue bold]"

    def orange(string):
        """Adds ANSI escape codes to print a string in orange around the string."""
        return "[orange bold]" + str(string) + "[/orange bold]"


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
    console.log(green(message))


def warning(message, *args, **kwargs):
    """Prints a message with log level warning."""
    message = yellow("Warning: ") + str(message)
    console.log(message)


def error(message, *args, **kwargs):
    """Prints a message with log level error."""
    message = red("Warning: ") + str(message)
    console.log(message, *args, **kwargs)


def debug(message, *args, **kwargs):
    """Prints a message with log level debug."""
    message = str(message)
    console.log(message, *args, **kwargs)


def info(message, *args, **kwargs):
    """Prints a message with log level info."""
    message = str(message)
    console.log(message, *args, **kwargs)
