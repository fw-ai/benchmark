import logging

RESET = "\033[0m"

BRIGHT_SKY = "\033[38;5;81m"
BRIGHT_CYAN = "\033[38;5;51m"
BRIGHT_TEAL = "\033[38;5;50m"
BRIGHT_GREEN = "\033[38;5;120m"
BRIGHT_BLUE = "\033[38;5;111m"
BRIGHT_PURPLE = "\033[38;5;183m"
BRIGHT_GOLD = "\033[38;5;228m"
BRIGHT_GRAY = "\033[38;5;250m"


class Formatter(logging.Formatter):
    def __init__(self, *, color: str, fmt: str, datefmt: str | None) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        return f"{self.color}{super().format(record)}{RESET}"


def create_handler(*, color: str, fmt: str, datefmt: str | None) -> logging.Handler:
    handler = logging.StreamHandler()
    handler.setFormatter(Formatter(color=color, fmt=fmt, datefmt=datefmt))
    return handler
