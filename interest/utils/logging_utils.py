import logging
from typing import Optional


def setup_logging(
    log_file: Optional[str] = "logs/app.log",
    level: int = logging.INFO
) -> None:
    """
    Set up logging configuration.

    Args:
        log_file (Optional[str]): Path to the log file.
          Defaults to "logs/app.log".
        level (int): Logging level. Defaults to logging.INFO.
    """
    logging.basicConfig(
        filename=log_file,
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)
