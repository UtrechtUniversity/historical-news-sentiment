import logging
import os
import sys
from typing import Optional


def setup_logging(
    log_dir: str = "logs",
    level: int = logging.INFO
) -> None:
    """
    Set up logging configuration for each script.

    Args:
        log_dir (str): Directory where logs will be saved. Defaults to "logs".
        level (int): Logging level. Defaults to logging.INFO.
    """

    # Ensure the logs directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Get the name of the running script
    script_name = os.path.basename(sys.argv[0])
    log_file = os.path.join(log_dir, f"{script_name}.log")

    logging.basicConfig(
        filename=log_file,
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Log to console as well
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)
