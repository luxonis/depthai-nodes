import logging
import os
from enum import Enum
from typing import Optional


class LogLevel(Enum):
    CRITICAL = "CRITICAL"
    DEBUG = "DEBUG"
    ERR = "ERR"
    INFO = "INFO"
    WARN = "WARN"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("depthai-nodes")
    if name:
        logger = logger.getChild(name)
    return logger


def setup_logging(level: Optional[str] = None, file: Optional[str] = None):
    """Globally configures logging for depthai_nodes package.

    @type level: str or None
    @param level: Logging level. One of "CRITICAL", "DEBUG", "ERR", "INFO", and "WARN".
        Can be changed using "DEPTHAI_NODES_LEVEL" env variable. If not set defaults to
        "DEPTHAI_LEVEL" if set or "WARN".
    @type file: str or None
    @param file: Path to a file where logs will be saved. If None, logs will not be
        saved. Defaults to None.
    """
    logger = get_logger()
    passed_level = get_log_level(level)
    env_dai_nodes_level = get_log_level(os.environ.get("DEPTHAI_NODES_LEVEL", None))
    env_dai_level = get_log_level(os.environ.get("DEPTHAI_LEVEL", None))

    used_level = passed_level or env_dai_nodes_level or env_dai_level
    if not used_level:
        used_level = LogLevel.WARN

    format = (
        file_format
    ) = "%(asctime)s [depthai-nodes] [%(levelname)s] [%(name)s] %(message)s"
    datefmt = "[%Y-%m-%d %H:%M:%S]"

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format, datefmt=datefmt))
    logger.addHandler(console_handler)

    if file is not None:
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(logging.Formatter(file_format, datefmt=datefmt))
        logger.addHandler(file_handler)

    logger.setLevel(used_level.value)
    logger.info(f"Using log level: {used_level}")


def get_log_level(level_str: Optional[str]) -> Optional[LogLevel]:
    try:
        if level_str is None:
            return None

        level_str = level_str.upper()  # type: ignore
        # mapping some DepthAI specific levels to python logger levels
        if level_str in "OFF":
            level_str = "WARN"
        elif level_str == "TRACE":
            level_str = "INFO"
        return LogLevel(level_str)
    except ValueError as e:
        raise e
