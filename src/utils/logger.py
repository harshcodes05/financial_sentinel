import sys
import logging
from src.config import settings

def setup_logger(name: str = "financial_sentinel") -> logging.Logger:
    """Configures structured logging for application modules."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if settings.ENV == "production" else logging.DEBUG)

    # Avoid duplicate log handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

logger = setup_logger()
