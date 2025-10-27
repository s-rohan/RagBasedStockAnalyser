import logging
import sys

def setup_logging(
    log_to_file: bool = False,
    filename: str = 'app.log',
    level: int = logging.INFO,
    logger_name: str = "Root"
) -> logging.Logger:
    logger = logging.getLogger(logger_name)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(level)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')

    if log_to_file:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Example usage (import and call setup_logging at the start of your main entry point)
# setup_logging(log_to_file=False, level=logging.INFO)
