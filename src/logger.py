# logger.py
# This file contains the logger configuration for the project.

import logging
from pathlib import Path
import sys
from tqdm import tqdm

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def get_logger(name: str) -> logging.Logger:
    """
    Creates a logger with the specified name and configuration.
    Args:
        name: Name of the logger (typically __name__)
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure handlers if they haven't been added yet
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create formatters with colors and better formatting
        console_formatter = logging.Formatter(
            '\033[2m%(asctime)s\033[0m | '  # Dim timestamp
            '%(levelname)-8s | '             # Aligned level name
            '%(name)s | '                    # Module name
            '%(message)s'                    # The message
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler using TqdmLoggingHandler
        console_handler = TqdmLoggingHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger
