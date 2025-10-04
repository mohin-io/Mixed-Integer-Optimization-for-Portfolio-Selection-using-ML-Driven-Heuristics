"""
Professional Logging Configuration

Provides structured logging with file rotation and console output.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_dir: str = "logs",
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Configure and return a logger with file rotation and console output.

    Args:
        name: Logger name (usually __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
        console_output: Whether to output to console
        file_output: Whether to output to file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Portfolio optimization started")
        >>> logger.error("Optimization failed", exc_info=True)
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        fmt='%(levelname)s - %(message)s'
    )

    # File handler with rotation
    if file_output:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = log_path / f"{name.replace('.', '_')}_{timestamp}.log"

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # Use simple format for console, detailed for file
        if not file_output:
            console_handler.setFormatter(detailed_formatter)
        else:
            console_handler.setFormatter(simple_formatter)

        logger.addHandler(console_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger doesn't have handlers, set it up
    if not logger.handlers:
        return setup_logger(name)

    return logger


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.

    Usage:
        class MyOptimizer(LoggerMixin):
            def optimize(self):
                self.logger.info("Starting optimization")
                # ... optimization code
                self.logger.debug("Optimization details", extra={'metrics': {...}})
    """

    @property
    def logger(self) -> logging.Logger:
        """Return logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__module__)
        return self._logger


# Example usage
if __name__ == '__main__':
    # Create logger
    logger = setup_logger(__name__)

    # Log at different levels
    logger.debug("This is a debug message")
    logger.info("Portfolio optimization started")
    logger.warning("High volatility detected")
    logger.error("Optimization failed")

    # Log with extra context
    logger.info(
        "Optimization completed",
        extra={
            'sharpe_ratio': 1.85,
            'return': 0.12,
            'volatility': 0.15
        }
    )

    print(f"Logs written to: logs/")
