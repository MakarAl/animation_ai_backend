"""
Logging utilities for Animation AI Backend.
Provides timestamped logging functions for consistent output formatting.
"""

import logging
from datetime import datetime
from typing import Optional
import sys


def setup_logger(name: str = "AnimationAI", level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger


def log(msg: str, level: str = "INFO", logger: Optional[logging.Logger] = None) -> None:
    """
    Log a message with timestamp.
    
    Args:
        msg: Message to log
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        logger: Logger instance (if None, uses default)
    """
    if logger is None:
        logger = setup_logger()
    
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    logger.log(log_level, msg)


def log_step(step_name: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a processing step with clear formatting.
    
    Args:
        step_name: Name of the current step
        logger: Logger instance (if None, uses default)
    """
    if logger is None:
        logger = setup_logger()
    
    logger.info(f"🔄 {step_name}")


def log_success(msg: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a success message.
    
    Args:
        msg: Success message
        logger: Logger instance (if None, uses default)
    """
    if logger is None:
        logger = setup_logger()
    
    logger.info(f"✅ {msg}")


def log_error(msg: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log an error message.
    
    Args:
        msg: Error message
        logger: Logger instance (if None, uses default)
    """
    if logger is None:
        logger = setup_logger()
    
    logger.error(f"❌ {msg}")


def log_warning(msg: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log a warning message.
    
    Args:
        msg: Warning message
        logger: Logger instance (if None, uses default)
    """
    if logger is None:
        logger = setup_logger()
    
    logger.warning(f"⚠️  {msg}")


def log_info(msg: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Log an info message.
    
    Args:
        msg: Info message
        logger: Logger instance (if None, uses default)
    """
    if logger is None:
        logger = setup_logger()
    
    logger.info(f"ℹ️  {msg}")


def print_timestamped(msg: str) -> None:
    """
    Print a message with timestamp (simple version without logger setup).
    
    Args:
        msg: Message to print
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def print_step(step_name: str) -> None:
    """
    Print a processing step with clear formatting.
    
    Args:
        step_name: Name of the current step
    """
    print_timestamped(f"🔄 {step_name}")


def print_success(msg: str) -> None:
    """
    Print a success message.
    
    Args:
        msg: Success message
    """
    print_timestamped(f"✅ {msg}")


def print_error(msg: str) -> None:
    """
    Print an error message.
    
    Args:
        msg: Error message
    """
    print_timestamped(f"❌ {msg}") 