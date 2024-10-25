import logging
import os
from datetime import datetime

def setup_logger(log_dir="logs", log_file="log.txt", level=logging.INFO):
    """Set up a logger that logs messages to both a file and the console.

    Args:
        log_dir (str): Directory where log files will be saved. Defaults to "logs".
        log_file (str): Name of the log file. Defaults to "log.txt".
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.
    
    returns:
        logger (logging.Logger): Configured logger instance.
    """
    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Define log file path with timestamp
    log_file_path = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{log_file}")
    
    # Set up the logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove previous handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# Example usage
if __name__ == "__main__":
    logger = setup_logger()
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")