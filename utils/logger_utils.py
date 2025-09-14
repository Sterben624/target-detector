import logging
import sys
from pathlib import Path
from datetime import datetime

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        # Create a copy of the record to avoid modifying the original
        import copy
        colored_record = copy.copy(record)
        
        # Add color to levelname
        if colored_record.levelname in self.COLORS:
            colored_record.levelname = f"{self.COLORS[colored_record.levelname]}{self.BOLD}{colored_record.levelname:<8}{self.RESET}"
        
        # Add color to logger name
        colored_record.name = f"\033[94m{colored_record.name}\033[0m"
        
        return super().format(colored_record)

def setup_logger(
    name: str = "AudioApp",
    level: str = "INFO",
    log_file: str = None,
    console_output: bool = True,
    file_level: str = None,
    console_level: str = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 3
):
    """
    Setup a beautiful logger with console and file output
    
    Args:
        name: Logger name
        level: Default log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (auto-generated if None)
        console_output: Enable console logging
        file_level: File-specific log level (uses 'level' if None)
        console_level: Console-specific log level (uses 'level' if None)
        max_bytes: Max size of log file before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        logging.Logger: Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set to lowest level, handlers will filter
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Convert string levels to logging constants
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    base_level = level_map.get(level.upper(), logging.INFO)
    file_log_level = level_map.get((file_level or level).upper(), base_level)
    console_log_level = level_map.get((console_level or level).upper(), base_level)
    
    # Create formatters
    console_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    file_format = "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s | %(message)s"
    
    # Colored formatter only for console
    console_formatter = ColoredFormatter(
        console_format,
        datefmt="%H:%M:%S"
    )
    
    # Plain formatter for file (no colors)
    file_formatter = logging.Formatter(
        file_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file or not console_output:  # Always create file handler if console is disabled
        if not log_file:
            # Auto-generate log file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"logs/{name.lower()}_{timestamp}.log"
        
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True)
        
        # Use RotatingFileHandler to prevent huge log files
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        if console_output:
            logger.info(f"📄 Log file: {log_file}")
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Get existing logger or create a default one
    
    Args:
        name: Logger name (uses root logger if None)
    
    Returns:
        logging.Logger: Logger instance
    """
    if name:
        return logging.getLogger(name)
    else:
        # Return root logger or create default if none exists
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            return setup_logger("DefaultLogger")
        return root_logger

# Convenience functions
def debug(msg: str, logger_name: str = None):
    """Quick debug log"""
    get_logger(logger_name).debug(msg)

def info(msg: str, logger_name: str = None):
    """Quick info log"""
    get_logger(logger_name).info(msg)

def warning(msg: str, logger_name: str = None):
    """Quick warning log"""
    get_logger(logger_name).warning(msg)

def error(msg: str, logger_name: str = None):
    """Quick error log"""
    get_logger(logger_name).error(msg)

def critical(msg: str, logger_name: str = None):
    """Quick critical log"""
    get_logger(logger_name).critical(msg)

# Example usage and testing
if __name__ == "__main__":
    # Test the logger
    print("🧪 Testing logger functionality...")
    
    # Create logger with default settings
    logger = setup_logger(
        name="TestLogger",
        level="DEBUG",
        console_output=True,
        log_file="test_app.log"
    )
    
    # Test different log levels
    logger.debug("🐛 This is a debug message")
    logger.info("ℹ️  This is an info message")
    logger.warning("⚠️  This is a warning message")
    logger.error("❌ This is an error message")
    logger.critical("💀 This is a critical message")
    
    # Test convenience functions
    info("📝 Using convenience function", "TestLogger")
    
    # Test logger with different settings
    audio_logger = setup_logger(
        name="AudioReceiver",
        level="INFO",
        console_level="INFO",
        file_level="DEBUG",
        log_file="logs/audio_app.log"
    )
    
    audio_logger.info("🎵 Audio logger initialized")
    audio_logger.debug("🔧 Debug info will only appear in file")
    
    print("✅ Logger test completed!")