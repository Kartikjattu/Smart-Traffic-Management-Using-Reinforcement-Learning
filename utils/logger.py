import logging
import sys
from datetime import datetime

def setup_logger(log_file=None):
    """Setup logger with console and file handlers"""
    logger = logging.getLogger('TrafficRL')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is None:
        log_file = f'logs/traffic_rl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger