import logging
import os
from vehicle_price.config import LOG_FILE  # Import the log file path

class Logger:
    def __init__(self, name):
        self.name = name

    def get_logger(self):
        # Ensure the artifacts directory exists
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(LOG_FILE)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger
