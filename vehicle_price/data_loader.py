import pandas as pd
from vehicle_price.logger import Logger

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.logger = Logger("DataLoader").get_logger()

    def load_data(self):
        try:
            self.logger.info("Loading dataset...")
            data = pd.read_csv(self.file_path)
            self.logger.info(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return None
