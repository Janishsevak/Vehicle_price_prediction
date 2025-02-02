import pandas as pd
from vehicle_price.logger import Logger

class DataCleaner:
    def __init__(self, data):
        self.data = data
        self.logger = Logger("DataCleaner").get_logger()

    def clean_data(self):
        try:
            self.logger.info("Cleaning dataset...")
            # Handle missing values
            self.data['price'].fillna(self.data['price'].median(), inplace=True)
            self.data.dropna(subset=['mileage', 'cylinders', 'fuel'], inplace=True)

            # Remove duplicates
            duplicates = self.data.duplicated().sum()
            if duplicates > 0:
                self.data.drop_duplicates(inplace=True)
                self.logger.info(f"Removed {duplicates} duplicate rows.")

            # Filter out invalid prices
            self.data = self.data[self.data['price'] > 0]
            self.logger.info("Data cleaning complete.")
            return self.data
        except Exception as e:
            self.logger.error(f"Data cleaning failed: {e}")
            return None
