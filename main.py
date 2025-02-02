import sys
import os
from vehicle_price.data_loader import DataLoader
from vehicle_price.data_cleaner import DataCleaner
from vehicle_price.eda import EDA
from vehicle_price.model import VehiclePriceModel
from vehicle_price.logger import Logger

def main():
    logger = Logger("main").get_logger()

    try:
        logger.info("Starting Vehicle Price Analysis Project...")

        # Load the data
        data_loader = DataLoader("Data/dataset.csv")
        data = data_loader.load_data()

        # Clean the data
        data_cleaner = DataCleaner(data)
        cleaned_data = data_cleaner.clean_data()

        # Perform EDA
        eda = EDA(cleaned_data)
        eda.perform_eda()

        # Train and evaluate model
        model = VehiclePriceModel()
        model.train(cleaned_data)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
