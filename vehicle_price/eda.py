import matplotlib.pyplot as plt
import seaborn as sns
from vehicle_price.logger import Logger

class EDA:
    def __init__(self, data):
        self.data = data
        self.logger = Logger("EDA").get_logger()

    def perform_eda(self):
        try:
            self.logger.info("Performing EDA...")

            # Distribution of price
            plt.figure(figsize=(12, 6))
            sns.histplot(self.data['price'], kde=True, bins=30, color='blue')
            plt.title("Distribution of Vehicle Prices")
            plt.xlabel("Price (USD)")
            plt.ylabel("Frequency")
            plt.show()

            # Boxplot for price
            plt.figure(figsize=(12, 6))
            sns.boxplot(self.data['price'], color='green')
            plt.title("Boxplot of Vehicle Prices")
            plt.xlabel("Price (USD)")
            plt.show()

            # Correlation heatmap
            numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
            correlation_matrix = self.data[numeric_columns].corr()

            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
            plt.title("Correlation Heatmap")
            plt.show()

            # Price by fuel type
            plt.figure(figsize=(14, 6))
            sns.boxplot(data=self.data, x="fuel", y="price", palette="Set2")
            plt.title("Price Distribution by Fuel Type")
            plt.xlabel("Fuel Type")
            plt.ylabel("Price (USD)")
            plt.show()

            self.logger.info("EDA complete.")
        except Exception as e:
            self.logger.error(f"EDA failed: {e}")
