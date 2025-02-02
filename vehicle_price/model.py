import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from vehicle_price.logger import Logger
from vehicle_price.config import ARTIFACTS_DIR

class VehiclePriceModel:
    def __init__(self):
        self.logger = Logger("VehiclePriceModel").get_logger()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_path = os.path.join(ARTIFACTS_DIR, "vehicle_price_model.pkl")
        self.encoder_path = os.path.join(ARTIFACTS_DIR, "encoder.pkl")  
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.required_features = None  # Store features for prediction

    def preprocess_data(self, data, fit_encoder=True):
        try:
            self.logger.info("Preprocessing data...")

            # Drop unnecessary columns
            data = data.drop(columns=["name", "description", "exterior_color", "interior_color"], errors="ignore")
            
            # Ensure required columns exist
            required_columns = ["cylinders", "doors", "mileage"]
            for col in required_columns:
             if col not in data.columns:
                self.logger.warning(f"Missing column: {col}. Filling with default values.")
                if col == "cylinders":
                    data[col] = 4  # Default to 4 cylinders
                elif col == "doors":
                    data[col] = 4  # Default to 4 doors
                elif col == "mileage":
                    data[col] = data["mileage"].median()  # Use median mileage
                    
            # Fill missing values
            data["cylinders"].fillna(data["cylinders"].median(), inplace=True)
            data["mileage"].fillna(data["mileage"].median(), inplace=True)
            data["doors"].fillna(data["doors"].mode()[0], inplace=True)

            # Separate categorical and numerical columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

            if fit_encoder:
                # Fit and transform during training
                encoded_features = self.encoder.fit_transform(data[categorical_cols])
                feature_names = self.encoder.get_feature_names_out(categorical_cols)

                # Save the encoder for future use
                with open(self.encoder_path, "wb") as f:
                    pickle.dump(self.encoder, f)

            else:
                # Load encoder during prediction
                with open(self.encoder_path, "rb") as f:
                    self.encoder = pickle.load(f)
                encoded_features = self.encoder.transform(data[categorical_cols])
                feature_names = self.encoder.get_feature_names_out(categorical_cols)

            # Convert encoded categorical data into a DataFrame
            encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=data.index)

            # Concatenate categorical and numerical data
            processed_data = pd.concat([data[numerical_cols], encoded_df], axis=1)

            # Store the required features during training
            if fit_encoder:
                self.required_features = processed_data.columns
            else:
                # Ensure all required features exist during prediction
                missing_cols = set(self.required_features) - set(processed_data.columns)
                for col in missing_cols:
                    processed_data[col] = 0  # Add missing columns with default value 0

                # Ensure correct column order
                processed_data = processed_data[self.required_features]

            self.logger.info("Data preprocessing complete.")
            return processed_data

        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            return None

    def train(self, data):
        try:
            self.logger.info("Starting model training...")

            # Preprocess the data
            processed_data = self.preprocess_data(data, fit_encoder=True)

            # Selecting features and target variable
            X = processed_data.drop(columns=["price"])
            y = processed_data["price"]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            self.model.fit(X_train, y_train)

            # Save the trained model
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)

            self.logger.info(f"Model trained and saved at {self.model_path}")

            # Evaluate the model
            self.evaluate(X_test, y_test)

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")

    def evaluate(self, X_test, y_test):
        try:
            self.logger.info("Evaluating the model...")

            predictions = self.model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)

            self.logger.info(f"Model Evaluation:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR² Score: {r2:.4f}")

        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")

    def predict(self, input_data):
        try:
            if not os.path.exists(self.model_path):
                self.logger.error("No trained model found. Train the model first.")
                return None

            # Load the trained model
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)

            # Convert input_data to DataFrame
            input_df = pd.DataFrame([input_data])

            # Apply the same preprocessing steps as training
            input_processed = self.preprocess_data(input_df, fit_encoder=False)

            # Make prediction
            prediction = model.predict(input_processed)[0]
            return prediction

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return None
