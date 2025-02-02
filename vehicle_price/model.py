import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from vehicle_price.logger import Logger
from vehicle_price.config import ARTIFACTS_DIR


class VehiclePriceModel:
    def __init__(self):
        self.logger = Logger("VehiclePriceModel").get_logger()
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        }
        self.best_model = None
        self.model_path = os.path.join(ARTIFACTS_DIR, "vehicle_price_model.pkl")
        self.encoder_path = os.path.join(ARTIFACTS_DIR, "encoder.pkl")
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.required_features = None

    def preprocess_data(self, data, fit_encoder=True, scale=False):
        try:
            self.logger.info("Preprocessing data...")
            data = data.drop(columns=["name", "description", "exterior_color", "interior_color"], errors="ignore")

            required_columns = ["cylinders", "doors", "mileage"]
            for col in required_columns:
                if col not in data.columns:
                    self.logger.warning(f"Missing column: {col}. Filling with default values.")
                    data[col] = 4 if col in ["cylinders", "doors"] else data["mileage"].median()

            data.fillna({"cylinders": 4, "mileage": data["mileage"].median(), "doors": 4}, inplace=True)

            categorical_cols = data.select_dtypes(include=['object']).columns
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

            if fit_encoder:
                encoded_features = self.encoder.fit_transform(data[categorical_cols])
                feature_names = [name.replace("[", "").replace("]", "").replace("<", "").replace(">", "").replace(" ", "_")
                                 for name in self.encoder.get_feature_names_out(categorical_cols)]
                with open(self.encoder_path, "wb") as f:
                    pickle.dump(self.encoder, f)
            else:
                with open(self.encoder_path, "rb") as f:
                    self.encoder = pickle.load(f)
                encoded_features = self.encoder.transform(data[categorical_cols])
                feature_names = [name.replace("[", "").replace("]", "").replace("<", "").replace(">", "").replace(" ", "_")
                                 for name in self.encoder.get_feature_names_out(categorical_cols)]

            encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=data.index)
            processed_data = pd.concat([data[numerical_cols], encoded_df], axis=1)

            if scale:
                scaler = StandardScaler()
                processed_data[numerical_cols] = scaler.fit_transform(processed_data[numerical_cols])

            if fit_encoder:
                self.required_features = processed_data.columns.tolist()
            else:
                missing_cols = set(self.required_features) - set(processed_data.columns)
                for col in missing_cols:
                    processed_data[col] = 0
                processed_data = processed_data[self.required_features]

            self.logger.info("Data preprocessing complete.")
            return processed_data

        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            return None

    def train(self, data):
        try:
            self.logger.info("Starting model training...")
            best_r2 = -1

            for model_name, model in self.models.items():
                scale = model_name == "Linear Regression"
                processed_data = self.preprocess_data(data, fit_encoder=True, scale=scale)

                X = processed_data.drop(columns=["price"])
                y = processed_data["price"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                self.logger.info(f"Training {model_name}...")
                model.fit(X_train, y_train)

                mae, mse, rmse, r2 = self.evaluate(model, X_test, y_test, model_name)

                if r2 > best_r2:
                    best_r2 = r2
                    self.best_model = model
                    best_model_name = model_name

            with open(self.model_path, "wb") as f:
                pickle.dump(self.best_model, f)

            self.logger.info(f"Best Model: {best_model_name} saved at {self.model_path}")

        except Exception as e:
            self.logger.error(f"Model training failed: {e}")

    def evaluate(self, model, X_test, y_test, model_name):
        try:
            self.logger.info(f"Evaluating {model_name}...")
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            self.logger.info(f"{model_name} Evaluation:\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR² Score: {r2:.4f}")
            return mae, mse, rmse, r2
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {e}")
            return None, None, None, None

    def predict(self, input_data):
        try:
            if not os.path.exists(self.model_path):
                self.logger.error("No trained model found. Train the model first.")
                return None
            with open(self.model_path, "rb") as f:
                model = pickle.load(f)
            input_df = pd.DataFrame([input_data])
            input_processed = self.preprocess_data(input_df, fit_encoder=False)
            prediction = model.predict(input_processed)[0]
            return prediction
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return None
