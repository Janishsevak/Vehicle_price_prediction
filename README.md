## Vehicle Price Prediction Project

Objective The primary objective of this project is to develop a machine learning-based solution to predict vehicle prices based on various features such as make, model, year, mileage, and engine specifications. By accurately predicting vehicle prices, this system can assist buyers, sellers, and dealerships in making informed pricing decisions.

## Dataset Description
•	Source: Provided dataset containing vehicle specifications and pricing details.
•	Rows: 1,002
•	Columns: 17 (16 features + 1 target variable)
•	Target Variable: Price (continuous variable representing the vehicle price in USD)
## Sample Features:
•	Name: Full vehicle name (make, model, trim)
•	Make: Manufacturer (e.g., Ford, Toyota, BMW)
•	Model: Model name
•	Year: Manufacturing year
•	Mileage: Distance covered by the vehicle
•	Fuel: Fuel type (Gasoline, Diesel, Electric)
•	Engine: Engine specifications
•	Cylinders: Number of engine cylinders
•	Transmission: Type of transmission (Automatic, Manual)
•	Body: Body style (SUV, Sedan, Pickup Truck)
•	Drivetrain: Drivetrain type (FWD, AWD, RWD)
## Methodology
1.	Data Preprocessing
o	Imputed missing values using the median for numerical features and the mode for categorical features.
o	Applied one-hot encoding to categorical variables.
o	Normalized numerical features to ensure consistency in model training.
o	Removed unrealistic values such as 0-price entries.

2.	Feature Engineering
o	Extracted meaningful features from the name column (e.g., trim level, brand impact on pricing).
o	Converted categorical features into numeric form using one-hot encoding.
o	Handled missing categorical values by introducing an 'Unknown' category where necessary.

3.	Model Training
o	Three machine learning models were implemented and evaluated: 
	Linear Regression (Baseline Model)
	Random Forest Regressor (Tree-based model for better performance on structured data)
	XGBoost Regressor (Boosting technique for optimized predictions)
o	The dataset was split into an 80-20 train-test ratio. Models were evaluated on test data using regression metrics.

## Results
The evaluation metrics for the models are as follows:
Metric	Linear Regression	Random Forest	XGBoost
MAE	214,557,243,329.82	4,627.09	4,676.42
MSE	455,216,423,086,776,278,581,248.00	63,372,259.20	57,189,971.93
RMSE	674,697,282,554.76	7,960.67	7,562.41
R² Score	-392,627,383,313,861,313,560,576.0000	0.8073	0.8261
Best Model: XGBoost Regressor

## Challenges & Solutions
•	Feature Importance: Some features (e.g., engine type, trim) had missing values. Solution: Used statistical imputation and encoding techniques to retain their impact.
•	Data Imbalance: Some price ranges were underrepresented. Solution: Weighted training to avoid model bias toward common price ranges.
•	High Variance in Pricing: Due to variations in make, model, and condition. Solution: Applied feature interactions to capture complex relationships.

## Conclusion                                                                                                     

The vehicle price prediction model achieved an R² score of 0.8261 using the XGBoost Regressor, demonstrating its effectiveness in predicting vehicle prices. The model accurately captures the impact of various features like make, model, year, mileage, and engine specifications on pricing. Future work will focus on improving generalization, deploying the model, and adding explainability for better decision-making.

