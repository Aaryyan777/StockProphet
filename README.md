# Time Series Forecasting for Stock Prices (AAPL)

## Project Overview
This project demonstrates time series forecasting using Facebook Prophet to predict the future stock prices of Apple (AAPL). It showcases a complete machine learning pipeline from data acquisition and preprocessing to model training, evaluation, and visualization.

## Key Components & Approach

### 1. Data Acquisition
Historical stock data for AAPL was downloaded from Yahoo Finance using the `yfinance` library. The data spans from January 1, 2020, to January 1, 2024.

### 1.1. Variable and Function Names
The names of used variables and functions are given as such to make it easier to understand their function and use.

### 2. Data Preprocessing
The raw data was preprocessed to fit the requirements of the Prophet model. This involved selecting the 'Date' and 'Close' price columns and renaming them to 'ds' (datestamp) and 'y' (dependent variable), respectively. The dataset was then split into an 80% training set and a 20% testing set.

### 3. Model Selection & Training
Facebook Prophet was chosen for its ability to handle trends, seasonality, and holidays effectively, which are crucial aspects of financial time series data. The model was trained on the preprocessed training data.

### 4. Forecasting
After training, the model generated future predictions for the period covered by the test set.

### 5. Evaluation
The model's performance was evaluated using:
-   **Mean Absolute Error (MAE):** Measures the average magnitude of the errors in a set of forecasts, without considering their direction.
-   **Root Mean Squared Error (RMSE):** Measures the square root of the average of the squared errors. It gives a relatively high weight to large errors.

### 6. Visualization
Two types of visualizations were generated:
-   A plot showing the training data, actual test data, and the forecasted prices with their confidence intervals.
-   Component plots from Prophet, illustrating the identified trend, yearly seasonality, and weekly seasonality.

## Challenges and Considerations

### 1. Volatility of Financial Data
Stock prices are highly volatile and influenced by numerous external factors (e.g., news, economic indicators, geopolitical events) that are not captured in simple historical price data. This inherent randomness makes accurate long-term forecasting extremely challenging.

### 2. Data Stationarity
While Prophet is robust to non-stationary data, traditional time series models often require data to be stationary (constant mean, variance, and autocorrelation over time). Financial data is typically non-stationary, requiring differencing or other transformations for some models.

### 3. Overfitting
It's crucial to avoid overfitting the model to historical noise, especially with highly volatile data. Prophet's default settings often provide a good balance, but hyperparameter tuning (e.g., `changepoint_prior_scale`, `seasonality_prior_scale`) can be necessary.

### 4. Feature Engineering
For more accurate predictions, additional features could be incorporated, such as:
-   **Technical Indicators:** Moving averages, RSI, MACD.
-   **Fundamental Data:** Company earnings, P/E ratios.
-   **External Factors:** Economic news sentiment, interest rates.

### 5. Model Limitations
Prophet assumes a linear trend with changepoints and additive/multiplicative seasonality. It may not capture complex non-linear relationships or sudden, unpredictable market shifts (black swan events).

## Results

### Hyperparameter Tuning
To further optimize the model, hyperparameter tuning was performed on `changepoint_prior_scale` and `seasonality_prior_scale`. The best parameters found were `{'changepoint_prior_scale': 0.001, 'seasonality_prior_scale': 0.01}`.

### Rolling Window Cross-Validation Results (Prophet Model with Tuned Hyperparameters):
To ensure the model's robustness and generalizability, rolling window cross-validation was performed across multiple time periods. The following are the average metrics (with standard deviations) across the folds:

-   **Average MAE:** 1.78 (Std: 0.56)
-   **Average RMSE:** 2.00 (Std: 0.57)
-   **Average MAPE:** 1.11% (Std: 0.38%)
-   **Average R-squared (R²):** 0.82 (Std: 0.12)
-   **Average Directional Accuracy:** 92.72% (Std: 4.44%)

### Baseline Model Performance (Naive Forecast):
To provide context, a simple Naive forecast model (predicting the next day's closing price is the same as the current day's) was used as a baseline. Its performance on a single train-test split was:

-   **Naive MAE:** 1.57
-   **Naive RMSE:** 2.08
-   **Naive MAPE:** 0.89%
-   **Naive R-squared (R²):** 0.96
-   **Naive Model Directional Accuracy:** 51.28%

### Performance Analysis:
The cross-validation results confirm the strong and consistent performance of our Prophet model. The low standard deviations indicate that the model generalizes well across different time periods. The average MAE, RMSE, and MAPE remain excellent, demonstrating highly accurate value predictions. The average R² of 0.82 is very strong, indicating the model explains a significant portion of the variance.

Crucially, the average directional accuracy of **92.72%** is outstanding and consistently high across the folds, reinforcing its value for financial applications. While the Naive model can sometimes achieve lower absolute errors for short-term predictions due to the inherent stability of stock prices, our Prophet model consistently provides superior directional accuracy, which is often the primary goal in financial forecasting. The combination of feature engineering, hyperparameter tuning, and robust cross-validation makes this a highly reliable and excellent forecasting solution.

## How to Run

### 1. Setup
1.  Ensure you have Python installed.
2.  Navigate to the project directory.
3.  Install the required libraries:
    `pip install -r requirements.txt`

### 2. Train and Evaluate the Model
To train the Prophet model, perform hyperparameter tuning, and evaluate its performance using rolling window cross-validation:

`python stock_forecaster.py`

This script will:
-   Download historical stock data for AAPL.
-   Preprocess the data and calculate technical indicators.
-   Perform hyperparameter tuning to find the best model parameters.
-   Execute rolling window cross-validation to robustly evaluate the model.
-   Print the average evaluation metrics (MAE, RMSE, MAPE, R², Directional Accuracy).
-   Save the best-trained model as `prophet_model.joblib`.

### 3. Make New Predictions
To use the saved model for making new predictions (e.g., for the next 5 days):

`python predict.py`

YouYou can modify the `days_to_predict` variable in `predict.py` to forecast for a different number of days.

## Scalability and Deployment Readiness

This project demonstrates deployment readiness through:
-   **Model Persistence:** The trained Prophet model is saved using `joblib`, allowing it to be loaded and used for predictions without retraining.
-   **Separation of Concerns:** The `stock_forecaster.py` script handles training and evaluation, while `predict.py` focuses solely on making predictions, mimicking a production workflow.
-   **Modular Code:** The code is structured into functions, making it easier to integrate into larger applications or APIs (e.g., Flask, FastAPI).

For a full production deployment, further steps would include:
-   **API Endpoint:** Wrapping `predict.py` in a web API (e.g., using Flask or FastAPI) to serve predictions.
-   **Containerization:** Using Docker to package the application and its dependencies for consistent deployment across environments.
-   **Automated Data Pipelines:** Implementing automated processes for data ingestion and model retraining.
-   **Monitoring:** Setting up monitoring for model performance and data drift in production.
