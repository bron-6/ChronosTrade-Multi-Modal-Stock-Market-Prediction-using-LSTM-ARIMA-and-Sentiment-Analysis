# ChronosTrade-Multi-Modal-Stock-Market-Prediction-using-LSTM-ARIMA-and-Sentiment-Analysis

# Time Series Forecasting: ARIMA and LSTM

This repository contains coursework for **Time Series Forecasting**, focusing on the implementation and comparison of **ARIMA** (statistical) and **LSTM** (deep learning) models for stock price prediction.

The work is organized into two weekly assignments, each with a Jupyter Notebook and a written report in LaTeX.

---


---

## Week 1: ARIMA Model for Time Series Forecasting

### Objective
To understand and implement the **ARIMA** model for time series forecasting, including data preprocessing, stationarity testing, model selection, forecasting, and diagnostic analysis.

### Key Concepts Covered
- Time series stationarity
- Differencing to remove trends
- Augmented Dickey–Fuller (ADF) test
- ACF and PACF plots for parameter selection
- ARIMA model components: AR, I, MA
- Model evaluation (MAE, MSE, RMSE)
- Residual diagnostics and walk-forward validation

### Files
- **`ARIMA_Assignment_Week1.ipynb`**
  - Complete ARIMA workflow:
    - Data collection from Yahoo Finance
    - Resampling and missing value handling
    - Stationarity testing and differencing
    - ACF/PACF analysis
    - Grid search for optimal (p, d, q)
    - Forecasting and evaluation
    - Residual analysis and diagnostics
- **`week1_report.pdf`**
  - Two-page LaTeX report explaining:
    - ARIMA theory and assumptions
    - Implementation details
    - Interpretation of plots and results
    - Challenges, limitations, and improvements

---

## Week 2: ARIMA vs LSTM for Stock Price Prediction

### Objective
To build an **LSTM model** for stock price prediction, compare it with **ARIMA**, and analyze the strengths and weaknesses of statistical versus deep learning approaches.

### Key Concepts Covered
- Differences between ARIMA and LSTM models
- Stationarity and differencing (ARIMA)
- ACF/PACF-based parameter selection
- Sliding window technique for supervised learning
- Data normalization for neural networks
- LSTM architecture and training
- Model comparison using error metrics

### Files
- **`ARIMA_vs_LSTM_Assignment_Week2.ipynb`**
  - End-to-end implementation:
    - 3–5 years of stock price data (Yahoo Finance)
    - Data preprocessing and normalization
    - ARIMA modeling and forecasting
    - LSTM model (stacked LSTM + dropout)
    - Forecast comparison and evaluation
    - Learning curves and visual diagnostics
- **`week2_report.pdf`**
  - Two-page LaTeX report covering:
    - ARIMA and LSTM fundamentals
    - Statistical vs deep learning approaches
    - Model performance comparison
    - Visualization interpretation
    - Conclusions and future extensions

---

## Tools and Libraries Used
- Python
- NumPy, Pandas
- Matplotlib
- Statsmodels
- Scikit-learn
- TensorFlow / Keras
- yfinance
- LaTeX (for reports)

---

## Key Learnings
- ARIMA is interpretable, fast, and effective for linear stationary time series.
- LSTM captures non-linear and long-term dependencies but requires more data and tuning.
- Proper preprocessing (stationarity, scaling, windowing) is critical for reliable forecasts.
- Model evaluation should be performed using both metrics and diagnostic plots.
- Walk-forward and validation techniques improve robustness of conclusions.

---

## Possible Extensions
- Seasonal models (SARIMA / SARIMAX)
- Additional features (volume, technical indicators)
- Bidirectional or attention-based LSTM models
- Hybrid ARIMA–LSTM approaches
- Probabilistic forecasting and uncertainty estimation

---
