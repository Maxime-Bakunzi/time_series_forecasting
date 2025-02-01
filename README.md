# Air Quality Forecasting Project

## Description
This project aims to forecast air quality using time series data. The notebook includes data loading, preprocessing, model training, and evaluation of various experiments to determine the best-performing model.

## Data Loading and Preprocessing
- The dataset is loaded and preprocessed to handle missing values and prepare features and targets for training.
- Key features include various air quality metrics, with a focus on predicting PM2.5 levels.

## Model Training and Evaluation
- Several models were defined using Keras, and training was conducted with different configurations.
- The models were compiled and trained on the preprocessed dataset, with training history recorded for analysis.

## Experiment Tracking
- Multiple experiments were conducted with varying parameters such as learning rate, batch size, and dropout rates.
- The results of each experiment, including RMSE and model parameters, were tracked for comparison.

## Results and Best Models
- The following models were evaluated:
  - **Model 1**: 
    - RMSE: [Insert RMSE value]
    - Parameters: [Insert parameters]
  - **Model 2**: 
    - RMSE: [Insert RMSE value]
    - Parameters: [Insert parameters]
  - **Model 3**: 
    - RMSE: [Insert RMSE value]
    - Parameters: [Insert parameters]
  
- The best model based on the lowest RMSE was Model [Insert Model Number], with an RMSE of [Insert RMSE value].

## Installation Instructions
To run this project, ensure you have the following dependencies installed:
```bash
pip install tensorflow matplotlib pandas tabulate

