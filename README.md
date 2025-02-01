
# Beijing Air Quality Forecasting with LSTM Networks

This repository contains experiments for forecasting Beijing’s air quality (PM2.5) using deep learning models based on Long Short-Term Memory (LSTM) networks. Over 25 experiments have been conducted with different hyperparameters and network architectures to determine the best configuration for accurate forecasting.

## Table of Contents

- [Beijing Air Quality Forecasting with LSTM Networks](#beijing-air-quality-forecasting-with-lstm-networks)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Setup and Dependencies](#setup-and-dependencies)
  - [Data Preprocessing](#data-preprocessing)
  - [Sequence Generation](#sequence-generation)
  - [Experiments and Configurations](#experiments-and-configurations)
    - [Experiments 1-20 (Single LSTM layer)](#experiments-1-20-single-lstm-layer)
    - [Experiments 21-25 (Stacked/Multiple LSTM Layers)](#experiments-21-25-stackedmultiple-lstm-layers)
  - [Results and Analysis](#results-and-analysis)
    - [Summary of Key Results](#summary-of-key-results)
      - [Experiments 1-20 (Single LSTM Layer)](#experiments-1-20-single-lstm-layer-1)
      - [Experiments 21-25 (Stacked LSTM Layers)](#experiments-21-25-stacked-lstm-layers)
    - [Best Submissions](#best-submissions)
  - [Training Loss Plots](#training-loss-plots)
  - [Usage](#usage)
  - [Conclusion](#conclusion)
  - [Acknowledgements](#acknowledgements)

## Overview

The goal of this project is to forecast the concentration of PM2.5 (air quality indicator) in Beijing using time series data. We preprocess the data, generate sequences suitable for LSTM networks, and experiment with various network configurations and hyperparameters. The experiments span simple single-layer LSTM models to deeper, stacked LSTM architectures.

## Project Structure
```
.
├── train.csv                 # Training dataset
├── test.csv                  # Test dataset
├── submissions/              # Directory for saving submission CSV files and plots
├── notebook.ipynb            # Jupyter Notebook containing the experiments
└── README.md                 # Project documentation 
```
## Setup and Dependencies

To run the experiments, ensure you have installed the following Python libraries:

- **pandas**: For data manipulation.
- **numpy**: For numerical operations.
- **tensorflow**: For building and training LSTM networks.
- **matplotlib**: For plotting training loss curves.
- **scikit-learn**: For scaling data and calculating performance metrics.

Install the dependencies via pip:

```bash
pip install pandas numpy tensorflow matplotlib scikit-learn
```

## Data Preprocessing

1. **Loading the Data:**  
   The `train.csv` and `test.csv` files are loaded into pandas DataFrames.

2. **Datetime Conversion and Feature Extraction:**  
   The `datetime` column is converted into a pandas datetime format. Additional time-based features such as `hour`, `dayofweek`, and `month` are extracted to enhance the model input.

3. **Missing Values:**  
   Missing values are handled by filling them with the mean of their respective numerical columns.

4. **Feature Scaling:**  
   Features (all columns except `pm2.5` and an identifier) are scaled using `StandardScaler` to ensure that the LSTM model trains effectively.

## Sequence Generation

A helper function creates input sequences of a specified length (`seq_len`) to transform the tabular data into a time series format. Each sequence is used as input to predict the next PM2.5 value. This step is crucial for feeding data into LSTM models.

## Experiments and Configurations

The experiments were split into two groups:

### Experiments 1-20 (Single LSTM layer)

- **Configuration Details:**  
  Each experiment in this group uses a single LSTM layer with varying:
  - Number of units (e.g., 32, 64, 128)
  - Dropout rates (0.2 to 0.5)
  - Batch sizes (32, 64, 128)
  - Learning rates (0.001, 0.005, 0.01)
  - Sequence lengths (24, 48, 72)
  - Epochs (30–50)

- **Experiment Tracking:**  
  Each experiment creates sequences, splits the data into training and validation sets (80/20 split), builds a Sequential LSTM model, trains it, evaluates it using RMSE (Root Mean Squared Error), and saves test predictions as CSV files in the `submissions/` directory.

- **Key Observations:**  
  - The best results from this batch were achieved by **Experiment 17 (RMSE ≈ 72.10)** and **Experiment 8 (RMSE ≈ 75.86)**.

### Experiments 21-25 (Stacked/Multiple LSTM Layers)

- **Configuration Details:**  
  To further improve performance, experiments 21–25 explore deeper architectures with multiple LSTM layers. For example:
  - **Experiment 25:**  
    - Architecture:  
      - LSTM layer with 256 units (return sequences enabled)
      - Dropout of 0.4
      - LSTM layer with 128 units (return sequences enabled)
      - Dropout of 0.3
      - LSTM layer with 64 units
      - Dropout of 0.2
      - Dense layer with 1 unit (for final output)
    - Other parameters include a batch size of 32, learning rate of 0.001, 50 epochs, and a sequence length of 72.

- **Experiment Tracking:**  
  The code dynamically builds the model based on the provided layer configurations, trains the model, evaluates its performance (RMSE), and saves submissions. Results are appended to a global list for comparison.

- **Key Observations:**  
  - Experiment 23 produced the lowest RMSE among the new experiments (RMSE ≈ 75.28).
  - Although Experiment 25 employed a deeper stacked architecture, its performance (RMSE ≈ 80.19) was not as competitive as the best single-layer model from the earlier experiments.

## Results and Analysis

### Summary of Key Results

#### Experiments 1-20 (Single LSTM Layer)
| Experiment | Parameters                        | Architecture | RMSE    |
|------------|-----------------------------------|--------------|---------|
| 1          | units=32, lr=0.001, seq=24         | LSTM(32)     | 82.88   |
| 2          | units=64, lr=0.001, seq=24         | LSTM(64)     | 81.52   |
| ...        | ...                               | ...          | ...     |
| 8          | units=64, lr=0.01, seq=48          | LSTM(64)     | 75.86   |
| 17         | units=64, lr=0.01, seq=72          | LSTM(64)     | 72.10   |
| ...        | ...                               | ...          | ...     |

#### Experiments 21-25 (Stacked LSTM Layers)
| Experiment | Parameters                          | Architecture   | RMSE    |
|------------|-------------------------------------|----------------|---------|
| 21         | layers=2, lr=0.001, seq=48           | Stacked LSTM   | 82.26   |
| 22         | layers=2, lr=0.001, seq=48           | Stacked LSTM   | 85.79   |
| 23         | layers=3, lr=0.001, seq=72           | Stacked LSTM   | 75.28   |
| 24         | layers=3, lr=0.001, seq=48           | Stacked LSTM   | 79.70   |
| 25         | layers=3, lr=0.001, seq=72           | Stacked LSTM   | 80.19   |

### Best Submissions

The top three submissions (considering both experiment groups) are:
1. **Experiment 17** – RMSE: 72.10
2. **Experiment 23** – RMSE: 75.28
3. **Experiment 8**  – RMSE: 75.86

These CSV files are saved in the `submissions/` directory (e.g., `best_submission_1.csv`, `best_submission_2.csv`, etc.).

## Training Loss Plots

For the best-performing models, training loss curves were plotted to visualize the learning progress over epochs. Each plot includes:
- A curve showing the decrease in training loss (MSE).
- A horizontal line indicating the final training loss.
- Annotations that include the experiment number and corresponding RMSE.

These plots are saved as PNG files in the `submissions/` folder (e.g., `best_model_1_loss_plot.png`).

## Usage

1. **Data Preparation:**  
   Place `train.csv` and `test.csv` in the repository root.

2. **Run the Notebook:**  
   Open the Jupyter Notebook (`notebook.ipynb`) and run all cells to execute the experiments. The code automatically creates a `submissions/` directory, trains models, evaluates them, saves submission CSV files, and generates plots.

3. **Review Results:**  
   - Check the console output for experiment summaries and RMSE scores.
   - Open the CSV files in the `submissions/` folder to review the test predictions.
   - View the training loss plots to analyze model convergence.

## Conclusion

This series of experiments illustrates the impact of hyperparameter tuning and model architecture on forecasting performance. The best result was achieved by a single-layer LSTM model (Experiment 17) with an RMSE of 72.10. Although deeper, stacked LSTM architectures were explored (Experiments 21–25), they did not outperform the best single-layer configuration. These findings help guide further refinements for improving air quality forecasts.

## Acknowledgements

- **Data Source:** The air quality dataset for Beijing.
- **Libraries:** Thanks to the maintainers of TensorFlow, Keras, scikit-learn, and pandas for providing robust tools for deep learning and data analysis.
