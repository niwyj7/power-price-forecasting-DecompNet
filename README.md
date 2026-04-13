# Energy Target Forecasting: DecompNet vs. Pure MLP

This repository contains a deep learning approach for forecasting a day-ahead electricity price (`da`). It features a custom neural network architecture, **DecompNet**, which explicitly separates system load trends from high-frequency seasonal fluctuations before applying non-linear feature interactions.

## Architecture

The `DecompNet` architecture first isolates the `system` load variable and passes it through the average pooling module to extract a smoothed `trend` component, subtracting this from the original signal to yield a zero-mean `seasonal` residual. These two decoupled signals—representing macro baseload shifts and local high-frequency fluctuations—are concatenated back alongside the raw weather features. This combined tensor is then fed into a deep Multi-Layer Perceptron (MLP) block equipped with Batch Normalisation and Dropout, allowing the dense network to learn complex, non-linear interactions between the weather and the stabilised seasonal baseline before aggregating the spatial nodes for the final prediction.

## Ablation Study: Performance Metrics

To prove the efficacy of the decomposition module, we use a **Pure MLP** as our baseline. The Pure MLP has the exact same depth, hidden dimensions, and parameter count as our main model, but lacks the `SeriesDecomp` module. 


| Model | RMSE | MAE | R² Score | Key Observations |
| :--- | :--- | :--- | :--- | :--- |
| **Pure MLP (Baseline)** | 115.42 | 92.51 | 0.45 | Highly vulnerable to trend drift; predictions often flatline around the historical mean. |
| **SystemDecompNet (Ours)** | **98.67** | **78.34** | **0.62** | Successfully isolates macro baseload shifts; maintains responsiveness to high-frequency weather changes. |


## Repository Structure

```text
energy-forecasting-decomp/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── main.py                   # Main training and evaluation pipeline
├── src/
│   ├── __init__.py
│   ├── models.py             # Contains SystemDecompNet and PureMLP
│   ├── dataloader.py         # Data pulling, feature engineering, and RollingDataLoader
│   └── baselines.py          # Naive persistence and statistical baselines
└── notebooks/
    └── 01_EDA_Analysis.ipynb # Exploratory Data Analysis & Noise Distribution
