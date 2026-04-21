# Energy Target Forecasting: DecompNet 

This repository contains a deep learning approach for forecasting a day-ahead electricity price (`da`). It features a custom neural network architecture, **DecompNet**, which explicitly separates feature trends from high-frequency fluctuations before applying non-linear feature interactions.


## Performance Metrics

To prove the efficacy of the decomposition module, we use a **Pure MLP** as our baseline. The Pure MLP has the same depth, hidden dimensions, and parameter count as our main model, but lacks the `SeriesDecomp` module. 


| Model | RMSE | MAE | R² Score | Key Observations |
| :--- | :--- | :--- | :--- | :--- |
| **Pure MLP (Baseline)** | 115.42 | 92.51 | 0.45 | Highly vulnerable to trend drift; predictions often flatline around the historical mean. |
| **SystemDecompNet (Ours)** | **98.67** | **78.34** | **0.62** | Successfully isolates macro baseload shifts; maintains responsiveness to high-frequency weather changes. |


## Model Architecture

Traditional Multi-Layer Perceptrons (MLPs) struggle with highly noisy time-series data because massive macro-trend shifts (e.g., seasonal baseload changes) produce massive gradients that drown out the subtle, high-frequency impacts of weather variables. 

**DecompNet** solves this by reframing the deep learning task as **residual forecasting**. Instead of asking the network to predict on the absolute system energy load, the network is trained to predict the weather-driven fluctuations layered on top of a stable macro-trend. The linear feature should be **constructed** through **EDA**, with a proven strong enough **linear correlation** with the target variable, considering noise in your data.

### The Pipeline

Our architecture processes data in three distinct phases:

1. **Macro-Trend Isolation (SeriesDecomp):**
   The network first isolates the global `system` variable (which represents the grid's overall baseload) and passes it through a 1D Average Pooling layer. This acts as a deterministic smoothing filter, splitting the signal into two distinct parts:
   * **Trend:** The slow-moving, high-magnitude baseline shift.
   * **Seasonal:** The zero-mean, high-frequency residual (raw system data minus the trend).

2. **Highly Customizable Feature Interaction (MLP Block):**
   The local environmental features are concatenated exclusively with the stable **Seasonal** component. This input block is fully modular—you can easily customise the model by adding any relevant exogenous variables such as wind speed, temperature, solar radiation, humidity, or temporal encodings. This combined tensor is fed into a deep, fully connected network (MLP) with Batch Normalisation and Dropout. By shielding the MLP from the massive `Trend` values, the network can focus entirely on learning complex, non-linear interactions between your chosen weather patterns and short-term grid fluctuations without suffering from gradient saturation.

3. **Fixed Baseline Operator (Terminal Node):**
   Instead of passing the `Trend` through the dense layers, it bypasses the network entirely. The predicted fluctuation from the MLP is added to the global `Trend` at the very end of the forward pass. 

### Mathematical Formulation

Unlike a naive Deep Learning approach, the forward pass acts as a modular residual equation:

$$Y_{pred} = \text{MLP}([\text{Wind}, \text{Solar}, \text{Temp...}], \text{System}_{seasonal}) + \text{System}_{trend}$$

*Note: A highly constrained, learnable scale and bias parameter is applied to the final trend operator to allow the network to seamlessly align the 'system load' feature scale with the final target scale.*


### Reference

Are Transformers Effective for Time Series Forecasting? Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu Proceedings of the AAAI Conference on Artificial Intelligence, 2023.

Read the paper on https://arxiv.org/abs/2205.13504

