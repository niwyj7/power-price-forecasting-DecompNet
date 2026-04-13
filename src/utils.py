import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def calculate_metrics(y_true, y_pred):
    """
    Calculate standard regression metrics for time-series forecasting.
    Handles NaN values automatically.
    """
    # Align indices just in case
    df = pd.DataFrame({'true': y_true, 'pred': y_pred}).dropna()
    
    if df.empty:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan}
        
    mae = mean_absolute_error(df['true'], df['pred'])
    rmse = np.sqrt(mean_squared_error(df['true'], df['pred']))
    r2 = r2_score(df['true'], df['pred'])
    
    return {
        "RMSE": round(rmse, 2), 
        "MAE": round(mae, 2), 
        "R2": round(r2, 4)
    }

def print_evaluation_report(target_df, predictions_dict):
    """
    Prints a formatted markdown table comparing model performances.
    """
    print("\n| Model | RMSE | MAE | R² Score |")
    print("| :--- | :--- | :--- | :--- |")
    
    for model_name, pred_df in predictions_dict.items():
        # Assuming pred_df has a single column and target_df has 'da'
        metrics = calculate_metrics(target_df['da'], pred_df.iloc[:, 0])
        print(f"| **{model_name}** | {metrics['RMSE']} | {metrics['MAE']} | {metrics['R2']} |")

def plot_predictions(target_df, predictions_dict, title="Energy Target Forecasting: Model Comparison", save_path=None):
    """
    Plots the ground truth vs. predicted values for all models.
    Great for adding visualizations to your README.md or EDA notebook.
    """
    plt.figure(figsize=(15, 6))
    
    # Plot Ground Truth
    plt.plot(target_df.index, target_df['da'], label='Actual (Ground Truth)', 
             color='black', linewidth=2, alpha=0.8)
    
    # Define colors for different models
    colors = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728']
    
    for i, (model_name, pred_df) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        plt.plot(pred_df.index, pred_df.iloc[:, 0], label=f'Predicted: {model_name}', 
                 color=color, linewidth=1.5, linestyle='--' if 'Baseline' in model_name else '-')

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Energy Target (da)', fontsize=12)
    
    # Formatting X-axis dates beautifully
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
