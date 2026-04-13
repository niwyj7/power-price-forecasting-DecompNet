import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Assuming dataloader.py remains largely the same, just ensure variable names match
from src.dataloader import prepare_data, str_date_delta, RollingDataLoader
from src.models import SystemDecompNet, PureMLP

def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train_loop(model, dataloader, device, epochs=25, lr=0.01, l1_lambda=0.008):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    
    for epoch in range(1, epochs + 1):
        total_loss, total_samples = 0.0, 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X)
            
            # MSE + L1 Regularization on the graph/spatial node weights
            loss = criterion(y_pred, y) + l1_lambda * torch.abs(model.node_aggregator.weight).sum()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * y.size(0)
            total_samples += y.size(0)
            
        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch: {epoch:03d} | Train MSE Loss: {total_loss/total_samples:.2f}')
            
    return model

@torch.no_grad()
def predict_all(model, dataloader, device):
    model.eval()
    all_preds = []
    
    for i, (X, _) in enumerate(dataloader):
        X = X.to(device)
        y_pred = model(X)
        
        start_idx = i * dataloader.dataset.stride
        end_idx = start_idx + dataloader.dataset.window_size
        time_index = dataloader.dataset.time_index[start_idx:end_idx]
        
        all_preds.append(pd.DataFrame(y_pred.cpu().numpy(), index=time_index, columns=['da_pred']))
    
    # Simple averaging overlapping rolling windows
    concat_values = pd.concat(all_preds)
    pred_df = concat_values.groupby(level=0).mean() 
    return pred_df.clip(lower=0, upper=1000)

def run_experiment(config, target_date):
    set_seeds(config['seed'])
    device = config['device']
    
    enddate = target_date.strftime('%Y%m%d')
    startdate = str_date_delta(enddate, -config['train_lookback_dates'])
    
    print(f"--- Preparing Data from {startdate} to {enddate} ---")
    feature_df, target_df = prepare_data(
        raw_features=config['raw_features'], 
        features=config['features'], 
        startdate=startdate, 
        enddate=enddate, 
        train=True
    )
    
    train_loader = RollingDataLoader(
        feature_df, target_df, 
        window_size=config['window_size'], 
        stride=1, shuffle=True
    )
    
    n_nodes = train_loader.dataset.n_nodes
    weather_dim = len(config['features']) - 1 # Assuming last is 'system'
    total_input_dim = len(config['features'])
    
    # Initialize both models for Ablation Study
    models = {
        "PureMLP_Baseline": PureMLP(input_dim=total_input_dim, nodes=n_nodes, seq_len=config['window_size']).to(device),
        "SystemDecompNet": SystemDecompNet(weather_dim=weather_dim, nodes=n_nodes, seq_len=config['window_size']).to(device)
    }

    results = {}
    for name, model in models.items():
        print(f"\n>>> Training {name}...")
        trained_model = train_loop(model, train_loader, device, epochs=config['epochs'])
        
        print(f">>> Predicting with {name}...")
        # Note: In a real scenario, prepare_data(train=False) for future dates here
        test_loader = RollingDataLoader(feature_df, target_df, window_size=config['window_size'], shuffle=False)
        preds = predict_all(trained_model, test_loader, device)
        results[name] = preds
    
    return results

if __name__ == '__main__':
    CONFIG = {
        'raw_features': ['tp','ssrd','sp','sf','rhu','d2','t2','win100_spd'], 
        'features': ['win100_spd','tp','ssrd','sp','sf','rhu','d2','t2','hour','dayofweek', 'system'],
        'train_lookback_dates': 30,
        'window_size': 24,
        'epochs': 25,
        'seed': 42,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    target_date = pd.to_datetime('2026-01-26')
    experiment_results = run_experiment(CONFIG, target_date)
    
    print("\n--- Final Prediction Hourly Means ---")
    for model_name, pred_df in experiment_results.items():
        print(f"\n{model_name}:")
        print(pred_df.resample('h').mean().head(5)) # Display first few hours
