import torch
import torch.nn as nn
import torch.nn.functional as F

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):

        x_t = x.permute(0, 2, 1)
        trend = self.moving_avg(x_t).permute(0, 2, 1)
        if trend.shape[1] != x.shape[1]:
            trend = trend[:, :x.shape[1], :]
        seasonal = x - trend
        return seasonal, trend

class MLPBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class SystemDecompNet(nn.Module):

    def __init__(self, weather_dim, nodes, seq_len, hidden_dim=128):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size=11)

        combined_dim = weather_dim + 2 
        
        self.feature_interaction = MLPBlock(combined_dim, hidden_dim)
        self.node_aggregator = nn.Linear(nodes, 1)
        self.time_regressor = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
     
        seq_len, nodes, f_dim = x.shape
        weather = x[:, :, :-1]
        system = x[:, :, -1:]

        sys_reshaped = system.permute(1, 0, 2) # [Nodes, Seq_len, 1]
        seasonal, trend = self.decomp(sys_reshaped)
        
        seasonal = seasonal.permute(1, 0, 2)
        trend = trend.permute(1, 0, 2)
        combined = torch.cat([weather, seasonal, trend], dim=-1)

        flat_combined = combined.reshape(-1, combined.shape[-1])
        interacted = self.feature_interaction(flat_combined)
        
        interacted = interacted.reshape(seq_len, nodes, -1)
        spatial_out = self.node_aggregator(interacted.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.time_regressor(spatial_out).squeeze(-1)
        return out

class PureMLP(nn.Module):

    def __init__(self, input_dim, nodes, seq_len, hidden_dim=128):
        super().__init__()
        self.feature_interaction = MLPBlock(input_dim, hidden_dim)
        self.node_aggregator = nn.Linear(nodes, 1)
        self.time_regressor = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        seq_len, nodes, _ = x.shape
        flat_x = x.reshape(-1, x.shape[-1])
        interacted = self.feature_interaction(flat_x)
        interacted = interacted.reshape(seq_len, nodes, -1)
        spatial_out = self.node_aggregator(interacted.permute(0, 2, 1)).permute(0, 2, 1)
        out = self.time_regressor(spatial_out).squeeze(-1)
        return out
