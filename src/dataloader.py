import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
# import EnergyDataLoader
# import EnergySQL

esql = EnergySQL("location", "location_Weather")
ed = EnergyDataLoader('location')

def str_date_delta(date='20250201', date_delta=-30):
    return (pd.Timestamp(date) + pd.Timedelta(days=date_delta)).strftime('%Y%m%d')

def select_features(features=[], NN=0, startdate='20250101', enddate='20250131'):
    df_train = # your code
    df_train['latitude'] = pd.to_numeric(df_train.index.get_level_values('latlon').str.extract(r'(\d+\.\d+)N', expand=False), errors='coerce')
    df_train['longitude'] = pd.to_numeric(df_train.index.get_level_values('latlon').str.extract(r'(\d+\.\d+)E', expand=False), errors='coerce')
    df_train.reset_index(inplace=True)
    df_train.set_index('datetime', inplace=True)
    df_train.drop(columns=['latlon', 'T'], inplace=True, errors='ignore')
    return df_train

def get_node_idx(df_train):
    unique_latitude = np.sort(df_train['latitude'].unique())
    unique_longitude = np.sort(df_train['longitude'].unique())
    df_train['latitude_rank'] = df_train['latitude'].map(dict(zip(unique_latitude, range(len(unique_latitude)))))
    df_train['longitude_rank'] = df_train['longitude'].map(dict(zip(unique_longitude, range(len(unique_longitude)))))
    sr_pos = df_train['latitude'] * df_train['longitude']
    unique_pos = np.sort(sr_pos.unique())
    df_train['idx'] = sr_pos.map(dict(zip(unique_pos, range(len(unique_pos)))))
    return df_train

def feature_engineering(df_train):
    df_train['hour'] = df_train.index.hour + df_train.index.minute / 60
    df_train['dayofweek'] = df_train.index.dayofweek
    df_train['month'] = df_train.index.month    

    start_date = df_train.index.min().normalize()
    end_date = df_train.index.max().normalize() + pd.Timedelta(hours=23, minutes=45)
    full_grid = pd.date_range(start=start_date, end=end_date, freq='15min')

    def reindex_group(g):
        g = g[~g.index.duplicated(keep='first')]
        return g.reindex(full_grid).interpolate(method='cubic').ffill().bfill()

    df_train = df_train.groupby(['latitude_rank', 'longitude_rank'], group_keys=False).apply(reindex_group)
    
    if 't2' in df_train.columns:
        df_train['temperature_ave'] = df_train.groupby(['latitude_rank', 'longitude_rank'])['t2'].transform(
            lambda x: x.resample('D').mean().reindex(x.index, method='ffill')
        )

    cols_to_scale = df_train.columns.drop(['idx'], errors='ignore')
    scaler = StandardScaler()
    df_train[cols_to_scale] = scaler.fit_transform(df_train[cols_to_scale])
    return df_train

def prepare_data(raw_features=[], features=[], NN=0, startdate='20250101', enddate='20250131', train=True):
    df_feature = select_features(features=raw_features, NN=NN,
                                 startdate=str_date_delta(startdate, -1), enddate=enddate)
    df_feature = get_node_idx(df_feature)
    df_feature = feature_engineering(df_feature)
    
    df_feature_ = df_feature[features + ['idx']].set_index('idx', append=True).sort_index()

    sys_data = ed.pull(['system'], start=str_date_delta(startdate, -1), end=enddate) #replace by your code
    sys_data.columns = ['system']
    
    sys_data_reindexed = sys_data.reindex(df_feature_.index.get_level_values(0))
    df_feature_['system'] = sys_data_reindexed['system'].values
    df_feature_['system'] = df_feature_['system'].ffill().bfill() # Handle missing values

    if train:
        y_ = ed.pull(['da'], start=startdate, end=enddate)
        y_ = y_.where(y_ != 0, -400)
    else:
        y_ = pd.DataFrame(
            np.nan,
            index=pd.date_range(start=startdate, end=str_date_delta(enddate, 1), freq='15min', inclusive='left'),
            columns=['da']
        )
        
    start_datetime, end_datetime = f'{startdate} 00:00:00', f'{enddate} 23:45:00'
    return df_feature_.loc[start_datetime: end_datetime], y_.loc[start_datetime: end_datetime]

class RollingDataset(Dataset):    
    def __init__(self, feature_df, target_df, window_size, stride=1):
        self.window_size = window_size
        self.stride = stride
        self.time_index = target_df.index
        
        times = feature_df.index.get_level_values(0).unique()
        self.n_nodes = len(feature_df.loc[times[0]])
        self.n_features = feature_df.shape[1]
        
        feature_values = feature_df.sort_index().values
        self.feature_array = torch.FloatTensor(
            feature_values.reshape(len(times), self.n_nodes, self.n_features)
        )
        self.target_array = torch.FloatTensor(target_df.sort_index().values)
        self.n_samples = (len(times) - window_size) // stride + 1

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size        
        X = self.feature_array[start_idx:end_idx] 
        y = self.target_array[start_idx:end_idx]   
        return X, y

class RollingDataLoader:
    def __init__(self, feature_df, target_df, window_size, stride=1, shuffle=True):
        self.dataset = RollingDataset(feature_df, target_df, window_size, stride)
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))
        
    def __len__(self):
        return len(self.dataset)
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for idx in self.indices:
            yield self.dataset[idx]
