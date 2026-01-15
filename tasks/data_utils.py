import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LOOKBACK = 7
CSV_PATH = os.path.join(SCRIPT_DIR, 'datasets', 'weather.csv')
SCALERS_DIR = os.path.join(SCRIPT_DIR, 'scalers')

class WeatherDataset:
    def __init__(self):
        print(f"Loading {CSV_PATH}...")
        try:
            self.df = pd.read_csv(CSV_PATH)
            # Ensure time is parsed for plotting
            self.df['time'] = pd.to_datetime(self.df['time'])
        except FileNotFoundError:
            print("Error: CSV not found. Generating dummy data...")
            dates = pd.date_range(start='1/1/2002', periods=1000)
            self.df = pd.DataFrame({
                'time': dates,
                'temperature_2m_mean (°C)': np.sin(np.linspace(0, 100, 1000)) * 10 + 20,
                'et0_fao_evapotranspiration (mm)': np.abs(np.sin(np.linspace(0, 100, 1000))) * 5
            })

        self.col_temp = 'temperature_2m_mean (°C)'
        self.col_et0 = 'et0_fao_evapotranspiration (mm)'

        self.scaler_temp = MinMaxScaler(feature_range=(0, 1))
        self.scaler_et0 = MinMaxScaler(feature_range=(0, 1))

    def create_sequences(self, data):
        """Helper to create LSTM sequences"""
        X, y = [], []
        for i in range(len(data) - LOOKBACK):
            X.append(data[i:i+LOOKBACK])
            y.append(data[i+LOOKBACK])
        return np.array(X), np.array(y)

    def get_train_test_split(self, target_col, scaler, split_ratio=0.8):
        """
        Splits data 80/20 and fits scaler ONLY on training data.
        Returns tensors for Train/Test and full raw data for plotting.
        """
        # 1. Get raw data
        raw_data = self.df[target_col].values.reshape(-1, 1).astype(float)
        dates = self.df['time'].values
        
        # 2. Calculate Split Index
        split_idx = int(len(raw_data) * split_ratio)
        
        # 3. Split raw data FIRST (before scaling)
        train_raw = raw_data[:split_idx]
        test_raw = raw_data[split_idx - LOOKBACK:]
        
        # 4. Fit scaler ONLY on training data (prevents data leakage)
        scaler.fit(train_raw)
        
        # 5. Transform both sets using the scaler fitted on training data
        train_scaled = scaler.transform(train_raw)
        test_scaled = scaler.transform(test_raw)
        
        # 6. Create sequences
        X_train, y_train = self.create_sequences(train_scaled)
        X_test, y_test = self.create_sequences(test_scaled)
        
        return (
            torch.FloatTensor(X_train), torch.FloatTensor(y_train),
            torch.FloatTensor(X_test), torch.FloatTensor(y_test),
            dates, raw_data, split_idx
        )

    def save_scalers(self):
        joblib.dump(self.scaler_temp, os.path.join(SCALERS_DIR, "scaler_temp.pkl"))
        joblib.dump(self.scaler_et0, os.path.join(SCALERS_DIR, "scaler_et0.pkl"))

    def get_latest_sequence(self):
        raw_temp = self.df[self.col_temp].values[-LOOKBACK:].reshape(-1, 1).astype(float)
        seq_temp = self.scaler_temp.transform(raw_temp)
        
        raw_et0 = self.df[self.col_et0].values[-LOOKBACK:].reshape(-1, 1).astype(float)
        seq_et0 = self.scaler_et0.transform(raw_et0)
        
        return torch.FloatTensor(seq_temp), torch.FloatTensor(seq_et0)