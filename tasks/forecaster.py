import torch
import joblib
import numpy as np
from lstm_models import WeatherLSTM
from data_utils import WeatherDataset
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
SCALERS_DIR = os.path.join(SCRIPT_DIR, 'scalers')

class TaskForecaster:
    def __init__(self):
        self.device = torch.device("cpu")
        
        # Load Data Utils just to get the latest sequence
        self.ds = WeatherDataset()
        
        # Load Models
        self.temp_model = WeatherLSTM()
        self.et0_model = WeatherLSTM()
        
        try:
            self.temp_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "temp_model.pth")))
            self.et0_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "et0_model.pth")))
            print("LSTMs loaded successfully.")
        except FileNotFoundError:
            print("WARNING: LSTM models not found. Please run train_tasks.py first.")

        self.temp_model.eval()
        self.et0_model.eval()

        # Load Scalers
        try:
            self.scaler_temp = joblib.load(os.path.join(SCALERS_DIR, "scaler_temp.pkl"))
            self.scaler_et0 = joblib.load(os.path.join(SCALERS_DIR, "scaler_et0.pkl"))

            self.ds.scaler_temp = self.scaler_temp
            self.ds.scaler_et0 = self.scaler_et0
        except:
            print("Scalers not found.")

    def predict(self, task_id, days):
        """
        Recursive prediction for N days.
        task_id: 0 = Temp, 1 = ET0
        """
        predictions = []
        
        # Get the real last 7 days from CSV
        seq_temp, seq_et0 = self.ds.get_latest_sequence()
        
        # Select Model and Sequence
        if task_id == 0:
            model = self.temp_model
            current_seq = seq_temp.unsqueeze(0) # Add batch dim -> [1, 7, 1]
            scaler = self.scaler_temp
        else:
            model = self.et0_model
            current_seq = seq_et0.unsqueeze(0)
            scaler = self.scaler_et0
            
        with torch.no_grad():
            for _ in range(days):
                # 1. Predict next step (normalized)
                next_val_norm = model(current_seq)
                
                # 2. Store prediction (inverse transform to get real value)
                val_real = scaler.inverse_transform(next_val_norm.numpy())[0][0]
                predictions.append(val_real)
                
                # 3. Update sequence for recursive prediction
                # Drop oldest value (index 0), append new prediction
                # current_seq is [1, 7, 1]. next_val_norm is [1, 1]
                new_step = next_val_norm.unsqueeze(1) # [1, 1, 1]
                current_seq = torch.cat((current_seq[:, 1:, :], new_step), dim=1)
                
        return predictions