import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from data_utils import WeatherDataset
from lstm_models import WeatherLSTM

def train_and_plot(task_name, dataset, col_name, scaler, save_path, epochs=50):
    print(f"\n=== Processing {task_name} ===")
    
    # 1. Get Split Data
    X_train, y_train, X_test, y_test, dates, raw_data, split_idx = \
        dataset.get_train_test_split(col_name, scaler)
    
    print(f"Train Samples: {len(X_train)} | Test Samples: {len(X_test)}")

    # 2. Setup Model
    model = WeatherLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss {loss.item():.5f}")

    # 4. Save Model
    torch.save(model.state_dict(), save_path)
    print("Model saved.")

    # 5. Verification & Plotting
    model.eval()
    with torch.no_grad():
        # Predict on Train
        train_pred = model(X_train)
        # Predict on Test
        test_pred = model(X_test)

    # Inverse Transform (Go back to real values)
    train_pred_real = scaler.inverse_transform(train_pred.numpy())
    test_pred_real = scaler.inverse_transform(test_pred.numpy())
    y_train_real = scaler.inverse_transform(y_train.numpy())
    y_test_real = scaler.inverse_transform(y_test.numpy())

    # Calculate Error (RMSE) on Test Set
    rmse = np.sqrt(np.mean((test_pred_real - y_test_real)**2))
    print(f">> Test RMSE: {rmse:.4f}")

    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))
    
    # Plot Actual Data (Entire set)
    # We plot dates vs raw_data
    plt.plot(dates, raw_data, label='Actual Data', color='lightgray', linewidth=2)

    # Plot Train Predictions
    # Lookback offset required for x-axis alignment
    train_dates = dates[7:split_idx] 
    plt.plot(train_dates, train_pred_real, label='Train Pred', color='blue', alpha=0.7)

    # Plot Test Predictions
    test_dates = dates[split_idx:]
    plt.plot(test_dates, test_pred_real, label='Test Pred', color='red', alpha=0.7)

    plt.axvline(x=dates[split_idx], color='black', linestyle='--', label='Train/Test Split')
    plt.title(f"{task_name} Prediction (LSTM)\nTest RMSE: {rmse:.2f}")
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Save plot image
    plt.savefig(f"graph_{task_name}.png")
    print(f"Graph saved as graph_{task_name}.png")
    # plt.show() # Uncomment if running locally with a screen

if __name__ == "__main__":
    ds = WeatherDataset()
    
    # 1. Temperature
    train_and_plot("Temperature", ds, ds.col_temp, ds.scaler_temp, "temp_model.pth", epochs=350)
    
    # 2. ET0
    train_and_plot("ET0", ds, ds.col_et0, ds.scaler_et0, "et0_model.pth", epochs=350)
    
    ds.save_scalers()
    print("\nAll Done. Check the .png files generated.")