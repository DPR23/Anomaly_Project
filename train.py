import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mlflow

# 1. Define LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        
    def forward(self, x):
        encoded, (hidden, _) = self.encoder(x)
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(hidden_repeated)
        return decoded

# 2. Generate Synthetic Sensor Data (Normal behavior)
# 1000 sequences of 24 hours, 1 feature (temperature)
print("Generating normal warehouse temperature data...")
X_train_np = np.sin(np.linspace(0, 100, 1000 * 24)).reshape(1000, 24, 1) 
X_train = torch.tensor(X_train_np, dtype=torch.float32)

# 3. Training Loop with MLflow Tracking
input_dim = 1
hidden_dim = 16
epochs = 15

model = LSTMAutoencoder(input_dim, hidden_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

mlflow.set_experiment("Cold_Store_Anomaly_Detection")
with mlflow.start_run():
    mlflow.log_param("hidden_dim", hidden_dim)
    mlflow.log_param("epochs", epochs)
    
    print("Training Autoencoder...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, X_train)
        loss.backward()
        optimizer.step()
        
        mlflow.log_metric("loss", loss.item(), step=epoch)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # 4. Save Model
    torch.save(model.state_dict(), "lstm_autoencoder.pth")
    print("Model saved to lstm_autoencoder.pth. Check MLflow UI for tracking.")
