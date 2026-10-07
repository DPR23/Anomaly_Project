import spaces
import gradio as gr
import torch
import torch.nn as nn

# 1. Define the identical model architecture
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

# 2. Load the trained weights
model = LSTMAutoencoder(input_dim=1, hidden_dim=16)
try:
    model.load_state_dict(torch.load("lstm_autoencoder.pth", map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    pass # Handled gracefully in UI

# 3. Prediction function
@spaces.GPU
def detect_anomaly(temp_input):
    try:
        temps = [float(x.strip()) for x in temp_input.split(',')]
        if len(temps) != 24:
            return "Error: Please enter exactly 24 comma-separated temperature readings."
        
        tensor_data = torch.tensor(temps, dtype=torch.float32).view(1, 24, 1)
        
        with torch.no_grad():
            reconstruction = model(tensor_data)
            
        loss = nn.MSELoss()(reconstruction, tensor_data).item()
        
        threshold = 0.5
        if loss > threshold:
            return f"⚠️ ANOMALY DETECTED! Equipment failure risk. (Error Score: {loss:.4f})"
        else:
            return f"✅ Normal Operations. (Error Score: {loss:.4f})"
            
    except Exception as e:
        return "Error processing input. Ensure it is only numbers and commas."

# 4. Gradio Interface
normal_data = "0.0, 0.26, 0.5, 0.71, 0.87, 0.97, 1.0, 0.97, 0.87, 0.71, 0.5, 0.26, 0.0, -0.26, -0.5, -0.71, -0.87, -0.97, -1.0, -0.97, -0.87, -0.71, -0.5, -0.26"
anomaly_data = "0.0, 0.26, 0.5, 0.71, 5.0, 6.0, 1.0, 0.97, 0.87, 0.71, 0.5, 0.26, 0.0, -0.26, -0.5, -0.71, -0.87, -0.97, -1.0, -0.97, -0.87, -0.71, -0.5, -0.26"

demo = gr.Interface(
    fn=detect_anomaly,
    inputs=gr.Textbox(lines=3, label="24-Hour Temperature Sequence"),
    outputs=gr.Textbox(label="System Status"),
    title="IoT Sensor Anomaly Detection for Cold Stores",
    description="Paste a sequence of 24 temperature readings to check for cooling anomalies.",
    examples=[[normal_data], [anomaly_data]]
)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
