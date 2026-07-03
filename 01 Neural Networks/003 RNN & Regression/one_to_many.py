#==========================================#
# Title:  Stock prices prediction with LSTM (One-to-Many)
# Author: hyunjun Choe
# Date:   2026-06-29
#==========================================#
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# --- Hyperparameter Settings ---
input_seq_length = 1       # Input sequence length (1)
output_seq_length = 30     # Output sequence length
epochs = 20
hidden_size = 50
num_layers = 2
batch_size = 1
lr = 0.0001
ticker = "SBUX"

"""
Step1: Data Preprocessing
"""
data = yf.download(ticker, start="2020-01-01", end="2023-12-31")
data = data['Close'].values.reshape(-1, 1)
input_size = data.shape[1]

scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Function to create sequences (One-to-Many)
def create_sequences(data, input_length, output_length):
    sequences = []
    targets = []
    for i in range(len(data) - input_length - output_length + 1):
        # Use 1 data point as a sequence
        sequences.append(data[i:i+input_length])
        # Use the next output_length data points as the target
        targets.append(data[i+input_length:i+input_length+output_length])
    return np.array(sequences), np.array(targets)

X, y = create_sequences(data_normalized, input_seq_length, output_seq_length)
# Adjust the shape of y to (num_samples, output_seq_length)
y = y.squeeze()

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""
Step2: Define LSTM Model
"""
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        # Set the output layer size to output_seq_length
        self.fc2 = nn.Linear(hidden_size//2, output_size)

    def forward(self, x):
        hidden_cell = (torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device),
                       torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))

        out, _ = self.lstm(x, hidden_cell)
        # Since input sequence length is 1, use only the last output
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.fc2(out)
        return out

"""
Step3: Define model, loss function, optimizer, and train
"""
# Set output_size to the output sequence length
model = LSTMModel(input_size, hidden_size, num_layers, output_seq_length)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
print("--- Starting One-to-Many Model Training ---")
for epoch in range(epochs):
    train_loss = 0.0
    for sequences, targets in train_loader:
        outputs = model(sequences)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    print(f'Epoch {str(epoch+1).zfill(2)}/{epochs}, Loss: {train_loss/len(train_loader):.7f}')

"""
Step4: Evaluate Model
"""
model.eval()
with torch.no_grad():
    # Check prediction results with one sample from the test data
    sample_X, sample_y = next(iter(test_loader))

    prediction = model(sample_X)

    prediction = scaler.inverse_transform(prediction.numpy())
    actual = scaler.inverse_transform(sample_y.numpy())

# Visualize the results (prediction for one sample)
plt.figure(figsize=(12, 6))
plt.title(f'One-to-Many LSTM Prediction (Predicting next {output_seq_length} days)')
# Set the x-axis as the prediction period (days)
x_axis = np.arange(output_seq_length)
plt.plot(x_axis, actual.flatten(), label='Actual Price')
plt.plot(x_axis, prediction.flatten(), label='Predicted Price', linestyle='--')
plt.xlabel('Days into the Future')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
