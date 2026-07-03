#==========================================#
# Title:  Stock prices prediction with LSTM (Many-to-One)
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

# --- 하이퍼파라미터 설정 ---
seq_length = 60         # 입력 시퀀스 길이
epochs = 20
hidden_size = 50
num_layers = 2
output_size = 1         # 출력 크기 (1개)
batch_size = 1
lr = 0.0001
ticker = "SBUX"

"""
Step1: 데이터 전처리
"""
# yfinance를 통해 주가 데이터 다운로드
data = yf.download(ticker, start="2020-01-01", end="2023-12-31")
data = data['Close'].values.reshape(-1, 1)
input_size = data.shape[1]

# 데이터 정규화 (0과 1 사이로)
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# 시퀀스 데이터 생성 함수 (Many-to-One)
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        # seq_length 만큼의 데이터를 sequence로 사용
        sequences.append(data[i:i+seq_length])
        # 바로 다음 날의 데이터를 target으로 사용
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

X, y = create_sequences(data_normalized, seq_length)

# 훈련 데이터와 테스트 데이터 분리 (80:20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# PyTorch 텐서로 변환
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# DataLoader 생성
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""
Step2: LSTM 모델 정의
"""
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, output_size)

    def forward(self, x):
        # 초기 hidden state와 cell state를 0으로 설정
        hidden_cell = (torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device),
                       torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))

        out, _ = self.lstm(x, hidden_cell)
        # Many-to-One: 시퀀스의 마지막 출력만 사용
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.fc2(out)
        return out

"""
Step3: 모델 정의, 손실 함수, 옵티마이저 설정 및 훈련
"""
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
print("--- Many-to-One 모델 훈련 시작 ---")
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
Step4: 모델 평가
"""
model.eval()
with torch.no_grad():
    predictions = []
    actuals = []
    for sequences, targets in test_loader:
        output = model(sequences)
        predictions.append(output.item())
        actuals.append(targets.item())

    # 예측값과 실제값을 다시 원래 스케일로 변환
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.title('Many-to-One LSTM Stock Price Prediction')
plt.plot(actuals, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
