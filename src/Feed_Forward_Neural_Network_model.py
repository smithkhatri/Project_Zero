import pandas as pd
from pathlib import Path
import torch
from torch import nn
import torch.optim as optim
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

project_root = Path(__file__).resolve().parent.parent

df1 = pd.read_csv(project_root / "data" / "processed" / "TSLA_features.csv")
df2 = pd.read_csv(project_root / "data" / "processed" / "AAPL_features.csv")
df3 = pd.read_csv(project_root / "data" / "processed" / "NVDA_features.csv")


def split_data(df):
    
    X = df[["Close", "High", "Low", "Open", "Volume", "Return", "MA_5", "MA_20", "Volatility"]] #9 features
    y = df["Target"]

    l = int(len(df)*0.8)
    

    X_train = X.iloc[:l]


    y_train = y.iloc[:l]

    X_test = X.iloc[l:]
    y_test = y.iloc[l:]

    return X_train, y_train, X_test, y_test


# 9 features x last 10 days = 90 input features

X_train, y_train, X_test, y_test = split_data(df1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



def create_10_day_window(arr):
    r = []
    for i in range(len(arr)- 9):
        r.append(arr[i:i+10].flatten())
    
    return np.array(r)


X_train_final = create_10_day_window(X_train_scaled)
y_train_final = np.array(y_train[9:])

X_test_final = create_10_day_window(X_test_scaled)
y_test_final = np.array(y_test[9:])

X = torch.tensor(X_train_final, dtype=torch.float32)
y = torch.tensor(y_train_final, dtype=torch.float32).unsqueeze(1)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(90, 900)
        self.layer2 = nn.Linear(900, 900)
        self.layer3 = nn.Linear(900, 900)
        self.layer4 = nn.Linear(900, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

model = NeuralNetwork()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(500):
    predictions = model(X)

    loss = criterion(predictions, y) # THIS | BELOW THIS I DON"T GET THE INTUITION

    optimizer.zero_grad() # THIS

    loss.backward() # THIS
    optimizer.step() # THIS | UNTIL HERE, I get it is loop and adjusting the weights but... how does it adjust the weights? What is the intuition behind this?
                        # I don't see the model being adjusted, it's indirectly?

    if epoch % 10 == 0:
        print("Epoch:", epoch, "Loss:", loss.item())



X = torch.tensor(X_test_final, dtype=torch.float32)
y = torch.tensor(y_test_final, dtype=torch.float32).unsqueeze(1)



model.eval()

with torch.no_grad():
    predictions = model(X)

predictions = predictions.numpy()
real_y = y.numpy()


print("accuracy:", np.mean((predictions > 0.5) == (real_y > 0.5)))