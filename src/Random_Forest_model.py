import pandas as pd
from pathlib import Path
import torch
from torch import nn
import torch.optim as optim

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

X_train, y_train, X1_test, y1_test = split_data(df1)

