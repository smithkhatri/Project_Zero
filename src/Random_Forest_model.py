import pandas as pd
from pathlib import Path
import torch
from torch import nn
import torch.optim as optim

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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




def AI_model(df):

    X_train, y_train, X_test, y_test = split_data(df)

    model = RandomForestClassifier(n_estimators=100, random_state=40)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy * 100, "%")

AI_model(df1)
AI_model(df2)
AI_model(df3)