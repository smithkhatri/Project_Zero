from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
file_path1 = project_root / "data" / "raw" / "TSLA_data.csv"
file_path2 = project_root / "data" / "raw" / "AAPL_data.csv"
file_path3 = project_root / "data" / "raw" / "NVDA_data.csv"

data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)
data3 = pd.read_csv(file_path3)

def process_data(data):
    data = data.drop([0]).reset_index(drop=True)
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data["Return"] = data["Close"].pct_change()
    data["MA_5"] = data["Close"].rolling(window=5).mean()
    data["MA_20"] = data["Close"].rolling(window=20).mean()
    data["Volatility"] = data["Return"].rolling(window=5).std()

    data["Target"] = (data["Close"].shift(-5) > data["Close"]).astype(int) #target for price after 5days

    data = data.dropna()
    return data


data1 = process_data(data1)
data2 = process_data(data2)
data3 = process_data(data3)


data1.to_csv(project_root / "data" / "processed" / "TSLA_features.csv", index=False)
data2.to_csv(project_root / "data" / "processed" / "AAPL_features.csv", index=False)
data3.to_csv(project_root / "data" / "processed" / "NVDA_features.csv", index=False)
