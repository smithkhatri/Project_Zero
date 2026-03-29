import yfinance as yf
import pandas as pd
from pathlib import Path

current_file = Path(__file__)

project_root = current_file.resolve().parent.parent

path_to_raw_data = project_root / "data" / "raw"

# print(f"Project root: {project_root}")

data = yf.download("TSLA", start="2010-01-01", end="2026-03-27")

data1 = yf.download("AAPL", start="2010-01-01", end="2026-03-27")

data2 = yf.download("NVDA", start="2010-01-01", end="2026-03-27")


data.to_csv(path_to_raw_data/ "TSLA_data.csv", index=False)
data1.to_csv(path_to_raw_data/ "AAPL_data.csv", index=False)
data2.to_csv(path_to_raw_data/ "NVDA_data.csv", index=False)