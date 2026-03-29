from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
file_path1 = project_root / "data" / "raw" / "TSLA_data.csv"
file_path2 = project_root / "data" / "raw" / "AAPL_data.csv"
file_path3 = project_root / "data" / "raw" / "NVDA_data.csv"




data1 = pd.read_csv(file_path1)
data2 = pd.read_csv(file_path2)
data3 = pd.read_csv(file_path3)


data1 = data1.drop([0]).reset_index(drop=True)
data2 = data2.drop([0]).reset_index(drop=True)
data3 = data3.drop([0]).reset_index(drop=True)


data1.to_csv(project_root / "data" / "processed" / "TSLA_data.csv", index=False)
data2.to_csv(project_root / "data" / "processed" / "AAPL_data.csv", index=False)
data3.to_csv(project_root / "data" / "processed" / "NVDA_data.csv", index=False)