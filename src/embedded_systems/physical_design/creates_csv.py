import pandas as pd
from pathlib import Path

here = Path(__file__).parent
excel = pd.read_excel(
    here / "Robot Mass and Centers.xlsx",
    sheet_name="Part Weights",
    header=None,
    skiprows=1,
)

# Columns of the excel we need for the mjfc file
excel = excel.reindex(columns=[0, 1, 2, 3, 4, 7, 8, 9])
excel.columns = ["name", "mass_g", "x_mm", "y_mm", "z_mm", "sx_mm", "sy_mm", "sz_mm"]

first_empty = excel["name"].isna().idxmax()
df = excel.iloc[:first_empty]

df.to_csv(here/ "robot_parts.csv", index=False)
print(df)
