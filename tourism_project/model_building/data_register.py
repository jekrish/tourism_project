"""
Data Registration
------------------
Reads the raw dataset from the repository's data folder, validates that all
the columns described in the data dictionary are present, and prints a short
summary. The dataset itself already lives inside the GitHub repo (it was
pushed once from this notebook), so there is no external dataset store to
register it with -- "registration" here means: validate + summarise.
"""

import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Columns expected from the data dictionary in the problem statement
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
    "DurationOfPitch",
]

missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("Columns:", list(df.columns))
print("\nMissing values per column:")
print(df.isna().sum())
print("\nTarget distribution (ProdTaken):")
print(df["ProdTaken"].value_counts())
print("\nTarget distribution (%):")
print((df["ProdTaken"].value_counts(normalize=True) * 100).round(2))
