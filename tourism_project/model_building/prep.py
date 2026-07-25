"""
Data Preparation
-----------------
Loads the registered dataset from the repository, cleans it, removes
unnecessary columns, and splits it into train/test sets. The splits are
saved at the repo root as CSV files so the GitHub Actions workflow can pass
them to the next job (model training) as a workflow artifact.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(RAW_PATH)
print("Dataset loaded successfully.", df.shape)

# ---------------------------------------------------------------------------
# 1. Remove unnecessary columns
# ---------------------------------------------------------------------------
# "Unnamed: 0" is a stray index column written out by a previous CSV export.
# "CustomerID" is a unique identifier with no predictive value.
drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
df = df.drop(columns=drop_cols)
print(f"Dropped identifier/index columns: {drop_cols}")

# ---------------------------------------------------------------------------
# 2. Data cleaning
# ---------------------------------------------------------------------------
# Remove exact duplicate rows, if any.
before = df.shape[0]
df = df.drop_duplicates()
print(f"Dropped {before - df.shape[0]} duplicate rows.")

# "Gender" has a dirty duplicate category ('Fe Male') that should be 'Female'.
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# The data dictionary defines MaritalStatus as Single/Married/Divorced, but
# the raw data also contains 'Unmarried', which represents the same group as
# 'Single'. Standardise it so training and serving use the same categories.
if "MaritalStatus" in df.columns:
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# Drop rows with a missing target, if any (nothing to learn from them).
df = df.dropna(subset=["ProdTaken"])

print("Missing values remaining per column:")
print(df.isna().sum())

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
target_col = "ProdTaken"
X = df.drop(columns=[target_col])
y = df[target_col]

# stratify=y keeps the (imbalanced) purchase ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Xtrain: {Xtrain.shape}, Xtest: {Xtest.shape}")
print("Train target distribution:\n", ytrain.value_counts(normalize=True).round(3))
print("Test target distribution:\n", ytest.value_counts(normalize=True).round(3))

# ---------------------------------------------------------------------------
# 4. Save the splits at the repo root so the workflow can upload them
#    as an artifact for the next job (model training).
# ---------------------------------------------------------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv to the working directory.")
