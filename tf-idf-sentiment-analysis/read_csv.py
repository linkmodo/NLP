import pandas as pd

# Read the first few rows of the CSV file
df = pd.read_csv("Reddit_Data.csv")
print("\nColumns in the dataset:", df.columns.tolist())
print("\nFirst 5 rows of the dataset:")
print(df.head())
