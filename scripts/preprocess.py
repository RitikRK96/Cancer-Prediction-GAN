import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/Cancer_Data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "../data/processed_data.pkl")

# Load dataset
df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded:")
print(df.head())  # Preview first few rows
print(f"📊 Data shape before processing: {df.shape}")

# **Drop unnecessary columns (id, Unnamed: 32)**
if "Unnamed: 32" in df.columns:
    df.drop(columns=["Unnamed: 32"], inplace=True)

if "id" in df.columns:
    df.drop(columns=["id"], inplace=True)

# Encode categorical column ('diagnosis': M = 1, B = 0)
label_encoder = LabelEncoder()
df["diagnosis"] = label_encoder.fit_transform(df["diagnosis"])

# Check for missing values
print("🧐 Checking missing values per column:")
print(df.isnull().sum())

# **Drop rows with missing values**
df.dropna(inplace=True)

# Ensure dataset is not empty
if df.shape[0] == 0:
    raise ValueError("❌ ERROR: No data left after dropping missing values!")

print(f"📊 Data shape after dropping missing values: {df.shape}")

# **Normalize numerical features**
scaler = StandardScaler()
processed_data = scaler.fit_transform(df.iloc[:, 1:].values)  # Excluding 'diagnosis' (first column)

# Save preprocessed data and scaler
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "wb") as f:
    pickle.dump((processed_data, scaler), f)

print(f"✅ Processed data saved at: {OUTPUT_PATH}")
