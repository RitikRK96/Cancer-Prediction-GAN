import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go one level up
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data.pkl")
GENERATED_DATA_PATH = os.path.join(BASE_DIR, "results", "generated_samples.npy")
FEATURE_PLOT_PATH = os.path.join(BASE_DIR, "results", "feature_distribution.png")

# Check if the necessary files exist
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ File not found: {DATA_PATH}")

if not os.path.exists(GENERATED_DATA_PATH):
    raise FileNotFoundError(f"❌ File not found: {GENERATED_DATA_PATH}")

# Load scaler and original data info
with open(DATA_PATH, "rb") as f:
    real_data, scaler = pickle.load(f)  # Ensure real data is loaded

# Load generated synthetic data
fake_data = np.load(GENERATED_DATA_PATH)

# Select a feature to compare
feature_index = 0  # Change this index to compare different features

plt.figure(figsize=(10, 5))
plt.hist(real_data[:, feature_index], bins=50, alpha=0.5, label="Real Data", color='blue')
plt.hist(fake_data[:, feature_index], bins=50, alpha=0.5, label="Generated Data", color='orange')
plt.legend()
plt.title(f"Feature Distribution (Feature {feature_index}): Real vs. Generated")
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.savefig(FEATURE_PLOT_PATH)
plt.show()

print(f"✅ Evaluation Complete! Check '{FEATURE_PLOT_PATH}'")
