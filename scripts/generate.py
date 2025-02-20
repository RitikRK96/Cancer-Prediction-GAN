import torch
import numpy as np
import pickle
import os
from train import Generator

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Absolute path to script's directory
DATA_PATH = os.path.join(BASE_DIR, "../data/processed_data.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "../models/generator.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "../results")
RESULT_FILE = os.path.join(RESULTS_DIR, "generated_samples.npy")

# Ensure necessary files exist
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Missing file: {DATA_PATH}. Ensure preprocessing has been run.")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Missing file: {MODEL_PATH}. Ensure model training is complete.")

# Load preprocessed scaler and trained Generator model
with open(DATA_PATH, "rb") as f:
    _, scaler = pickle.load(f)

latent_dim = 16
input_dim = scaler.mean_.shape[0]

generator = Generator(latent_dim, input_dim)
generator.load_state_dict(torch.load(MODEL_PATH))
generator.eval()

# Generate synthetic data and reverse scale it back to original range
noise = torch.randn(1000, latent_dim)  # Generate 1000 samples of noise.
generated_data = generator(noise).detach().numpy()
generated_data = scaler.inverse_transform(generated_data)

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Save generated data to file
np.save(RESULT_FILE, generated_data)
print(f"✅ Synthetic data saved successfully at {RESULT_FILE}!")
