import streamlit as st
import numpy as np
import pickle
import os
import torch
import matplotlib.pyplot as plt

from scripts.train import Generator  # Import the trained GAN Generator

# Set project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_data.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "generator.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load Data Scaler
@st.cache_data
def load_scaler():
    try:
        with open(DATA_PATH, "rb") as f:
            _, scaler = pickle.load(f)
        return scaler
    except FileNotFoundError:
        st.error("❌ Processed data file not found! Please preprocess the data first.")
        return None

# Load Trained Generator Model
@st.cache_resource
def load_generator():
    scaler = load_scaler()
    if scaler is None:
        return None

    latent_dim = 16
    input_dim = scaler.mean_.shape[0]
    
    generator = Generator(latent_dim, input_dim)
    try:
        generator.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        generator.eval()
        return generator
    except FileNotFoundError:
        st.error("❌ Generator model not found! Train the GAN first.")
        return None

# Generate Synthetic Data
def generate_synthetic_data(generator, scaler, num_samples=1000):
    noise = torch.randn(num_samples, 16)  # Generate noise
    generated_data = generator(noise).detach().numpy()
    return scaler.inverse_transform(generated_data)  # Reverse scale data

# Visualization
def plot_feature_distribution(fake_data):
    plt.figure(figsize=(10, 5))
    plt.hist(fake_data[:, 0], bins=50, alpha=0.5, label="Generated Data")
    plt.legend()
    plt.title("Feature Distribution of Generated Data")
    plt.savefig(os.path.join(RESULTS_DIR, "feature_distribution.png"))
    st.image(os.path.join(RESULTS_DIR, "feature_distribution.png"))

# Streamlit UI
st.title("🧬 Cancer Prediction GAN")
st.write("A Generative Adversarial Network (GAN) for generating synthetic cancer-related data.")

# Load GAN Model
generator = load_generator()

if generator:
    num_samples = st.slider("Select Number of Synthetic Samples", min_value=100, max_value=5000, step=100, value=1000)
    
    if st.button("🔄 Generate Synthetic Data"):
        scaler = load_scaler()
        fake_data = generate_synthetic_data(generator, scaler, num_samples)

        # Save generated data
        np.save(os.path.join(RESULTS_DIR, "generated_samples.npy"), fake_data)
        st.success(f"✅ {num_samples} synthetic samples generated successfully!")
        
        # Plot results
        plot_feature_distribution(fake_data)

        st.download_button(
            label="📥 Download Synthetic Data",
            data=fake_data.tobytes(),
            file_name="generated_samples.npy",
            mime="application/octet-stream"
        )

# Footer
# Footer Styling
footer = """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            text-align: center;
            padding: 15px;
            background-color: #0E1117;
            color: white;
            font-size: 14px;
        }
        .footer a {
            color: #1DA1F2;
            text-decoration: none;
            font-size: 14px;
            margin: 0 10px;
        }
        .footer .name {
            font-size: 18px;
            font-weight: bold;
        }
    </style>
    <div class="footer">
        🚀 Developed by <span class="name">Ritik Kumar</span> <br/> 
        <a href="https://www.linkedin.com/in/ritik-kumar-34ab70218/" target="_blank">🔗 LinkedIn</a> | 
        <a href="https://github.com/RitikRK96" target="_blank">💻 GitHub</a>
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
