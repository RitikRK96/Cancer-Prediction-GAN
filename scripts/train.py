import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/processed_data.pkl")
with open(DATA_PATH, "rb") as f:
    data, scaler = pickle.load(f)


data = torch.tensor(data, dtype=torch.float32)

# GAN Models
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
input_dim = data.shape[1]
latent_dim = 16
batch_size = 64
epochs = 500
lr = 0.0002

# Initialize models and optimizers
generator = Generator(latent_dim, input_dim)
discriminator = Discriminator(input_dim)
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr)
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    real_data = data[torch.randint(0, len(data), (batch_size,))]

    # Train Discriminator
    optimizer_d.zero_grad()
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    d_real_loss = criterion(discriminator(real_data), real_labels)
    noise = torch.randn(batch_size, latent_dim)
    fake_data = generator(noise)
    d_fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)

    d_loss = d_real_loss + d_fake_loss
    d_loss.backward()
    optimizer_d.step()

    # Train Generator
    optimizer_g.zero_grad()
    g_loss = criterion(discriminator(fake_data), real_labels)
    g_loss.backward()
    optimizer_g.step()

    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}] D Loss: {d_loss:.4f} G Loss: {g_loss:.4f}")

# Save models
os.makedirs("../models", exist_ok=True)
torch.save(generator.state_dict(), "../models/generator.pth")
torch.save(discriminator.state_dict(), "../models/discriminator.pth")

print("✅ Training Complete! Models saved.")
