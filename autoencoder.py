import numpy as np
import torch
import torch.nn as nn


# Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(100, 55),
            nn.ReLU(),
            nn.Linear(55, 35),
            nn.ReLU(),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(35, 55),
            nn.ReLU(),
            nn.Linear(55, 100),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded





