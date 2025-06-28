import torch
import torch.nn as nn


class TextureAutoencoder(nn.Module):
    def __init__(self):
        super(TextureAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # Conv1: 128x128 -> 64x64
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # Conv2: 64x64 -> 32x32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # Conv3: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc_encoder = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 16 * 16),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            # TransConv1: 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # TransConv2: 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # TransConv3: 64x64 -> 128x128
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_encoder(x)

        # Decode
        x = self.fc_decoder(x)
        x = x.view(x.size(0), 64, 16, 16)  # Unflatten
        x = self.decoder(x)

        return x

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc_encoder(x)
        return x

    def decode(self, x):
        x = self.fc_decoder(x)
        x = x.view(x.size(0), 64, 16, 16)
        x = self.decoder(x)
        return x