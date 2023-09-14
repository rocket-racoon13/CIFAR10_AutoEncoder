import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, kernel_size=5),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x