"""
    Autoencoder Networks
"""

from dsne_ae.archs.base import BaseAutoencoder
from common.utils.pytorch_modules import Flatten, Reshape
import torch.nn as nn


class Autoencoder1(BaseAutoencoder):
    """
    Autoencoder 1
    """

    def __init__(self, *args, **kwargs):
        super(Autoencoder1, self).__init__(*args, **kwargs)

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)

    def make_encoder(self):
        return nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(256, momentum=0.1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            Flatten(),
            nn.Linear(in_features=1024, out_features=128),
            nn.BatchNorm1d(128, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(p=0.2)

        )

    def make_decoder(self):
        return nn.Sequential(

            nn.Linear(in_features=128, out_features=1024),
            nn.BatchNorm1d(1024, momentum=0.1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            Reshape(256, 2, 2),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8, momentum=0.1),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(3, 3), stride=1),
            nn.Tanh()
        )

    def make_classifier(self):
        return nn.Sequential(
            nn.Linear(in_features=128, out_features=2)
        )
