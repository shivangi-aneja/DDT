"""
    Classifier Networks
"""

from classifier.archs.base import BaseClassifier
from common.utils.pytorch_modules import Flatten
import torch.nn as nn

# This architecture is used for DDCNet
class Classifier1(BaseClassifier):
    """
    Test Classifier 1
    """

    def __init__(self, *args, **kwargs):
        super(Classifier1, self).__init__(*args, **kwargs)

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)

    def make_classifier(self):
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
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256, momentum=0.1),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128, momentum=0.1),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=2)
        )


class Classifier2(BaseClassifier):
    """
    Test Classifier 2
    """

    def __init__(self, *args, **kwargs):
        super(Classifier2, self).__init__(*args, **kwargs)

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)

    def make_classifier(self):
        return nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(8, track_running_stats=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(16, track_running_stats=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(128, track_running_stats=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(256, track_running_stats=False),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64, track_running_stats=False),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=2),
            nn.Softmax()
        )


class Classifier3(BaseClassifier):
    """
    Test Classifier 3
    """

    def __init__(self, *args, **kwargs):
        super(Classifier3, self).__init__(*args, **kwargs)

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)

    def make_classifier(self):
        return nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.ReLU(),

            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=2),
            nn.Softmax()
        )
