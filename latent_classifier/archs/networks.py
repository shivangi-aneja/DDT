"""
    Autoencoder Networks
"""

from latent_classifier.archs.base import BaseClassifier
import torch.nn as nn


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
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2),
            nn.Softmax()
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
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
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
            nn.Linear(in_features=256, out_features=64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=2),
            nn.Softmax()
        )
