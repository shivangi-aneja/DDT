"""
base class for classifier creation
"""
import torch.nn as nn


class BaseClassifier(nn.Module):
    """
    Base Class for classifiers
    """

    def __init__(self):
        super(BaseClassifier, self).__init__()
        self.classifier = self.make_classifier()
        self.init()

    def make_classifier(self):
        """
        create the classifier
        :return: classifier
        """
        raise NotImplementedError('`make_classifier` is not implemented')

    def init(self):
        """
        init method
        :return:
        """
        pass

    def forward(self, x):
        """
        pass input through the encoder and reconstruct with decoder
        :param x: original input
        :return: x_recon : reconstructed input, z : latent representation
        """

        for m in self.classifier:
            x = m(x)
        return x
