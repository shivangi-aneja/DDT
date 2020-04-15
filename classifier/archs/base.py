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
        z = None
        ctr = 0
        for m in self.classifier:
            if isinstance(m, nn.MaxPool2d):
                output_size = x.size()
                x, pool_idx = m(x)
            else:
                x = m(x)
                if ctr == 30:
                    z = x
            ctr += 1
        return z, x
