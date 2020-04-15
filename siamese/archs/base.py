"""
base class for autoencoder creation
"""
import torch.nn as nn

class SiameseNetwork(nn.Module):
    """
    Base Class for Siamese Network
    """

    def __init__(self):
        """
        initialize the parameters
        :param latent_dim: latent dimension of the autoencoder
        :param dropout: dropout rate for the autoencoder
        """
        super(SiameseNetwork, self).__init__()
        self.network = self.make_network()
        self.init()

    def make_network(self):
        """
        create the encoder part of the autoencoder
        :return: encoder
        """
        raise NotImplementedError('`make_network` is not implemented')

    def init(self):
        """
        init method
        :return:
        """
        pass

    def forward(self, x1, x2, x3):
        """
        pass input through the encoder and reconstruct with decoder
        :param x: original input
        :return: x_recon : reconstructed input, z : latent representation
        """
        z1, z2, z3 = None, None, None
        ctr = 0
        for m in self.network:

            if isinstance(m, nn.MaxPool2d):
                x1, pool_idx = m(x1)
                x2, pool_idx = m(x2)
                x3, pool_idx = m(x3)
            else:
                x1 = m(x1)
                x2 = m(x2)
                x3 = m(x3)
                if ctr == 30:
                    z1 = x1
                    z2 = x2
                    z3 = x3
            ctr += 1
        return x1, z1, x2, z2, x3, z3
