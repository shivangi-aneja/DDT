"""
base class for autoencoder creation
"""
import torch.nn as nn
import torch


class BaseAutoencoder(nn.Module):
    """
    Base Class for autoencoders
    """

    def __init__(self):
        """
        initialize the parameters
        :param latent_dim: latent dimension of the autoencoder
        :param 101: dropout rate for the autoencoder
        """
        super(BaseAutoencoder, self).__init__()
        self.encoder = self.make_encoder()
        self.decoder = self.make_decoder()
        self.classifier = self.make_classifier()
        self.init()

    def make_encoder(self):
        """
        create the encoder part of the autoencoder
        :return: encoder
        """
        raise NotImplementedError('`make_encoder` is not implemented')

    def make_decoder(self):
        """
        create the decoder part of the autoencoder
        :return: decoder
        """
        raise NotImplementedError('`make_decoder` is not implemented')

    def make_classifier(self):
        """
        create the decoder part of the autoencoder
        :return: decoder
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
        :param y: original target
        :param selection: to mask out or not
        :return: x_recon : reconstructed input, z : latent representation
        """
        unpool_info = []

        for m in self.encoder:
            if isinstance(m, nn.MaxPool2d):
                output_size = x.size()
                x, pool_idx = m(x)
                unpool_info.append({'output_size': output_size,
                                    'indices': pool_idx})
            else:
                x = m(x)
        z = x

        for m in self.decoder:
            if isinstance(m, nn.MaxUnpool2d):
                x = m(x, **unpool_info.pop())
            else:
                x = m(x)
        x_recon = x

        for m in self.classifier:
            y_hat = m(z)

        return z, y_hat, x_recon
