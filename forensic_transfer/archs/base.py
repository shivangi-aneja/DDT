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
        :param dropout: dropout rate for the autoencoder
        """
        super(BaseAutoencoder, self).__init__()
        self.encoder = self.make_encoder()
        self.decoder = self.make_decoder()
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

    def init(self):
        """
        init method
        :return:
        """
        pass

    def forward(self, x, y, selection=False):
        """
        pass input through the encoder and reconstruct with decoder
        :param x: original input
        :param y: original target
        :param selection: to mask out or not
        :return: x_recon : reconstructed input, z : latent representation
        """
        unpool_info = []
        real = torch.cat((torch.ones(64), torch.zeros(64))).cuda()
        fake = torch.cat((torch.zeros(64), torch.ones(64))).cuda()

        for m in self.encoder:
            if isinstance(m, nn.MaxPool2d):
                output_size = x.size()
                x, pool_idx = m(x)
                unpool_info.append({'output_size': output_size,
                                    'indices': pool_idx})
            else:
                x = m(x)

        z_unmasked = x

        if selection:
            # Masks out the values depending on class (y = 0 is real, y=1 is fake)
            mask = torch.where((y == 0.).unsqueeze(1), real, fake)
            x *= mask  # Mask out the latent space

        for m in self.decoder:
            if isinstance(m, nn.MaxUnpool2d):
                x = m(x, **unpool_info.pop())
            else:
                x = m(x)
        x_recon = x
        return z_unmasked, x_recon
