from common.utils.pytorch_modules import Flatten, Reshape
import torch.nn as nn
import torch

class Autoencoder1(nn.Module):
    """
    Test Autoencoder 1
    """

    def __init__(self, latent_dim):
        super(Autoencoder1, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            Flatten(),
            nn.Linear(in_features=18432, out_features=latent_dim),
        )

        self.decoder = nn.Sequential(

            nn.Linear(in_features=latent_dim, out_features=18432),
            Reshape(512, 6, 6),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=64, out_channels=6, kernel_size=(3, 3), stride=1)

        )

        for layer in self.encoder.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)

        for layer in self.decoder.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)


    def forward(self, x):

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
        return z, x_recon

class Autoencoder2(nn.Module):
    """
    Test Autoencoder 2
    """

    def __init__(self, latent_dim):
        super(Autoencoder2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            Flatten(),
            nn.Linear(in_features=1024, out_features=latent_dim),
        )

        self.decoder = nn.Sequential(

            nn.Linear(in_features=latent_dim, out_features=1024),
            nn.BatchNorm1d(1024, momentum=0.1),
            nn.ReLU(),
            Reshape(256, 2, 2),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(3, 3), stride=1),
            nn.Tanh()

        )

        for layer in self.encoder.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)

        for layer in self.decoder.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)


    def forward(self, x):

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
        return z, x_recon


class VAE(nn.Module):
    """
    Test Autoencoder 2
    """

    def __init__(self, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            Flatten(),
        )

        self.fc1 = nn.Linear(in_features=1024, out_features=latent_dim)
        self.fc2 = nn.Linear(in_features=1024, out_features=latent_dim)

        self.decoder = nn.Sequential(

            nn.Linear(in_features=latent_dim, out_features=1024),
            nn.BatchNorm1d(1024, momentum=0.1),
            nn.ReLU(),
            Reshape(256, 2, 2),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.MaxUnpool2d(kernel_size=(2, 2), stride=2),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(3, 3), stride=1),
            nn.Tanh()

        )

        for layer in self.encoder.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)

        for layer in self.decoder.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)


    # randomly samples a vector from mean and variance given
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        unpool_info = []

        for m in self.encoder:
            if isinstance(m, nn.MaxPool2d):
                output_size = x.size()
                x, pool_idx = m(x)
                unpool_info.append({'output_size': output_size,
                                    'indices': pool_idx})
            else:
                x = m(x)

        mu = self.fc1(x)
        logvar = self.fc2(x)

        x = self.reparameterize(mu, logvar)
        z = x
        for m in self.decoder:
            if isinstance(m, nn.MaxUnpool2d):
                x = m(x, **unpool_info.pop())
            else:
                x = m(x)
        x_recon = x
        return z, mu, logvar, x_recon


class Autoencoder3(nn.Module):
    """
    Test Autoencoder 2
    """

    def __init__(self, latent_dim):
        super(Autoencoder3, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            Flatten(),
            nn.Linear(in_features=1024, out_features=latent_dim),
        )

        for layer in self.encoder.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)


    def forward(self, x):

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
        return z


class VAE_Encoder(nn.Module):
    """
    Test Autoencoder 2
    """

    def __init__(self, latent_dim):
        super(VAE_Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, return_indices=True),

            Flatten(),
        )

        self.fc1 = nn.Linear(in_features=1024, out_features=latent_dim)
        self.fc2 = nn.Linear(in_features=1024, out_features=latent_dim)


        for layer in self.encoder.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_normal_(layer.weight.data)
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)


    # randomly samples a vector from mean and variance given
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        unpool_info = []

        for m in self.encoder:
            if isinstance(m, nn.MaxPool2d):
                output_size = x.size()
                x, pool_idx = m(x)
                unpool_info.append({'output_size': output_size,
                                    'indices': pool_idx})
            else:
                x = m(x)

        mu = self.fc1(x)
        logvar = self.fc2(x)

        z = self.reparameterize(mu, logvar)
        return z, mu, logvar