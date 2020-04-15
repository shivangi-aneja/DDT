from common.utils.pytorch_modules import Flatten, Reshape
from torchvision import models
from common.models.resnet_helper import *


class VariationalEncoder0(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalEncoder0, self).__init__()
        self.resnet_model = models.resnet18(pretrained=True)
        self.conv1 = self.resnet_model.conv1
        self.bn1 = self.resnet_model.bn1
        self.relu = self.resnet_model.relu
        self.maxpool = self.resnet_model.maxpool
        self.layer1 = self.resnet_model.layer1
        self.layer2 = self.resnet_model.layer2
        self.layer3 = self.resnet_model.layer3
        self.layer4 = self.resnet_model.layer4
        self.avgpool = self.resnet_model.avgpool
        self.fc1 = nn.Linear(512, latent_dim)
        self.fc2 = nn.Linear(512, latent_dim)

    def encode(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        h1 = self.fc1(x)
        h2 = self.fc2(x)

        return h1, h2

    # randomly samples a vector from mean and variance given
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class VariationalEncoder1(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalEncoder1, self).__init__()
        self.resnet_model = models.resnet18(pretrained=True)
        self.conv1 = self.resnet_model.conv1
        self.bn1 = self.resnet_model.bn1
        self.relu = self.resnet_model.relu
        self.maxpool = self.resnet_model.maxpool
        self.layer1 = self.resnet_model.layer1
        self.layer2 = self.resnet_model.layer2
        self.layer3 = self.resnet_model.layer3
        self.avgpool = self.resnet_model.avgpool
        self.fc1 = nn.Linear(256, latent_dim)
        self.fc2 = nn.Linear(256, latent_dim)

    def encode(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        h1 = self.fc1(x)
        h2 = self.fc2(x)

        return h1, h2

    # randomly samples a vector from mean and variance given
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ForensicEncoder1(nn.Module):
    def __init__(self, latent_dim):
        super(ForensicEncoder1, self).__init__()
        self.resnet_model = models.resnet18(pretrained=True)
        self.conv1 = self.resnet_model.conv1
        self.bn1 = self.resnet_model.bn1
        self.relu = self.resnet_model.relu
        self.maxpool = self.resnet_model.maxpool
        self.layer1 = self.resnet_model.layer1
        self.layer2 = self.resnet_model.layer2
        self.layer3 = self.resnet_model.layer3
        self.avgpool = self.resnet_model.avgpool
        self.fc1 = nn.Linear(256, latent_dim)

    def encode(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        h1 = self.fc1(x)
        return h1

    def forward(self, x):
        h1 = self.encode(x)
        return h1


class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.resnet_model = models.resnet18(pretrained=True)
        self.conv1 = self.resnet_model.conv1
        self.bn1 = self.resnet_model.bn1
        self.relu = self.resnet_model.relu
        self.maxpool = self.resnet_model.maxpool
        self.layer1 = self.resnet_model.layer1
        self.layer2 = self.resnet_model.layer2
        self.layer3 = self.resnet_model.layer3
        self.avgpool = self.resnet_model.avgpool
        self.fc1 = nn.Linear(256, 2)

    def encode(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        h1 = self.fc1(x)
        return h1

    def forward(self, x):
        out = self.encode(x)
        return out


class EncoderLatent(nn.Module):
    def __init__(self, latent_dim):
        super(EncoderLatent, self).__init__()
        self.resnet_model = models.resnet18(pretrained=True)
        self.conv1 = self.resnet_model.conv1
        self.bn1 = self.resnet_model.bn1
        self.relu = self.resnet_model.relu
        self.maxpool = self.resnet_model.maxpool
        self.layer1 = self.resnet_model.layer1
        self.layer2 = self.resnet_model.layer2
        self.layer3 = self.resnet_model.layer3
        self.avgpool = self.resnet_model.avgpool
        self.fc1 = nn.Linear(256, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 2)

    def encode(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        z = self.relu(self.fc1(x))
        h1 = self.fc2(z)
        return z, h1

    def forward(self, x):
        z, out = self.encode(x)
        return z, out


class VariationalEncoder2(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalEncoder2, self).__init__()
        self.resnet_model = models.resnet18(pretrained=True)
        self.conv1 = self.resnet_model.conv1
        self.bn1 = self.resnet_model.bn1
        self.relu = self.resnet_model.relu
        self.maxpool = self.resnet_model.maxpool
        self.layer1 = self.resnet_model.layer1
        self.layer2 = self.resnet_model.layer2
        self.avgpool = self.resnet_model.avgpool
        self.fc1 = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(128, latent_dim)

    def encode(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.maxpool(x)
        x = self.layer2(x)
        # x = self.maxpool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        h1 = self.fc1(x)
        h2 = self.fc2(x)

        return h1, h2

    # randomly samples a vector from mean and variance given
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar