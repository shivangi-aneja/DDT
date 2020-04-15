from torch import nn
import torch
from torchvision import models


class encoder_vae1(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(encoder_vae1, self).__init__()
        # Classifier layers
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2_1 = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        self.fc2_2 = nn.Linear(in_features=hidden_dim, out_features=latent_dim)

    def encode(self, x):
        x = self.relu(self.fc1(x))
        h1 = self.fc2_1(x)
        h2 = self.fc2_2(x)
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


class encoder_vae2(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(encoder_vae2, self).__init__()
        # Classifier layers
        self.fc1_1 = nn.Linear(in_features=input_dim, out_features=latent_dim)
        self.fc1_2 = nn.Linear(in_features=input_dim, out_features=latent_dim)

    def encode(self, x):
        h1 = self.fc1_1(x)
        h2 = self.fc1_2(x)
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


class encoder_resnet18(nn.Module):
    def __init__(self):
        super(encoder_resnet18, self).__init__()
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
        print("Resnet pretrained loaded")
        # Classifier layers
        # self.prelu = nn.PReLU()
        # self.fc1 = nn.Linear(in_features=512, out_features=hidden_dim)
        # self.fc2_1 = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        # self.fc2_2 = nn.Linear(in_features=hidden_dim, out_features=latent_dim)

    # randomly samples a vector from mean and variance given
    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return mu + eps * std

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        resnet_feat = torch.flatten(x, 1)     # resnet feature vector
        # x = self.prelu(self.fc1(resnet_feat))
        # mu = self.fc2_1(x)
        # logvar = self.fc2_2(x)
        # z = self.reparameterize(mu, logvar)
        return resnet_feat  #, z, mu, logvar