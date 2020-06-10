from torchvision import models
import torch
from torch import nn


class ResNet18Features(nn.Module):
    def __init__(self):
        super(ResNet18Features, self).__init__()
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
        print("ImageNet Pretrained ResNet-101 successfully loaded")

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
        x = torch.flatten(x, 1)
        return x


class ResNet18Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNet18Encoder, self).__init__()
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
        print("ImageNet Pretrained ResNet-18 successfully loaded")

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
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


class ResNet18DDT(nn.Module):
    def __init__(self, latent_dim):
        super(ResNet18DDT, self).__init__()
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


class ResNet18Classifier(nn.Module):
    def __init__(self, latent_dim=128):
        super(ResNet18Classifier, self).__init__()
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
        self.fc2 = nn.Linear(latent_dim, 64)
        self.fc3 = nn.Linear(64, 2)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc3.weight.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc3(self.relu(self.fc2(x)))
        return x


class ResNet18CCSA(nn.Module):
    def __init__(self, latent_dim):
        super(ResNet18CCSA, self).__init__()
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
        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool
        )

        self.fc1 = nn.Linear(512, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        self.fc3 = nn.Linear(64, 2)
        nn.init.xavier_normal_(self.fc1.weight.data)
        nn.init.xavier_normal_(self.fc2.weight.data)
        nn.init.xavier_normal_(self.fc3.weight.data)
        self.classifier = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc3
        )

        print("ImageNet Pretrained ResNet-101 successfully loaded")

    def forward(self, input):
        feature = torch.flatten(self.features(input), 1)
        pred = self.classifier(feature)
        return pred, feature
