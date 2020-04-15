from common.utils.pytorch_modules import Flatten, Reshape
from torchvision import models
from common.models.resnet_helper import *


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


class ResNet18VariationalEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(ResNet18VariationalEncoder, self).__init__()
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


class ResNet18VariationalCombinedEncoder(nn.Module):
    def __init__(self, latent_dim, face_model_path=None, lip_seq_model_path=None):
        super(ResNet18VariationalCombinedEncoder, self).__init__()

        self.face_model = ResNet18VariationalEncoder(pretrained=True, latent_dim=latent_dim)
        # self.face_model.load_state_dict(torch.load(face_model_path))

        self.lip_seq_model = ResNet18VariationalEncoderResidual(pretrained=True, latent_dim=latent_dim)
        # self.lip_seq_model.load_state_dict(torch.load(lip_seq_model_path))

        self.fusion_2d = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
                                       nn.LeakyReLU(), nn.MaxPool2d(kernel_size=2, stride=2))

        # self.fusion_3d = nn.Sequential(nn.Conv3d(1024, 512, 1, stride=2, padding=1, dilation=1, bias=True),
        #                             nn.LeakyReLU(), nn.MaxPool3d(kernel_size=2, stride=2))
        # self.fc = nn.Sequential(nn.Linear(8192, 2048), nn.ReLU(), nn.Dropout(p=0.85),
        #                         nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(p=0.85),
        #                         nn.Linear(512, 101))

        self.fc = nn.Sequential(nn.Linear(8192, 2048), nn.LeakyReLU(), nn.Dropout(p=0.85))
        self.fc_mu = nn.Linear(in_features=2048, out_features=128)
        self.fc_logvar = nn.Linear(in_features=2048, out_features=128)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_face, x_lip):
        # Face features
        x_face = self.face_model.conv1(x_face)
        x_face = self.face_model.bn1(x_face)
        x_face = self.face_model.relu(x_face)
        x_face = self.face_model.maxpool(x_face)
        x_face = self.face_model.layer1(x_face)
        x_face = self.face_model.layer2(x_face)
        x_face = self.face_model.layer3(x_face)
        x_face = self.face_model.layer4(x_face)

        x_lip = self.lip_seq_model.conv1(x_lip)
        x_lip = self.lip_seq_model.bn1(x_lip)
        x_lip = self.lip_seq_model.relu(x_lip)
        x_lip = self.lip_seq_model.maxpool(x_lip)
        x_lip = self.lip_seq_model.layer1(x_lip)
        x_lip = self.lip_seq_model.layer2(x_lip)
        x_lip = self.lip_seq_model.layer3(x_lip)
        x_lip = self.lip_seq_model.layer4(x_lip)

        x_combined = torch.cat((x_face, x_lip), dim=1)
        for i in range(x_face.size(1)):
            x_combined[:, (2 * i), :, :] = x_face[:, i, :, :]
            x_combined[:, (2 * i + 1), :, :] = x_lip[:, i, :, :]

        # x_combined = x_combined.view(x_combined.size(0), 1024, 1, 8, 8)
        x_combined = x_combined.view(x_combined.size(0), 1024, 8, 8)
        x_combined = self.fusion_2d(x_combined)
        x_combined = x_combined.view(x_combined.size(0), -1)
        x_combined = self.fc(x_combined)

        mu = self.fc_mu(x_combined)
        logvar = self.fc_logvar(x_combined)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ResNet18VariationalCombinedEncoder2(nn.Module):
    def __init__(self, latent_dim):
        super(ResNet18VariationalCombinedEncoder2, self).__init__()
        self.face_model = ResNet18Encoder(latent_dim=latent_dim)
        self.lip_model = ResNet18Encoder(latent_dim=latent_dim)
        self.lrelu = nn.LeakyReLU()
        self.fc_mu = nn.Linear(in_features=2 * latent_dim, out_features=128)
        self.fc_logvar = nn.Linear(in_features=2 * latent_dim, out_features=128)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_face, x_lip):
        z_face = self.face_model(x_face)
        z_lip = self.lip_model(x_lip)
        z_combined = torch.cat([z_face, z_lip], dim=1)
        mu = self.fc_mu(z_combined)
        logvar = self.fc_logvar(z_combined)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ResNet18VariationalEncoderResidual(nn.Module):
    def __init__(self, pretrained, latent_dim):
        super(ResNet18VariationalEncoderResidual, self).__init__()
        self.resnet_model = models.resnet18(pretrained=pretrained)
        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
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


class ResNet18VAE(nn.Module):
    def __init__(self, pretrained, latent_dim):
        super(ResNet18VAE, self).__init__()
        self.resnet_model = models.resnet18(pretrained=pretrained)

        self.encoder = nn.Sequential(
            self.resnet_model.conv1,
            self.resnet_model.bn1,
            self.resnet_model.relu,
            self.resnet_model.maxpool,
            self.resnet_model.layer1,
            self.resnet_model.layer2,
            self.resnet_model.layer3,
            self.resnet_model.layer4,
            self.resnet_model.avgpool,
            Flatten()
        )
        self.fc1 = nn.Linear(512, latent_dim)
        self.fc2 = nn.Linear(512, latent_dim)

        self.reshape = Reshape(latent_dim, 1, 1)

        # ResNet-18 Decoder

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),  # Output HxW = 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # Output HxW = 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # Output HxW = 16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),  # Output HxW = 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # Output HxW = 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),  # Output HxW = 128x128
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),

            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            # Output HxW = 256x256
            nn.Tanh()
        )

    def encode(self, input):
        x = self.encoder(input)
        h1 = self.fc1(x)
        h2 = self.fc2(x)

        return h1, h2

    def decode(self, input):
        x = self.decoder(self.reshape(input))
        return x

    # randomly samples a vector from mean and variance given
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return z, mu, logvar, x_recon


class ResNet18ForensicTransfer(nn.Module):
    def __init__(self, latent_dim):
        super(ResNet18ForensicTransfer, self).__init__()
        self.resnet_model = models.resnet18(pretrained=True)

        self.encoder = nn.Sequential(
            self.resnet_model.conv1,
            self.resnet_model.bn1,
            self.resnet_model.relu,
            self.resnet_model.maxpool,
            self.resnet_model.layer1,
            self.resnet_model.layer2,
            self.resnet_model.layer3,
            self.resnet_model.layer4,
            self.resnet_model.avgpool,
            Flatten(),
            nn.Linear(512, latent_dim)
        )
        self.reshape = Reshape(latent_dim, 1, 1)

        # self.decoder = PPM(fc_dim=latent_dim)
        # ResNet-18 Decoder
        # self.decoder = nn.Sequential(
        #     UpBlockResNet(in_channels=latent_dim, out_channels=512),
        #     UpBlockResNet(in_channels=512, out_channels=256),
        #     UpBlockResNet(in_channels=256, out_channels=128),
        #     UpBlockResNet(in_channels=128, out_channels=64),
        #     UpBlockResNet(in_channels=64, out_channels=32),
        #     UpBlockResNet(in_channels=32, out_channels=16),
        #     UpBlockResNet(in_channels=16, out_channels=8),
        #     UpBlockResNet(in_channels=8, out_channels=3, non_linearity=nn.Tanh())
        # )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),  # Output HxW = 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # Output HxW = 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # Output HxW = 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),  # Output HxW = 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # Output HxW = 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),  # Output HxW = 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 3, 4, 2, 1, bias=False),  # Output HxW = 256x256
            nn.Tanh()
        )

    def forward(self, x):
        # Encode
        z = self.encoder(x)
        x = self.reshape(z)
        x_recon = self.decoder(x)
        return z, x_recon


class ResNet18ForensicTransferResidual(nn.Module):
    def __init__(self, latent_dim):
        super(ResNet18ForensicTransferResidual, self).__init__()
        self.resnet_model = models.resnet18(pretrained=True)

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            self.resnet_model.bn1,
            self.resnet_model.relu,
            self.resnet_model.maxpool,
            self.resnet_model.layer1,
            self.resnet_model.layer2,
            self.resnet_model.layer3,
            self.resnet_model.layer4,
            self.resnet_model.avgpool,
            Flatten(),
        )
        self.reshape = Reshape(latent_dim, 1, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 1, 0, bias=False),  # Output HxW = 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # Output HxW = 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # Output HxW = 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),  # Output HxW = 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),  # Output HxW = 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),  # Output HxW = 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 6, 4, 2, 1, bias=False),  # Output HxW = 256x256
            nn.Tanh()
        )

    def forward(self, x):
        # Encode
        z = self.encoder(x)
        x = self.reshape(z)
        x_recon = self.decoder(x)
        return z, x_recon
