from torch import nn


class CLASSIFIER(nn.Module):
    def __init__(self, latent_dim):
        super(CLASSIFIER, self).__init__()
        # Classifier layers
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=latent_dim, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        out = self.fc2(self.relu(self.fc1(x)))
        return out


class DISCRIMINATOR(nn.Module):
    def __init__(self, latent_dim):
        super(DISCRIMINATOR, self).__init__()
        # Classifier layers
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(in_features=latent_dim, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        out = self.sigmoid(self.fc2(self.prelu(self.fc1(x))))
        return out
