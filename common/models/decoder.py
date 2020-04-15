from torch import nn


class decoder_vae1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(decoder_vae1, self).__init__()
        # Classifier layers
        self.prelu = nn.PReLU()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = self.fc2(self.prelu(self.fc1(x)))
        return x


class decoder_vae2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(decoder_vae2, self).__init__()
        # Classifier layers
        self.fc1 = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x):
        x = self.fc1(x)
        return x
