import torch.nn as nn


class LinearRegressionforInsurance(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionforInsurance, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, 1024),  # output size 1024
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),  # BatchNorm with 1024
            nn.Dropout(0.4),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 512),  # output size 512
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),  # BatchNorm with 512
            nn.Dropout(0.3),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(512, 256),  # output size 256
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),  # BatchNorm with 256
            nn.Dropout(0.2),
        )
        self.layer4 = nn.Sequential(
            nn.Linear(256, 128),  # output size 128
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),  # BatchNorm with 128
            nn.Dropout(0.1),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(128, 1)  # output size 1 (final prediction)
        )

    def forward(self, X):
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        return self.layer5(X)
