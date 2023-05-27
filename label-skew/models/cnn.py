from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class CNNTarget(nn.Module):
    def __init__(self, in_channels=3, n_kernels=500, out_dim=10):
        super(CNNTarget, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def produce_feature(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN_B(nn.Module):
    def __init__(self, in_channels=3, n_kernels=500, out_dim=10):
        super(CNN_B, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_kernels, 5)
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 5)
        self.bn2 = nn.BatchNorm2d(2*n_kernels)

        self.fc1 = nn.Linear(2 * n_kernels * 5 * 5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, out_dim)

    def produce_feature(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        return x

    def getallfea(self, x):
        fealist = []
        x = self.conv1(x)
        fealist.append(x)
        x = self.bn1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        fealist.append(x)
        x = self.bn2(x)
        x = self.pool(F.relu(x))

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        fealist.append(x)
        x = F.relu(self.bn3(x))
        x = self.fc2(x)
        fealist.append(x)
        return fealist

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.fc3(x)
        return x