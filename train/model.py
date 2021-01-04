import torch
import torch.nn as nn
from utils import orthogonal_init

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NatureDQN(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super(NatureDQN, self).__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)


class ResidualBlock(nn.Module):
  def __init__(self, in_channels):
    super(ResidualBlock, self).__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1), nn.ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1), nn.ReLU()
    )

  def forward(self, x):
    out = self.layers(x)
    return out + x


class ImpalaBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ImpalaBlock, self).__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ResidualBlock(out_channels),
        ResidualBlock(out_channels)
    )

  def forward(self, x):
    x = self.layers(x)
    return x


class Impala(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super(Impala, self).__init__()
    self.layers = nn.Sequential(
        ImpalaBlock(in_channels=in_channels, out_channels=16),
        ImpalaBlock(in_channels=16, out_channels=32),
        ImpalaBlock(in_channels=32, out_channels=32),
        nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=32 * 8 * 8, out_features=feature_dim),
        nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    x = self.layers(x)
    return x
