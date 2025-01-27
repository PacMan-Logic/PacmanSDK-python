import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# state-value network for pacman
class PacmanNet(nn.Module):
    def __init__(self, input_channel_num, num_actions, extra_size):
        super().__init__()
        self.channels = input_channel_num
        self.embeddings = nn.ModuleList(
            [nn.Embedding(9, 16) for _ in range(input_channel_num)])
        self.conv1 = nn.Conv2d(64, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        self.encoder = nn.Linear(extra_size, 64)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_actions)

    def forward(self, x, y):
        B, C, H, W = x.shape
        embedded_channels = []
        for i in range(self.channels):
            flattened_channel = x[:, i, :, :].view(B, -1).long()
            embedded_channel = self.embeddings[i](flattened_channel)
            embedded_channel = embedded_channel.view(
                B, 16, H, W)
            embedded_channels.append(embedded_channel)
        # Concatenate along the channel dimension
        x = torch.cat(embedded_channels, dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn(x)
        x = F.relu(self.conv3(x))
        y = F.sigmoid(self.encoder(y))
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x+y))
        return self.fc2(x)


# state-value network for ghost
class GhostNet(nn.Module):
    def __init__(self, input_channel_num, num_actions, extra_size):
        super().__init__()
        self.channels = input_channel_num
        self.embeddings = nn.ModuleList(
            [nn.Embedding(9, 16) for _ in range(input_channel_num)])

        self.conv1 = nn.Conv2d(64, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        self.encoder = nn.Linear(extra_size, 64)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_actions*3)

    def forward(self, x, y):
        B, C, H, W = x.shape
        embedded_channels = []
        for i in range(self.channels):
            flattened_channel = x[:, i, :, :].view(B, -1).long()
            embedded_channel = self.embeddings[i](flattened_channel)
            embedded_channel = embedded_channel.view(
                B, 16, H, W)
            embedded_channels.append(embedded_channel)
        # Concatenate along the channel dimension
        x = torch.cat(embedded_channels, dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn(x)
        x = F.relu(self.conv3(x))
        # print(x.shape)

        y = F.sigmoid(self.encoder(y))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x+y))
        return self.fc2(x).view(-1, 3, 5)


# test the shape of the output
if __name__ == "__main__":
    rand_input = torch.rand(1, 4, 38, 38)
    extra_input = torch.rand(1, 10)
    pacman_net = PacmanNet(4, 5, 10)
    res = pacman_net(rand_input, extra_input)
    print(res.shape)

    ghost_net = GhostNet(4, 5, 10)
    res = ghost_net(rand_input, extra_input)
    print(res.shape)
