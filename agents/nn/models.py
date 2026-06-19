import torch
import torch.nn as nn
import torch.nn.functional as F

# for 21x21 inputs (2)
# for 41x41 inputs (4)
OUT_DIM = {2: 4, 4: 14}
class Encoder(nn.Module):
    def __init__(self, z_dim=20, h_dim=256):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(2, 32, (3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.batch1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.conv4 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.batch2 = nn.BatchNorm2d(32)
        out_dim = OUT_DIM[2]
        self.fc = nn.Linear(32 * out_dim * out_dim, h_dim)
        self.fc1 = nn.Linear(h_dim, z_dim)

    def encoder(self, x):
        x = x.reshape(-1, 21, 21, 2).permute(0, 3, 1, 2)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.batch1(x)
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.batch2(x)
        x = torch.flatten(x, start_dim=1)
        x = F.elu(self.fc(x))
        x = self.fc1(x)
        return x

    def forward(self, x):
        return self.encoder(x)