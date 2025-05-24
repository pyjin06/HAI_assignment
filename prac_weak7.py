import torch
import torch.nn as nn

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN,self).__init__()
        self.conv_seq = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64,3)
        )
    def forward(self, x):
        x = self.conv_seq(x)
        x = self.fc_seq(x)
        return x
model = MyCNN()
x = torch.randn(4,1,28,28)
y = model(x)
print(y.shape)    