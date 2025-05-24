import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])

train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN,self).__init__()
        self.conv_seq = nn.Sequential(
            nn.Conv2d(1,16),
            nn.ReLU(),

            nn.Conv2d(16, 32),
            nn.ReLU()
        )
        self.fc_seq = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    
    def forward(self, x):
        x = self.conv_seq(x)
        x = self.fc_seq(x)
        return x
    
model = MyCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

epochs = 5
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")

img = Image.open('shoesImage.png').convert('L')
to_tensor = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

model.eval()
x = to_tensor(img).unsqueeze(0)
with torch.no_grad():
    logits = model(x)
    probs = nn.functional.softmax(logits, dim=1)

print(probs)