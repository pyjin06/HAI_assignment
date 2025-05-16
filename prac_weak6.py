import torch
import torch.nn as nn
import torch.optim as optim

class Mymodule(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.seq = nn.Sequential(
            nn.Linear(17,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5,1)
        )
    
    def forward(self,x):
        x = self.flatten(x)
        y = self.seq(x)
        return y
    
myfirstmodule = Mymodule()
    
loss_fn = nn.MSELoss()
optimizor = optim.SGD(myfirstmodule.parameters(),lr=0.01)

epochs = 100

for epoch in range(1, epochs+1):
    inputs = torch.randn(1,17)
    labels = torch.randn(1,1)

    optimizor.zero_grad()

    outputs = myfirstmodule(inputs)

    loss = loss_fn(outputs, labels)

    loss.backward()

    optimizor.step()

    print(f"epoch = {epoch}, loss = {loss.item():.4f}")