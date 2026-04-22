import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([
    [0.,0.],
    [0.,1.],
    [1.,0.],
    [1.,1.]
])

y = torch.tensor([
    [1.],
    [0.],
    [0.],
    [1.]
])

class XNORNet(nn.Module):
    def __init__(self, hidden=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,hidden),
            nn.Sigmoid(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)
    
model = XNORNet(4)