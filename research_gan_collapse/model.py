"""Small convolutional Stacked-MNIST GAN and an independent digit classifier."""
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self,channels=3):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(64,64*7*7),nn.Unflatten(1,(64,7,7)),
            nn.BatchNorm2d(64),nn.ReLU(),nn.ConvTranspose2d(64,32,4,2,1,bias=False),
            nn.BatchNorm2d(32),nn.ReLU(),nn.ConvTranspose2d(32,channels,4,2,1),nn.Tanh())
    def forward(self,z):return self.net(z)

class Discriminator(nn.Module):
    def __init__(self,channels=3):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(channels,32,4,2,1),nn.LeakyReLU(.2),
            nn.Conv2d(32,64,4,2,1),nn.LeakyReLU(.2),nn.Flatten(),nn.Linear(64*7*7,1))
    def forward(self,x):return self.net(x).squeeze(1)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),
            nn.Linear(32*7*7,64),nn.ReLU())
        self.head=nn.Linear(64,10)
    def forward(self,x):return self.head(self.features(x))

def initialize(module):
    if isinstance(module,(nn.Linear,nn.Conv2d,nn.ConvTranspose2d)):
        nn.init.normal_(module.weight,0,.02)
        if module.bias is not None:nn.init.zeros_(module.bias)

def configure(threads=4):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
