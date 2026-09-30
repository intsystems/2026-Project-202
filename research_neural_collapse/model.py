"""Small CIFAR residual classifier; all convolutions and the head are trained."""
import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout))
        self.skip = (nn.Identity() if cin == cout and stride == 1 else
                     nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                                   nn.BatchNorm2d(cout)))

    def forward(self, x):
        return torch.relu(self.main(x) + self.skip(x))


class SmallResNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width), nn.ReLU(),
            Block(width, width), Block(width, 2*width, 2),
            Block(2*width, 4*width, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(4*width, 10, bias=False)

    def forward(self, x, return_features=False):
        h = self.features(x)
        z = self.head(h)
        return (z, h) if return_features else z


if __name__ == '__main__':
    import time
    import json
    torch.set_num_interop_threads(1)
    results = []
    for threads in [2, 4, 8]:
        torch.set_num_threads(threads)
        for batch in [64, 128]:
            model = SmallResNet().to(memory_format=torch.channels_last)
            x = torch.randn(batch, 3, 32, 32).to(memory_format=torch.channels_last)
            y = torch.randint(10, (batch,))
            opt = torch.optim.SGD(model.parameters(), lr=.03, momentum=.9)
            for i in range(10):
                t = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                nn.functional.cross_entropy(model(x), y).backward()
                opt.step()
                if i == 3: start = time.perf_counter()
            sec = (time.perf_counter()-start)/6
            results.append(dict(threads=threads, batch=batch, update_seconds=sec))
            print(results[-1], flush=True)
    print('Parameters:', sum(p.numel() for p in model.parameters()))
    from pathlib import Path
    Path(__file__).with_name('speed_probe.json').write_text(json.dumps(results, indent=2))
