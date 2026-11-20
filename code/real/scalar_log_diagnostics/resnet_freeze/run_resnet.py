"""E7: the E5 setting on a real network -- ResNet-18 on full CIFAR-10 (Colab GPU).

See PROTOCOL.md for the pre-registered predictions. This script only trains and
logs; analyse_resnet.py scores the logs with the MG configuration and the online
detector frozen on the CPU experiments.

Scenarios
  scratch   ResNet-18 (CIFAR stem: 3x3 conv, no max-pool) from random init, 32x32.
  finetune  ImageNet-pretrained torchvision ResNet-18, CIFAR-10 upsampled to 64x64,
            all layers trained until the event (transfer learning).

Arms (event at --event, default step 4 000 of 10 000)
  base       nothing
  batch_up   batch 128 -> 512 (less gradient noise, the same parameters move)
  lr10       lr / 10
  lr100      lr / 100
  freeze_head  only the final linear layer keeps training
  prune80    global magnitude pruning of 80 % of conv/linear weights, mask kept
  prune95    the same at 95 %

Logged at every step: parameter norm (the primary observer), mini-batch loss, norms
of the stem, layer1-4 and fc; a 1 024-dim CountSketch of the parameter UPDATE, from
which the independent reference (participation ratio of the update covariance per
window) is computed. The full trajectory (11 M parameters x 10 000 steps) is never
stored.
"""
from __future__ import annotations

import argparse
import json
import pickle
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ARMS = ("base", "batch_up", "lr10", "lr100", "freeze_head", "prune80", "prune95")
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def load_cifar(root: Path):
    # Colab hung here silently once: a half-downloaded archive and no output for
    # 40 minutes. Every stage now says what it is doing, and the download reports
    # its progress, so a stall is visible at once.
    d = root / "cifar-10-batches-py"
    if not d.exists():
        root.mkdir(parents=True, exist_ok=True)
        tgz = root / "cifar-10-python.tar.gz"
        if not tgz.exists() or tgz.stat().st_size < 170_000_000:
            print(f"downloading CIFAR-10 to {tgz} ...", flush=True)
            last = [-1]

            def hook(blocks, bs, total):
                pct = int(100 * blocks * bs / max(total, 1))
                if pct // 10 != last[0]:
                    last[0] = pct // 10
                    print(f"   {min(pct, 100)} %", flush=True)
            urllib.request.urlretrieve(URL, tgz, reporthook=hook)
        print("extracting ...", flush=True)
        with tarfile.open(tgz) as t:
            t.extractall(root)

    def read(name):
        with open(d / name, "rb") as f:
            b = pickle.load(f, encoding="bytes")
        return b[b"data"].reshape(-1, 3, 32, 32), np.array(b[b"labels"])
    xs, ys = zip(*[read(f"data_batch_{i}") for i in range(1, 6)])
    xt, yt = read("test_batch")
    return np.concatenate(xs), np.concatenate(ys), xt, yt


def build(scenario: str, seed: int):
    torch.manual_seed(seed)
    if scenario == "scratch":
        net = torchvision.models.resnet18(num_classes=10)
        net.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        net.maxpool = nn.Identity()
        mean, std, size = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 32
    else:
        w = None if scenario == "finetune_noweights" else torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        net = torchvision.models.resnet18(weights=w)
        net.fc = nn.Linear(512, 10)
        mean, std, size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 64
    return net, torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1), size


def groups(net):
    out = {}
    for n, p in net.named_parameters():
        g = n.split(".")[0]
        g = g if g in ("layer1", "layer2", "layer3", "layer4", "fc") else "stem"
        out.setdefault(g, []).append(p)
    return out


def run(args, arm, seed, data):
    dev = torch.device(args.device)
    X, y, Xt, yt = data
    net, mean, std, size = build(args.scenario, seed)
    net = net.to(dev).to(memory_format=torch.channels_last)
    Xg = torch.tensor(X, dtype=torch.uint8, device=dev)
    yg = torch.tensor(y, device=dev)
    mean, std = mean.to(dev), std.to(dev)

    def prep(xb):
        xb = (xb.float() / 255 - mean) / std
        return F.interpolate(xb, size=size, mode="bilinear", align_corners=False) if size != 32 else xb

    params = list(net.parameters())
    P = sum(p.numel() for p in params)
    lr = args.lr
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(1000 + seed)
    bs, masks = args.batch, None
    # CountSketch of the update, hashes from NumPy so the torch stream is untouched
    h = np.random.default_rng(12345)
    sk_idx = torch.tensor(h.integers(0, 1024, size=P), device=dev)
    sk_sign = torch.tensor(h.integers(0, 2, size=P) * 2.0 - 1.0, device=dev, dtype=torch.float32)
    grp = groups(net)
    steps = args.steps
    logs = {"param_norm": np.empty(steps), "batch_loss": np.empty(steps)}
    for g in grp:
        logs[f"norm_{g}"] = np.empty(steps)
    upd = np.empty((steps, 1024), dtype=np.float32)
    prev = torch.cat([p.detach().reshape(-1).float() for p in params])
    scaler = torch.amp.GradScaler(enabled=(dev.type == "cuda"))
    t0 = time.perf_counter()
    for t in range(steps):
        if t == args.event:
            if arm.startswith("lr"):
                for gp in opt.param_groups:
                    gp["lr"] = lr / float(arm[2:])
            elif arm == "freeze_head":
                for n, p in net.named_parameters():
                    p.requires_grad_(n.startswith("fc."))
                opt = torch.optim.SGD(net.fc.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            elif arm == "batch_up":
                bs = args.batch * 4
            elif arm.startswith("prune"):
                frac = float(arm[5:]) / 100
                w = [p for p in params if p.dim() > 1]
                allw = torch.cat([p.detach().abs().reshape(-1) for p in w])
                k = int(frac * allw.numel())
                thr = allw.kthvalue(k).values
                masks = [(p.detach().abs() > thr).to(p.dtype) for p in w]
                with torch.no_grad():
                    for p, m in zip(w, masks):
                        p.mul_(m)
        net.train()
        idx = torch.as_tensor(rng.integers(0, len(X), bs), device=dev)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=(dev.type == "cuda")):
            loss = lossf(net(prep(Xg[idx]).contiguous(memory_format=torch.channels_last)), yg[idx])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if masks is not None:
            with torch.no_grad():
                for p, m in zip([p for p in params if p.dim() > 1], masks):
                    p.mul_(m)
        with torch.no_grad():
            cur = torch.cat([p.detach().reshape(-1).float() for p in params])
            d = cur - prev
            prev = cur
            upd[t] = torch.zeros(1024, device=dev).index_add_(0, sk_idx, sk_sign * d).cpu().numpy()
            logs["param_norm"][t] = cur.norm().item()
            logs["batch_loss"][t] = loss.item()
            for g, ps in grp.items():
                logs[f"norm_{g}"][t] = torch.sqrt(sum((p.detach().float() ** 2).sum() for p in ps)).item()
        if args.verbose and t % 1000 == 0:
            print(f"   step {t} loss {loss.item():.3f} {time.perf_counter() - t0:.0f}s", flush=True)
    train_s = time.perf_counter() - t0
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(Xt), 500):
            xb = torch.tensor(Xt[i:i + 500], device=dev)
            correct += (net(prep(xb)).argmax(1).cpu().numpy() == yt[i:i + 500]).sum()
    return logs, upd, {"P": P, "test_acc": float(correct / len(Xt)), "train_s": train_s,
                       "final_loss": float(logs["batch_loss"][-200:].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["scratch", "finetune", "finetune_noweights"], default="scratch")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[20, 21, 22])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--event", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results_resnet")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) / args.scenario
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {args.device}", flush=True)
    if args.device == "cpu":
        print("WARNING: no GPU -- this will take about a day. Runtime -> Change runtime type -> T4 GPU.",
              flush=True)
    data = load_cifar(Path(args.data))
    print(f"CIFAR-10 loaded: {len(data[0])} train, {len(data[2])} test", flush=True)
    for seed in args.seeds:
        for arm in args.arms:
            f = out / f"logs_{arm}_s{seed}.npz"
            if f.exists():
                print("skip", f.name); continue
            logs, upd, meta = run(args, arm, seed, data)
            np.savez_compressed(f, update_sketch=upd, **logs)
            json.dump({**vars(args), "arm": arm, "seed": seed, **meta},
                      open(out / f"meta_{arm}_s{seed}.json", "w"), indent=1)
            print(f"{args.scenario} {arm:12s} s{seed} test {meta['test_acc']:.3f} "
                  f"{meta['train_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
