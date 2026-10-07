"""Joint-embedding SSL on MNIST with a small MLP encoder (CPU, 1 thread).

One run = one configuration x one seed. Per training step it logs cheap scalars
(SSL loss, parameter norm, gradient norm, embedding norm of one fixed probe image); every
`eval_every` steps it evaluates the label-free proxies used in the literature (RankMe,
alpha-ReQ, LiDAR, SimSiam output-std) on fixed unlabeled images, and every `probe_every`
steps the ground truth (linear-probe and kNN accuracy with labels).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
RAW = HERE.parents[1] / "research_gan_collapse" / "data" / "MNIST" / "raw"


def load_mnist():
    def imgs(name):
        return np.fromfile(RAW / name, np.uint8, offset=16).reshape(-1, 1, 28, 28)

    def labs(name):
        return np.fromfile(RAW / name, np.uint8, offset=8).astype(np.int64)
    return (torch.from_numpy(imgs("train-images-idx3-ubyte")), labs("train-labels-idx1-ubyte"),
            torch.from_numpy(imgs("t10k-images-idx3-ubyte")), labs("t10k-labels-idx1-ubyte"))


# ---- model ------------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self, width=512, rep=256, proj_hidden=256, emb=64, predictor=True,
                 pred_hidden=64, proj_bn=True):
        super().__init__()
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(784, width), nn.BatchNorm1d(width), nn.ReLU(),
                                      nn.Linear(width, rep), nn.BatchNorm1d(rep), nn.ReLU())
        pb = [nn.BatchNorm1d(proj_hidden)] if proj_bn else []
        self.projector = nn.Sequential(nn.Linear(rep, proj_hidden), *pb, nn.ReLU(), nn.Linear(proj_hidden, emb))
        self.predictor = (nn.Sequential(nn.Linear(emb, pred_hidden), nn.BatchNorm1d(pred_hidden), nn.ReLU(),
                                        nn.Linear(pred_hidden, emb)) if predictor else nn.Identity())

    def forward(self, x):
        h = self.backbone(x)
        z = self.projector(h)
        return h, z, self.predictor(z)


# ---- augmentations ----------------------------------------------------------------------
# (translation px, rotation deg, scale +-, erase prob, erase max side frac, noise sd, pixel dropout)
AUG = {"weak": (2, 10, 0.05, 0.0, 0.0, 0.05, 0.0),
       "medium": (4, 20, 0.15, 0.5, 0.5, 0.10, 0.1),
       "strong": (6, 30, 0.25, 0.8, 0.7, 0.25, 0.3)}


def augment(x, strength, g):
    t, rot, sc, ep, es, noise, drop = AUG[strength]
    B = x.shape[0]
    u = lambda: torch.rand(B, generator=g) * 2 - 1  # noqa: E731
    ang = u() * rot * math.pi / 180
    s = 1 + u() * sc
    tx, ty = u() * t / 14, u() * t / 14
    c, sn = torch.cos(ang) / s, torch.sin(ang) / s
    theta = torch.stack([torch.stack([c, -sn, tx], 1), torch.stack([sn, c, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
    if ep > 0:
        ii = torch.arange(28).float()
        cx, cy = torch.rand(B, generator=g) * 28, torch.rand(B, generator=g) * 28
        hw = torch.rand(B, generator=g) * es * 14 + 2
        hh = torch.rand(B, generator=g) * es * 14 + 2
        m = ((ii[None, :, None] - cy[:, None, None]).abs() < hh[:, None, None]) & \
            ((ii[None, None, :] - cx[:, None, None]).abs() < hw[:, None, None])
        on = (torch.rand(B, generator=g) < ep)[:, None, None]
        x = x * (~(m & on)).unsqueeze(1).float()
    if drop > 0:
        x = x * (torch.rand(x.shape, generator=g) >= drop).float()
    if noise > 0:
        x = x + noise * torch.randn(x.shape, generator=g)
    return x


# ---- losses -----------------------------------------------------------------------------

def neg_cos(p, z):
    return -F.cosine_similarity(p, z, dim=-1).mean()


def vicreg_terms(z1, z2):
    inv = F.mse_loss(z1, z2)
    std = lambda z: torch.sqrt(z.var(0) + 1e-4)  # noqa: E731
    var = 0.5 * (F.relu(1 - std(z1)).mean() + F.relu(1 - std(z2)).mean())

    def cov(z):
        z = z - z.mean(0)
        c = (z.T @ z) / (len(z) - 1)
        off = c - torch.diag(torch.diag(c))
        return (off ** 2).sum() / z.shape[1]
    return inv, var, 0.5 * (cov(z1) + cov(z2))


def nt_xent(z1, z2, temp):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    s = z @ z.T / temp
    n = len(z1)
    s.fill_diagonal_(-1e9)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
    return F.cross_entropy(s, tgt)


# ---- label-free proxies -----------------------------------------------------------------

def rankme(Z, eps=1e-7):
    s = np.linalg.svd(np.asarray(Z, np.float64), compute_uv=False)
    p = s / (s.sum() + eps) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def alpha_req(H, fit=(1, 128)):
    H = np.asarray(H, np.float64)
    ev = np.sort(np.clip(np.linalg.eigvalsh(np.cov(H.T)), 0, None))[::-1]
    lo, hi = fit
    hi = min(hi, len(ev))
    i = np.arange(lo, hi + 1)
    e = ev[lo - 1:hi]
    ok = e > 1e-12 * max(ev[0], 1e-30)
    if ok.sum() < 3:
        return float("nan")
    return float(-np.polyfit(np.log(i[ok]), np.log(e[ok]), 1)[0])


def lidar(Zq, delta=1e-4, eps=1e-8):
    """Thilak et al. 2024. Zq: (n, q, d) embeddings of q augmentations of n samples."""
    Zq = np.asarray(Zq, np.float64)
    n, q, d = Zq.shape
    mu_c = Zq.mean(1)
    mu = mu_c.mean(0)
    Sb = np.cov((mu_c - mu).T, bias=False)
    W = (Zq - mu_c[:, None, :]).reshape(-1, d)
    Sw = W.T @ W / (n * (q - 1)) + delta * np.eye(d)
    ew, Uw = np.linalg.eigh(Sw)
    Wi = Uw @ np.diag(ew ** -0.5) @ Uw.T
    lam = np.clip(np.linalg.eigvalsh(Wi @ Sb @ Wi), 0, None)
    p = lam / (lam.sum() + eps) + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def out_std(Z):
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return float(Zn.std(0).mean() * np.sqrt(Z.shape[1]))


def probe_scores(Htr, ytr, Hte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Htr)
    a, b = sc.transform(Htr), sc.transform(Hte)
    a, b = np.nan_to_num(a), np.nan_to_num(b)
    clf = LogisticRegression(max_iter=300, C=0.1)
    clf.fit(a, ytr)
    lin = float((clf.predict(b) == yte).mean())
    an = Htr / (np.linalg.norm(Htr, axis=1, keepdims=True) + 1e-12)
    bn = Hte / (np.linalg.norm(Hte, axis=1, keepdims=True) + 1e-12)
    sim = bn @ an.T
    idx = np.argpartition(-sim, 20, axis=1)[:, :20]
    votes = ytr[idx]
    knn = np.array([np.bincount(v, minlength=10).argmax() for v in votes])
    return lin, float((knn == yte).mean())


# ---- training ---------------------------------------------------------------------------

DEFAULT = dict(method="simsiam", steps=4000, bs=128, width=256, lr=0.05, wd=1e-4, momentum=0.9, warmup=100,
               aug="medium", emb=64, proj_hidden=256, predictor=True, stopgrad=True, proj_bn=True,
               vic_inv=25.0, vic_var=25.0, vic_cov=1.0, temp=0.2, ema=0.99,
               eval_every=250, probe_every=500, n_eval=2048, n_lidar=256, q_lidar=8,
               n_probe_tr=5000, n_probe_te=5000)


def embed(net, X, bs=1024):
    net.eval()
    hs, zs = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            h, z, _ = net(X[i:i + bs].float() / 255)
            hs.append(h); zs.append(z)
    net.train()
    return torch.cat(hs).numpy(), torch.cat(zs).numpy()


def run(cfg, seed, data, out_path=None, verbose=False):
    c = dict(DEFAULT); c.update(cfg)
    Xtr, ytr, Xte, yte = data
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    rs = np.random.RandomState(1234)                  # fixed eval/probe subsets across runs
    perm = rs.permutation(len(Xtr))
    ev_idx, pr_idx = perm[:c["n_eval"]], perm[c["n_eval"]:c["n_eval"] + c["n_probe_tr"]]
    te_idx = rs.permutation(len(Xte))[:c["n_probe_te"]]
    probe_img = Xtr[perm[-1:]].float() / 255          # one fixed probe image, never trained on specially
    Xev, Xpr, Xpt = Xtr[ev_idx], Xtr[pr_idx], Xte[te_idx]
    ypr, ypt = ytr[pr_idx], yte[te_idx]

    net = Net(width=c["width"], emb=c["emb"], proj_hidden=c["proj_hidden"],
              predictor=c["predictor"] and c["method"] in ("simsiam", "byol"), proj_bn=c["proj_bn"])
    target = copy.deepcopy(net) if c["method"] == "byol" else None
    if target is not None:
        for p in target.parameters():
            p.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=c["lr"], momentum=c["momentum"], weight_decay=c["wd"])

    T = c["steps"]
    logs = {k: np.zeros(T, np.float32) for k in
            ("loss", "param_norm", "grad_norm", "probe_z_norm", "probe_h_norm")}
    evals = []
    t_train = t_eval = t_probe = 0.0
    cost = {"rankme": 0.0, "alpha": 0.0, "lidar": 0.0, "embed": 0.0}
    n = len(Xtr)

    def evaluate(step, with_probe):
        nonlocal t_eval, t_probe
        t0 = time.perf_counter()
        H, Z = embed(net, Xev)
        t1 = time.perf_counter(); cost["embed"] += t1 - t0
        r = {"step": step, "rankme_h": rankme(H), "rankme_z": rankme(Z)}
        t2 = time.perf_counter(); cost["rankme"] += t2 - t1
        r["alpha_h"] = alpha_req(H)
        t3 = time.perf_counter(); cost["alpha"] += t3 - t2
        gl = torch.Generator().manual_seed(777)
        xs = Xev[:c["n_lidar"]].float() / 255
        net.eval()
        with torch.no_grad():
            Zq = torch.stack([net(augment(xs, c["aug"], gl))[1] for _ in range(c["q_lidar"])], 1).numpy()
        net.train()
        r["lidar_z"] = lidar(Zq)
        t4 = time.perf_counter(); cost["lidar"] += t4 - t3
        r["out_std"] = out_std(Z)
        r["h_dead"] = float((H.std(0) < 1e-6).mean())
        t_eval += time.perf_counter() - t0
        if with_probe:
            t0 = time.perf_counter()
            Htr, _ = embed(net, Xpr)
            Hte, _ = embed(net, Xpt)
            r["lin_acc"], r["knn_acc"] = probe_scores(Htr, ypr, Hte, ypt)
            t_probe += time.perf_counter() - t0
        evals.append(r)
        if verbose:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    evaluate(0, True)
    for step in range(T):
        t0 = time.perf_counter()
        lr = c["lr"] * min(1.0, (step + 1) / c["warmup"])
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = torch.randint(0, n, (c["bs"],), generator=g)
        x = Xtr[idx].float() / 255
        x1, x2 = augment(x, c["aug"], g), augment(x, c["aug"], g)
        h1, z1, p1 = net(x1)
        h2, z2, p2 = net(x2)
        m = c["method"]
        if m == "simsiam":
            sg = (lambda v: v.detach()) if c["stopgrad"] else (lambda v: v)
            loss = 0.5 * (neg_cos(p1, sg(z2)) + neg_cos(p2, sg(z1)))
        elif m == "byol":
            with torch.no_grad():
                _, t1, _ = target(x1)
                _, t2, _ = target(x2)
            loss = 0.5 * ((2 - 2 * F.cosine_similarity(p1, t2, dim=-1)).mean() +
                          (2 - 2 * F.cosine_similarity(p2, t1, dim=-1)).mean())
        elif m == "vicreg":
            inv, var, cov = vicreg_terms(z1, z2)
            loss = c["vic_inv"] * inv + c["vic_var"] * var + c["vic_cov"] * cov
        elif m == "simclr":
            loss = nt_xent(z1, z2, c["temp"])
        else:
            raise ValueError(m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            gn = torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None))
        opt.step()
        if target is not None:
            with torch.no_grad():
                for pt, po in zip(target.parameters(), net.parameters()):
                    pt.mul_(c["ema"]).add_(po.detach(), alpha=1 - c["ema"])
                for bt, bo in zip(target.buffers(), net.buffers()):
                    bt.copy_(bo)
        with torch.no_grad():
            pn = torch.sqrt(sum((p ** 2).sum() for p in params))
            net.eval()
            hp, zp, _ = net(probe_img)
            net.train()
        logs["loss"][step] = loss.item()
        logs["param_norm"][step] = pn.item()
        logs["grad_norm"][step] = gn.item()
        logs["probe_z_norm"][step] = zp.norm().item()
        logs["probe_h_norm"][step] = hp.norm().item()
        t_train += time.perf_counter() - t0
        if (step + 1) % c["eval_every"] == 0:
            evaluate(step + 1, (step + 1) % c["probe_every"] == 0 or step + 1 == T)
    meta = {"cfg": c, "seed": seed, "t_train": t_train, "t_eval": t_eval, "t_probe": t_probe,
            "cost": cost, "evals": evals}
    if out_path is not None:
        np.savez_compressed(str(out_path) + ".npz", **logs)
        json.dump(meta, open(str(out_path) + ".json", "w"), default=float)
    return logs, meta
