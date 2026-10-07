"""Run a list of SSL configurations x seeds with 3 single-thread workers; skips finished runs.

usage: python run_grid.py <outdir> <grid_name> <seed> [<seed> ...]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DATA = None


def _init():
    global DATA
    import torch
    torch.set_num_threads(1)
    import ssl_core as S
    DATA = S.load_mnist()


def _job(args):
    name, cfg, seed, outdir = args
    import ssl_core as S
    out = Path(outdir) / f"{name}_s{seed}"
    if Path(str(out) + ".json").exists():
        return name, seed, "skip", 0.0
    t = time.time()
    try:
        _, meta = S.run(cfg, seed, DATA, out_path=out)
        last = meta["evals"][-1]
        msg = f"lin={last.get('lin_acc', float('nan')):.4f} knn={last.get('knn_acc', float('nan')):.4f} rk={last['rankme_h']:.1f}/{last['rankme_z']:.1f}"
    except Exception as e:  # keep the grid going
        msg = f"FAIL {e!r}"
    return name, seed, msg, time.time() - t


def main():
    outdir, grid, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
    import configs
    G = getattr(configs, grid)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    jobs = [(n, c, s, outdir) for s in seeds for n, c in G.items()]
    with Pool(int(os.environ.get("NWORK", 3)), initializer=_init) as p:
        for name, seed, msg, dt in p.imap_unordered(_job, jobs):
            print(f"{time.strftime('%H:%M:%S')} {name} s{seed} {dt:.0f}s {msg}", flush=True)


if __name__ == "__main__":
    main()
