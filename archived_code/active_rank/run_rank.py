"""Train a registered grokking run while recording the parameter-trajectory sketch.

Reuses ``grokking_train``'s loop unchanged -- the configs in ``runs.py`` are the ones that
produced the published logs -- and attaches :class:`rank_probe.RankProbe`.

    python run_rank.py mod_wd1 --outdir /content/out
    python run_rank.py mod_wd0 --outdir /content/out --tag mod_wd0_s1 --set seed=43

Writes two files per run into ``--outdir``:

    <csv_name>            the standard training log, unchanged
    <tag>_rank.npz        the sketched parameter and function trajectories
"""

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "grokking_train"))

import runs                                                 # noqa: E402
from grok import train                                      # noqa: E402
from rank_probe import RankProbe                            # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="a key from grokking_train/runs.py, e.g. mod_wd1")
    p.add_argument("--outdir", default="rank_logs")
    p.add_argument("--tag", default=None,
                   help="name the outputs '<tag>_train.csv' / '<tag>_rank.npz'. Required "
                        "when several variants of one run key are trained, since they "
                        "otherwise overwrite each other.")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--n-sketch", type=int, default=2)
    p.add_argument("--n-probe", type=int, default=256)
    p.add_argument("--probe-seed", type=int, default=0)
    p.add_argument("--source", default="train", choices=("train", "val"))
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--device", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="override a RunConfig field (repeatable)")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)

    overrides = {}
    for pair in a.overrides:
        k, v = pair.split("=", 1)
        overrides[k.strip()] = v.strip()
    if a.device:
        overrides["device"] = a.device

    name = a.tag or a.run
    if a.tag:
        overrides.setdefault("csv", f"{a.tag}_train.csv")

    config = runs.get(a.run)
    if overrides:
        config = config.with_overrides(overrides)

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    probe = RankProbe(dim=a.dim, n_sketch=a.n_sketch, n_probe=a.n_probe,
                      seed=a.probe_seed, source=a.source,
                      progress_every=a.progress_every)

    t0 = time.perf_counter()
    print(f"[{name}] {config.description}", flush=True)
    frame, path = train(config, outdir=outdir, progress=False,
                        overwrite=a.force, observer=probe)
    npz = probe.save(outdir / f"{name}_rank.npz")
    print(f"[{name}] {len(frame)} logged rows, {probe.n_params} parameters, "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[{name}] -> {path}", flush=True)
    print(f"[{name}] -> {npz}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
