"""Pilot (seed 99 only, never reused): training behaviour under constant LR and the shape of the
final-accuracy-vs-decay-time curve, to choose the condition ranges. Looks only at training
behaviour (losses, accuracies), never at MG or any competitor statistic."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_pilot"


def job(c):
    sys.path.insert(0, str(HERE))
    import lrrun
    r = lrrun.run_unit(c, 99, OUT, grid_override=[1000, 2000, 3000, 4000, 5000, 6000, 7000])
    acc = {k: round(v["test_acc"], 3) for k, v in r["branches"].items()}
    print(c["name"], "none", round(r["none"]["test_acc"], 3), "cos", round(r["cosine"]["test_acc"], 3),
          acc, f"trunk {r['t_trunk']:.0f}s br {r['t_branches']:.0f}s cos {r['t_cosine']:.0f}s", flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    conds = [dict(name=f"p_lr{lr}", lr=lr, bs=int(sys.argv[1]) if len(sys.argv) > 1 else 32, T=8000)
             for lr in (0.005, 0.02, 0.08)]
    with Pool(3) as p:
        p.map(job, conds)
