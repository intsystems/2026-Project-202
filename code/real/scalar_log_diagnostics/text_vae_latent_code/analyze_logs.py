import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analyze import measure

BLUE, ORANGE, GREEN, GRAY, INK = "#0072B2", "#D55E00", "#009E73", "#7B858D", "#20262B"


def load_branch(arm, run_dir):
    if run_dir is None:
        logs = HERE / "logs"
        return (pd.read_csv(logs / f"vae_{arm}_log.csv", float_precision="round_trip"),
                pd.read_csv(logs / f"vae_{arm}_mi.csv"))
    branch = run_dir / ("freebits" if arm == "protected" else arm)
    return (pd.read_csv(branch / "logs.csv", float_precision="round_trip"),
            pd.read_csv(branch / "reference.csv"))


def style(ax,xlabel='',ylabel=''):
    ax.set_xlabel(xlabel,labelpad=7);ax.set_ylabel(ylabel,labelpad=7)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.tick_params(labelsize=21)


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.pdf")
    plt.close(fig)


def real_logs(run_dir):
    H=5.6                                   # taller figure; margins kept in inches
    fig,axs=plt.subplots(3,1,figsize=(11,H),sharex=True)
    arms=[('base',GRAY,'Base'),('regularized',ORANGE,'Code suppressed'),
          ('protected',GREEN,'Free bits')]
    bottom,top_band,gap,big_gap=.833,.833,.26,.43        # inches: x label, legend+zoom title, row gaps
    h=(H-bottom-top_band-gap-big_gap)/3.15; h0=1.15*h     # MG/MI rows and the taller NLL row
    y2=bottom/H; y1=(bottom+h+gap)/H; y0=(bottom+2*h+gap+big_gap)/H
    axs[2].set_position([.15,y2,.80,h/H]); axs[1].set_position([.15,y1,.80,h/H])
    axs[0].set_position([.15,y0,.43,h0/H])
    zoom=fig.add_axes([.71,y0,.20,h0/H])
    for arm,color,label in arms:
        log, mi = load_branch(arm, run_dir)
        assert np.array_equal(log.step.to_numpy(), np.arange(3072)) and mi.step.max() == 3072
        probe = log.probe_nll.to_numpy()
        ends = range(512, len(probe) + 1, 128)
        scores = [measure(probe[end - 512:end], 512) for end in ends]
        assert not any(score["degenerate"] for score in scores)
        axs[0].plot(log.step,log.probe_nll,color=color,lw=1.25,label=label)
        segment=log[log.step.between(1024,1408)]
        zoom.plot(segment.step,segment.probe_nll,color=color,lw=1.25)
        axs[1].plot(ends,[score["MG"] for score in scores],color=color,lw=1.8,marker='o',ms=3)
        axs[2].plot(mi.step,mi.MI,color=color,lw=1.8,marker='o',ms=3)
    for ax,ylabel in zip(axs,['Probe NLL','MG@probe NLL','MI']):
        style(ax,'',ylabel);ax.yaxis.label.set_size(21)
        ax.axvline(1024,color=INK,ls='--',lw=1.1)
        ax.set_xlim(0,3072);ax.set_xticks([0,1024,2048,3072])
    axs[1].yaxis.label.set_size(15)
    axs[1].axvspan(1024,1536,color=GRAY,alpha=.12,zorder=0)
    axs[0].set_yticks([4.5,7.5]);axs[0].set_ylim(4,8)
    axs[1].set_yticks([4,8]);axs[1].set_ylim(3,9)
    axs[2].set_yticks([0,5]);axs[2].set_ylim(-.25,6)
    axs[2].set_xlabel('Optimization step')
    fig.legend(handles=[Line2D([],[],color=c,lw=2,label=t) for _,c,t in arms],
               loc='upper center',bbox_to_anchor=(.55,1.0),ncol=3,frameon=False,fontsize=19)
    zoom.set_xlim(1024,1408);zoom.set_ylim(4.5,5.3)
    zoom.set_xticks([1024,1408]);zoom.set_yticks([4.6,5.2])
    zoom.tick_params(labelsize=17,pad=3)
    zoom.yaxis.tick_right()              # keep tick labels away from the inset connectors
    zoom.set_title('Zoom after branching',fontsize=19,pad=7,loc='center')
    for side in ('top','right','bottom','left'):
        zoom.spines[side].set_visible(True);zoom.spines[side].set_color(INK);zoom.spines[side].set_linewidth(.9)
    zoom.axvline(1024,color=INK,ls='--',lw=1.1)
    mark_inset(axs[0],zoom,loc1=2,loc2=3,fc='none',ec=INK,lw=.8)
    save(fig,'vae_real_log')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the VAE figure from bundled CSVs or a new run.")
    parser.add_argument("--output", type=Path, default=HERE.parents[2] / "runs" / "redrawn" / HERE.name)
    parser.add_argument("--run-dir", type=Path, help="Directory containing the base, regularized, and freebits runs.")
    args = parser.parse_args()
    OUTPUT = args.output
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 21, "axes.labelsize": 21,
                          "text.color": INK, "pdf.fonttype": 42})
    with threadpool_limits(limits=1):
        real_logs(args.run_dir)
    print(OUTPUT)
