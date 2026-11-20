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

from experiment_runner.plotting import options

BLUE, ORANGE, GREEN, GRAY, INK = "#0072B2", "#D55E00", "#009E73", "#7B858D", "#20262B"


def read(name):
    return pd.read_csv(HERE / "logs" / name)


def style(ax,xlabel='',ylabel=''):
    ax.set_xlabel(xlabel,labelpad=7);ax.set_ylabel(ylabel,labelpad=7)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.tick_params(labelsize=21)


def save(fig, name):
    fig.savefig(OUTPUT / f"{name}.pdf")
    plt.close(fig)


def synthetic():
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    fig.subplots_adjust(left=.19, right=.99, top=.90, bottom=.25)
    d=read('stationary_summary.csv')
    d=d.query('window==8192 and observer=="probe_error"')
    ax.plot(d.r,d['median'],'o-',color=BLUE,lw=2.2,ms=7,label='MG from one scalar')
    ax.plot([1,6],[1,6],ls='--',color=GRAY,lw=1.5,label='Known modes')
    ax.set_xticks([1,2,4,6]);style(ax,'Independent oscillatory modes','MG estimate')
    ax.set_ylim(.5,7.6);ax.set_yticks([2,4,6])   # headroom so the legend clears both lines
    ax.legend(frameon=False,fontsize=19,loc='upper left',borderaxespad=.2)
    save(fig,"known_linear_modes")


if __name__ == "__main__":
    OUTPUT = options(HERE)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 21, "axes.labelsize": 21,
                          "text.color": INK, "pdf.fonttype": 42})
    synthetic()
    print(OUTPUT)
