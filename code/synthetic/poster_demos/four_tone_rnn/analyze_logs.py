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
    fig.subplots_adjust(left=.19, right=.99, top=.83, bottom=.25)
    g=read('runs.csv')
    for i,(arm,color) in enumerate([('H4',GRAY),('M4',ORANGE),('T4',BLUE)]):
        v=g[g.arm==arm].MG_n0.to_numpy();assert len(v)==5
        ax.scatter(i+np.linspace(-.11,.11,5),v,s=40,color=color)
        ax.plot([i-.20,i+.20],[np.median(v)]*2,color=INK,lw=2)
    ax.set_xticks([0,1,2],['1','2','4']);ax.set_ylim(0,8)
    ax.set_xlim(-.6,2.6)
    style(ax,'Independent target rhythms','One-neuron MG')
    ax.set_title('Four sinusoids in every target',fontsize=18,pad=12)
    save(fig,'four_tone_rnn')


if __name__ == "__main__":
    OUTPUT = options(HERE)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 21, "axes.labelsize": 21,
                          "text.color": INK, "pdf.fonttype": 42})
    synthetic()
    print(OUTPUT)
