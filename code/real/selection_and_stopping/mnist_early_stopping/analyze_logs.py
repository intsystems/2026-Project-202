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


def decisions():
    fig=plt.figure(figsize=(11,4.6))
    ax=fig.add_axes([.13,.26,.83,.66])
    stop=read(f'decisions_10_17.csv').query('noise==0.4 and action=="stop" and method=="MG"')
    rows=[]
    for seed in range(10,18):
        g=read(f'seed{seed}_metrics.csv');rows.append(g)
        ax.plot(g.step,100*g.test_acc,color=GRAY,lw=1,alpha=.25)
    mean=pd.concat(rows).groupby('step').test_acc.mean()
    ax.plot(mean.index,100*mean,color=GRAY,lw=2,label='Continued training')
    ax.scatter(stop.step,100*stop.test_acc,color=BLUE,s=43,label=f'MG stop ({len(stop)} runs)',zorder=4)
    for step,group in stop.groupby('step'):
        ax.annotate(f'{len(group)} runs',xy=(step,100*group.test_acc.max()),
                    xytext=(step,97),ha='center',va='bottom',fontsize=19,color=BLUE,
                    arrowprops={'arrowstyle':'-','color':BLUE,'lw':1.2},zorder=5)
    ax.set_xlim(0,2048);ax.set_ylim(0,100);ax.set_xticks([0,512,1024,1536,2048])
    style(ax,'Optimization step','Test accuracy (%)')
    ax.legend(frameon=False,fontsize=20,loc='lower right')
    save(fig,'early_stopping')


if __name__ == "__main__":
    OUTPUT = options(HERE)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 21, "axes.labelsize": 21,
                          "text.color": INK, "pdf.fonttype": 42})
    decisions()
    print(OUTPUT)
