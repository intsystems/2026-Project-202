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


def real_logs():
    fig,axs=plt.subplots(2,2,figsize=(11,4.3),sharex=True,sharey='row')
    fig.subplots_adjust(left=.15,right=.96,top=.81,bottom=.21,hspace=.45,wspace=.15)
    mg=read(f'force_rolling_mg.csv')
    assert len(mg)==18 and not mg.degenerate.any()
    for col,(checkpoint,color,title) in enumerate([(0,ORANGE,'Before learning'),(40000,BLUE,'After learning')]):
        log=read(f'force_checkpoint_{checkpoint:05d}_log.csv')
        g=mg[mg.checkpoint==checkpoint]
        axs[0,col].plot(log.evaluation_time,log.neuron_0_activity,color=color,lw=.9)
        axs[1,col].plot(g.time,g.MG,color=color,lw=2,marker='o',ms=4)
        axs[0,col].set_title(title,fontsize=23,pad=9)
        axs[1,col].axhline(1,color=GRAY,ls='--',lw=1)
        axs[1,col].axvspan(0,409.6,color=GRAY,alpha=.1,zorder=0)
        for row in range(2):
            style(axs[row,col],'Evaluation time' if row else '',
                  ('Activity' if row==0 else 'MG estimate') if col==0 else '')
            axs[row,col].set_xlim(0,820);axs[row,col].set_xticks([0,400,800])
        axs[0,col].set_ylim(-1.05,1.05);axs[0,col].set_yticks([-1,0,1])
        axs[1,col].set_ylim(0,7);axs[1,col].set_yticks([0,2,4,6])
    axs[1,1].text(420,2.0,'Cycle: d = 1',fontsize=19,color=INK)
    fig.text(.13,.94,'Fixed weights; shading: collecting the first MG window.',fontsize=19,color=INK)
    save(fig,'rnn_real_log')


if __name__ == "__main__":
    OUTPUT = options(HERE)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 21, "axes.labelsize": 21,
                          "text.color": INK, "pdf.fonttype": 42})
    real_logs()
    print(OUTPUT)
