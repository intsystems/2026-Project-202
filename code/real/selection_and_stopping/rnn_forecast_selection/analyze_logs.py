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
    fig=plt.figure(figsize=(11,4.0))
    ax=fig.add_axes([.32,.30,.64,.60])
    time=read('query_paired_summary.csv')
    for i,(key,label,color) in enumerate([('MG_q128','MG, 128 queries',BLUE),('validation','Held-out search',ORANGE),('MG','Full MG',GRAY)]):
        vals=1000*time[key].to_numpy();ax.scatter(vals,i+np.linspace(-.10,.10,len(vals)),color=color,s=18,alpha=.35)
        ax.scatter(np.median(vals),i,color=color,s=75,marker='D',zorder=4)
    ax.set_yticks(range(3),['MG, 128 queries','Held-out search','Full MG']);ax.set_ylim(2.5,-.5)
    ax.set_xlim(left=0);style(ax,'Selection + chosen fit (ms)')
    choice=read('query_fresh_comparisons.csv').pivot(index=['seed','arm'],columns='method',values='model')
    assert len(choice)==160 and (choice.MG==choice.MG_q128).all()
    fig.legend(handles=[Line2D([],[],color=GRAY,marker='o',ls='',alpha=.4,label='Input'),
                        Line2D([],[],color=GRAY,marker='D',ls='',label='Median')],
               loc='upper right',bbox_to_anchor=(.98,1.01),ncol=2,frameon=False,fontsize=19)
    save(fig,'forecast_selection')



if __name__ == "__main__":
    OUTPUT = options(HERE)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 21, "axes.labelsize": 21,
                          "text.color": INK, "pdf.fonttype": 42})
    decisions()
    print(OUTPUT)
