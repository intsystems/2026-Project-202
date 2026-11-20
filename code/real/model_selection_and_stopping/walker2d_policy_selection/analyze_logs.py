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
    fig=plt.figure(figsize=(11,3.3))
    ax=fig.add_axes([.235,.34,.725,.55])
    rl=read('checkpoint_selections.csv').query('seed>=232')
    labels=['Nominal reward','MG','Entropy','Fixed policy']
    for i,(key,color) in enumerate([('reward',GRAY),('MG',BLUE),('entropy',GREEN),('fixed_pilot',ORANGE)]):
        g=rl[rl.method==key];assert len(g)==4
        ax.plot([0,g.reward.mean()],[i,i],color=color,lw=4,solid_capstyle='butt')
        ax.scatter(g.reward,i+np.linspace(-.09,.09,len(g)),color=color,s=22,alpha=.4)
        ax.scatter(g.reward.mean(),i,color=color,s=120,marker='D',zorder=4,edgecolors='white',linewidths=1.2)
    ax.set_yticks(range(4),labels);ax.set_ylim(3.5,-.5);ax.set_xlim(0,5.2)
    style(ax,'Policy performance after transfer\nEnvironment with actuator lag and noise')
    fig.legend(handles=[Line2D([],[],color=GRAY,marker='o',ls='',alpha=.4,label='Run'),
                        Line2D([],[],color=GRAY,marker='D',ls='',label='Mean')],
               loc='upper right',bbox_to_anchor=(.98,1.01),ncol=2,frameon=False,fontsize=19)
    save(fig,'policy_selection')



if __name__ == "__main__":
    OUTPUT = options(HERE)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 21, "axes.labelsize": 21,
                          "text.color": INK, "pdf.fonttype": 42})
    decisions()
    print(OUTPUT)
