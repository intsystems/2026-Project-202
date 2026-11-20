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


def resnet():
    fig,axs=plt.subplots(2,2,figsize=(11,6.3),sharex=True)
    fig.subplots_adjust(left=.14,right=.97,top=.79,bottom=.18,hspace=.65,wspace=.30)
    for i,(mode,title) in enumerate([('scratch','Random initialization'),('finetune','ImageNet-pretrained initialization')]):
        norms = read(f'{mode}_param_norm.csv')
        raw = {arm: norms[f'{arm}_s20'].to_numpy() for arm in ['base', 'freeze_head']}
        w=read(f'{mode}_windows.csv')
        steps=np.arange(10000)
        base=w[(w.arm=='base')&(w.seed==20)]
        frz=w[(w.arm=='freeze_head')&(w.seed==20)]
        # Before step 4,000 both runs follow the same trajectory: draw it in gray only.
        axs[i,0].plot(steps,raw['base'],color=GRAY,lw=1.9)
        axs[i,0].plot(steps[:4001],raw['freeze_head'][:4001],color=GRAY,lw=1.9)
        axs[i,0].plot(steps[4000:],raw['freeze_head'][4000:],color=BLUE,lw=1.9)
        axs[i,1].plot(base.end,base.MG,color=GRAY,lw=2.3,marker='o',ms=3.4)
        pre=frz[frz.end<=4000]; post=frz[frz.end>=4000]   # windows ending by 4,000 contain no frozen steps
        axs[i,1].plot(pre.end,pre.MG,color=GRAY,lw=2.3,marker='o',ms=3.4)
        axs[i,1].plot(post.end,post.MG,color=BLUE,lw=2.3,marker='o',ms=3.4,markevery=list(range(1,len(post))))
        for j in range(2):
            ax=axs[i,j];ax.axvline(4000,color=ORANGE,ls='--',lw=1.5)
            ax.set_xlim(0,10000);ax.set_xticks([0,4000,8000])
            style(ax,'Optimization step' if i else '', 'Weight norm' if j==0 else 'MG estimate')
        axs[i,1].set_ylim(0,6)
        event=w[(w.arm=='freeze_head')&(w.seed==20)&(w.end==5000)]
        assert len(event)==1
        value=float(event.MG.iloc[0])
        axs[i,1].scatter([5000],[value],s=95,facecolors='white',edgecolors=BLUE,lw=1.8,zorder=5)
        axs[i,1].annotate('Detected',xy=(5000,value),xytext=(6300,5.25),fontsize=19,color=BLUE,
                          arrowprops={'arrowstyle':'->','color':BLUE,'lw':1.1})
    axs[0,0].set_title('Random initialization',loc='left',fontsize=19,pad=12)
    axs[1,0].set_title('ImageNet-pretrained initialization',loc='left',fontsize=19,pad=12)
    axs[0,1].set_title('MG from the same log',loc='left',fontsize=19,pad=12)
    axs[1,1].set_title('MG from the same log',loc='left',fontsize=19,pad=12)
    fig.legend(handles=[Line2D([],[],color=GRAY,lw=2,label='Unchanged'),
                        Line2D([],[],color=BLUE,lw=2,label='Backbone frozen'),
                        Line2D([],[],color=ORANGE,ls='--',lw=1.5,label='Freeze: step 4,000')],
               loc='upper center',bbox_to_anchor=(.54,1.005),ncol=3,frameon=False,fontsize=19)
    det=pd.concat([read(f'{mode}_detector_per_run.csv') for mode in ['scratch','finetune']])
    det=det[det.stat=='MG'];assert det[det.arm=='freeze_head'].hit.sum()==6
    null=det[det.arm.isin(['base','batch_up','scale','smooth'])]
    assert len(null)==24 and not null.false_alarm.any()
    save(fig,'resnet')


if __name__ == "__main__":
    OUTPUT = options(HERE)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 21, "axes.labelsize": 21,
                          "text.color": INK, "pdf.fonttype": 42})
    resnet()
    print(OUTPUT)
