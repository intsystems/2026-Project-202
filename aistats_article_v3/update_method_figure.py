"""Illustration: all panels use the same fully sampled periodic record."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
H=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
t=np.arange(2048);period=40*np.sqrt(2);tau=10;k=8;T=tau
x=np.cos(2*np.pi*t/period);y=np.column_stack([x[:-tau],x[tau:]])
i=721;excluded=np.abs(np.arange(len(y))-i)<=T
distance=np.linalg.norm(y-y[i],axis=1)
nn=np.argsort(np.where(excluded,np.inf,distance))[:k];radius=distance[nn[-1]]
blue='#2563eb';orange='#d97706';gray='#7c858f'
fig,axs=plt.subplots(1,3,figsize=(3.25,1.85),gridspec_kw={'width_ratios':[1.1,1,1]})
fig.subplots_adjust(left=.14,right=.99,bottom=.41,top=.82,wspace=.58)
a,b,c=axs
a.plot(t[:160],x[:160],color=blue,lw=.7)
a.set(xlabel='Sample index $t$',ylabel='$x_t$',xticks=[0,150],yticks=[-1,1],title='(a) Scalar log')
b.scatter(y[:,0],y[:,1],s=1.5,color=blue,alpha=.4)
b.scatter(*y[excluded].T,s=10,color=gray,marker='x',linewidth=.5)
b.scatter(*y[i],s=22,facecolor='white',edgecolor='black',zorder=5)
b.set(xlabel='$x_t$',ylabel=r'$x_{t+\tau}$',xticks=[-1,1],yticks=[-1,1],title='(b) Delay vectors')
b.set_aspect('equal',adjustable='box')
c.scatter(*y[~excluded].T,s=5,color=blue,alpha=.55)
c.scatter(*y[excluded].T,s=15,color=gray,marker='x',linewidth=.6)
c.scatter(*y[nn].T,s=14,color=orange,zorder=4)
c.scatter(*y[i],s=22,facecolor='white',edgecolor='black',zorder=5)
c.add_patch(Circle(y[i],radius,fill=False,ls='--',color=orange,lw=.7))
c.plot([y[i,0],y[nn[-1],0]],[y[i,1],y[nn[-1],1]],color=orange,lw=.8)
c.set(xlim=(y[i,0]-1.5*radius,y[i,0]+1.5*radius),ylim=(y[i,1]-1.5*radius,y[i,1]+1.5*radius),
      xticks=[],yticks=[],title='(c) Neighbors',xlabel=r'Radius $r_{ik}$')
c.set_aspect('equal',adjustable='box')
for ax in axs:ax.set_title(ax.get_title(),fontsize=6.8,pad=7);ax.tick_params(labelsize=6,length=2,pad=1)
handles=[Line2D([],[],marker='.',ls='',color=blue,label='Admissible delay vectors'),
         Line2D([],[],marker='x',ls='',color=gray,label=r'Excluded: $|i-j|\leq T$'),
         Line2D([],[],marker='o',ls='',markerfacecolor='white',color='black',label='Query vector'),
         Line2D([],[],marker='.',ls='',color=orange,label=r'$k$ nearest vectors')]
fig.legend(handles=handles,loc='lower center',ncol=2,fontsize=6.2,frameon=False,
           columnspacing=.8,handletextpad=.2,bbox_to_anchor=(.5,.02))
fig.savefig(H/'figures/icomp_method.pdf');plt.close(fig)
(H/'method_figure_provenance.json').write_text(json.dumps(dict(kind='illustration, not experimental evidence',
    formula='x_t=cos(2*pi*t/(40*sqrt(2)))',samples=2048,tau=tau,E=2,k=k,Theiler=T,query=i),indent=2))
print('Method illustration: all panels use the same fully sampled periodic record.')
