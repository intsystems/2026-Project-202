"""One-off editorial revision; no experiment outcomes are modified."""
from pathlib import Path
H=Path(__file__).resolve().parent
def write(name,text): (H/name).write_text(text.strip()+'\n',encoding='utf-8')
def edit(name,old,new):
    p=H/name;s=p.read_text(encoding='utf-8-sig')
    assert old in s,(name,old[:60])
    p.write_text(s.replace(old,new),encoding='utf-8')

write('sections/abstract.tex',r'''
\begin{abstract}
\claude{Training can restrict parameter motion, suppress latent information, or produce more regular behavior. We develop a method for monitoring these changes from one scalar log, combining delay reconstruction with pooled local-dimension estimation. Dynamical-systems theory motivates an active-component interpretation, which we test using known oscillations and learned recurrent generators. In applications, MG detects training restrictions and latent information loss, and supports policy and forecast selection. A new self-supervised learning study shows that gradient-norm MG ranks 29 configurations with mean Spearman correlation 0.712 over five test seeds, exceeding several representation-based proxies while remaining comparable to simpler scalar statistics. A sampled-query implementation preserves all 160 full-MG forecast choices on a separate cohort at lower measured cost. The method provides a geometric view of training from scalar data; absolute dimension recovery requires suitable recurrence and observation conditions, while practical decisions are checked against task-specific references.}
\end{abstract}
''')
write('sections/introduction.tex',r'''
\claude{
\section{Introduction}\label{sec:intro}
A training loss can decrease while the learned representation loses useful information. A recurrent network can reproduce its target more accurately while its activity settles onto a simpler cycle. Both changes matter, but measuring them directly may require saving activations, analyzing parameter trajectories, or propagating tangent dynamics. We ask how much of this information can be obtained from the scalar logs already available during training.

Our starting point is delay-embedding theory: under suitable assumptions, delayed observations retain the geometry of the underlying dynamics \citep{takens1981detecting,sauer1991embedology,stark1999delay,stark2003delay}. We apply the pooled MacKay--Ghahramani (MG) form \citep{mackay2005comments} of the Levina--Bickel estimator \citep{levina2004maximum} to these observations. In recurrent systems, local geometry can reveal how many independent components are active. In nonstationary training, we instead test whether the same statistic tracks changes measured independently.

\paragraph{Contributions.}
We make this connection usable in three ways.
\begin{enumerate}[leftmargin=*,itemsep=2pt,topsep=3pt]
\item \textbf{Dimension estimation from a scalar record.} We specify the observable, delay coordinates, temporal exclusion, numerical checks, and interpretation conditions. Synthetic oscillators and learning experiments test the active-component interpretation; single-neuron records distinguish independent phases from harmonics.
\item \textbf{Monitoring and model selection in learned systems.} We compare MG with independent measurements in CNNs, text VAEs, and reinforcement learning. Gradient-norm MG also ranks self-supervised learning configurations without accessing embeddings or labels on the evaluated runs. We retain the strongest scalar alternatives in each comparison.
\item \textbf{Low data requirements and measured computational savings.} Analysis depends on the scalar window rather than the network's parameter dimension. In the specified recurrent benchmark, MG is 14--19 times faster than the full Lyapunov spectrum. Using 128 queries preserves all 160 full-MG forecast choices on 20 new seeds with a 16.3-fold speedup over full MG.
\end{enumerate}
Our contribution is the measurement procedure and its empirical evaluation; delay embeddings and local dimension estimators are established tools. The experiments distinguish recovery of known structure, detection of independently measured changes, and downstream decisions.

\paragraph{Relation to existing measurements.}
Intrinsic-dimension methods analyze point clouds and representations \citep{facco2017estimating,ansuini2019intrinsic,pope2021intrinsic}; other work studies fractal dimensions of optimizer trajectories and invariant measures \citep{simsekli2020hausdorff,camuto2021fractal,birdal2021intrinsic,tan2024limitations}. Random-walk PCA \citep{antognini2018pca} and Koopman analysis \citep{redman2024equivalent} help interpret training trajectories. We measure local temporal geometry from one scalar, comparing it with participation ratio, Hessian spectra, and grokking-related changes in function complexity and weight rank \citep{gao2017theory,cohen2021gradient,liu2022omnigrok,yunis2024rank,humayun2024deep}. Roughness, recurrence, and entropy test whether this geometry adds useful information beyond simpler features.
}
''')
p=H/'sections/method.tex';s=p.read_text(encoding='utf-8-sig')
start=s.index('Given one scalar');end=s.index(r'\subsection{Delay reconstruction')
s=s[:start]+r'''Given a scalar observation $x_t=\phi(s_t)$ of a state $s_t$, we seek changes in the complexity of its dynamics. During training, $s_t$ contains the $P$ trainable parameters and any optimizer variables. For a trained recurrent network with fixed weights, $s_t$ is its activity. These are distinct trajectories, observed through the same measurement procedure.

In a recurrent regime, let $\mu$ describe the long-run fraction of time spent in state-space regions: this is the occupation measure. We assume that this limiting measure exists and is ergodic; finite records do not prove either property. For a Euclidean ball $B(s,\epsilon)$ in a fixed state representation, define
\begin{equation}
d_{\rm act}=\lim_{\epsilon\to0^+}
\frac{\log\mu(B(s,\epsilon))}{\log\epsilon},
\label{eq:active}
\end{equation}
when the limit exists and is constant $\mu$-almost everywhere. Here $\epsilon$ is a positive radius. On a smooth manifold with positive regular density, this exponent equals the manifold dimension; $r$ independent oscillator phases therefore require $r$ local coordinates. Fixed invertible linear rescaling preserves the limit, although finite-scale estimates can change.

During nonstationary training we use a different target: a change in MG that agrees with restricted updates, reduced latent information, or more regular behavior measured independently. Such simplification need not improve performance: losing useful latent information is an undesirable change that a monitor should detect.

''' +s[end:]
start=s.index(r'\subsection{Measurement and interpretation pipeline}')
s=s[:start]+r'''\subsection{Measurement and interpretation pipeline}\label{sec:mg_conditions}
Applying the estimator requires an observable and a timescale. The following steps connect these choices to the scientific claim being tested.
\begin{enumerate}[leftmargin=*,itemsep=1pt,topsep=2pt]
\item \textbf{Choose the scalar and reference outcome.} Specify the event and an independent measurement of it. Select the observable on development runs; record at fixed intervals and measure acquisition cost.
\item \textbf{Select window, lag, and decision threshold.} Fix $W$, stride, $E$, $\tau$, $k$, $T$, and the decision rule before evaluation. Adaptive settings need an explicit rule shared by matched runs.
\item \textbf{Compute MG and reject numerical degeneracy.} Use completed windows, dated by their endpoints. Retain raw amplitude and floor flags; check sensitivity to $W$, $\tau$, and $2E$. Report missing estimates and noise-dominated windows.
\item \textbf{Distinguish dimension recovery from change detection.} Dimension estimation requires recurrence, a stable occupation measure, a dimension-preserving observation, and sufficiently many returns above the noise scale. Otherwise, evaluate MG against the reference outcome and scalar baselines, including detection delay and false positives.
\end{enumerate}
Inadequate embedding dimension prevents faithful reconstruction, while finite-sample MG can exceed $E$. Stable estimates alone do not certify embedding assumptions; changes in SGD noise and temporal correlation can also affect the statistic.
}
'''
s=s.replace('We choose settings on development records and hold them fixed during evaluation, and report detection delay.','We select settings on development runs, freeze them for evaluation, and report detection delay.')
old=s[s.index(r'\caption{\claude{From a scalar'):s.index(r'\label{fig:method_geometry}')]
s=s.replace(old,r'''\caption{\claude{Scalar samples, delay vectors, and neighbor distances. An illustrative periodic signal (a) produces the two-coordinate reconstruction (b). Panel (c) enlarges a neighborhood: blue dots are admissible vectors, gray crosses are temporally excluded, the open circle is the query, and orange dots are its $k$ nearest neighbors. The radius ratios enter Eq.~\eqref{eq:mg}.}}
''')
write('sections/method.tex',s)

p=H/'sections/applications.tex';s=p.read_text(encoding='utf-8-sig')
s=s.replace('without redevelopment','without retuning')
s=s.replace('Thus detector transfer is task-specific; these paired records share seeds (\\appref{app:baselines}).','The weaker result on shifted intervention times limits transfer of the original threshold. Runs sharing a seed are paired observations; \\appref{app:baselines} gives the comparison details.')
# Shorten repeated interpretation, retaining the quantitative evidence in the body.
s=s.replace('This transfers a detector from about 15 thousand to 11 million parameters for a concrete training change. The global parameter norm is informative for backbone freezing, but the same detector misses the tested learning-rate reductions and pruning interventions in ResNet-18. Separately explored layer norms and random projections introduce false positives. Thus transfer succeeds for the freezing task, while other interventions require further evaluation of the observable and detector.','The freezing result transfers from about 15 thousand to 11 million parameters. The same rule misses learning-rate reductions and pruning; layer norms and random projections also introduce false positives. Transfer therefore depends on the event and observable.')
s=s.replace('This result validates MG as a signal of latent information loss and partial preservation. Standard deviation and spectral entropy also detect the loss on all nine seeds. In a separate cyclic-VAE test, using MG to schedule expensive reference checks finds 28/41 transitions at the primary budget, compared with 38/41 for periodic checks. Association with information loss therefore does not by itself imply better scheduling.','Standard deviation and spectral entropy also detect information loss on all nine seeds. Using MG to schedule reference checks is less successful: a separate cyclic-VAE test finds 28/41 transitions, versus 38/41 for periodic checks. Tracking information loss and scheduling measurements are distinct tasks.')
start=s.index(r'\subsection{Grokking:');end=s.index(r'\subsection{Walker2d:',start)
s=s[:start]+r'''\subsection{Grokking: a change in training dynamics}
Parameter-norm MG falls near generalization in four modular-arithmetic runs when its windows match the resolution of stored trajectory sketches. Those sketches independently show a reversible fall in detrended covariance participation ratio. Across analysis settings, 24/26 usable comparisons separate these runs from two references, but reuse the same trajectories. Roughness changes are stronger, a memorizing run also simplifies, and the full-batch setting does not reproduce the effect. MG detects a transition here; its decrease is not specific to generalization.

'''+s[end:]
s=s.replace(r'\input{sections/forecast_utility}',r'\input{sections/campaign_main}'+'\n'+r'\input{sections/forecast_utility}')
write('sections/applications.tex',s)
write('sections/campaign_main.tex',r'''
\subsection{Self-supervised learning: choosing configurations}\label{sec:ssl}
Self-supervised training can lose useful distinctions between images, so low training loss alone does not identify a good representation. We test whether a gradient-norm log helps select a configuration before extracting embeddings. On MNIST, 29 configurations span four joint-embedding objectives and a 784--256--256 MLP encoder. We choose the scalar and score direction using linear-probe accuracy on seeds 0--1, then freeze them for seeds 10--12 and two further seeds 20--21. The resulting selector needs no labels on the evaluated runs; its development used labels.

Gradient-norm MG achieves mean Spearman correlation 0.712 with probe accuracy and selects a configuration only 0.41 percentage points below the best on average. Corresponding correlations are 0.321 for encoder RankMe, 0.404 for the $\alpha$-ReQ slope, and 0.124 for LiDAR. Embedding spread reaches 0.730, and several simple scalar features are competitive. Thus one log supports useful selection without an embedding pass, while the strongest alternative depends on the comparison. \appref{app:campaign} reports all scores, the separate confirmation cohort, and the limitations of the proxy implementations and single dataset.

\paragraph{Warning of loss spikes.}
In a two-layer transformer trained on modular addition, MG of log-loss warns of 22/23 test spikes within a 500-step horizon, with AUROC 0.949 and 2.44 false positives per 10,000 scored steps. Attention-logit and attention-entropy monitors reach 0.645 and 0.911 AUROC. Roughness reaches 0.960 at the same false-positive rate and lower cost. This experiment supports scalar warning of the slingshot regime; prevention of spikes and transfer to language models remain untested (\appref{app:campaign}).
''')
p=H/'sections/forecast_utility.tex';s=p.read_text(encoding='utf-8-sig')
start=s.index('In this setting, the sampled-query');end=s.index('\n',start)
s=s[:start]+'The measured saving is in selection and fitting from an existing record; training the recurrent network is a separate cost.'+s[end:]
s=s.replace('We test whether scalar geometry supports a useful decision: selecting a forecasting model for a trained recurrent network.','We next use scalar geometry to select a forecast model for trained recurrent activity.')
write('sections/forecast_utility.tex',s)
p=H/'sections/cost_discussion.tex';s=p.read_text(encoding='utf-8-sig')
start=s.index(r'\enlargethispage');s=s[:start]+r'''\section{Conclusion}
We developed and tested a way to monitor neural dynamics from one scalar record. Delay-embedding theory explains the active-component interpretation, synthetic experiments test it, and applications connect the statistic to independently measured changes. The new SSL study extends its use to configuration selection, with low average selection regret and stronger ranking than several embedding-based proxies in this setting. Sampled queries further reduce the cost of forecast selection while retaining full-MG decisions. Together, these results support an inexpensive geometric diagnostic across distinct learning tasks. Its interpretation depends on the observable and regime, and simple scalar baselines remain essential comparisons.
}
'''
start=s.index(r'\paragraph{Storage');end=s.index(r'\begin{table}',start)
s=s[:start]+r'''MG analyzes one scalar per step, with storage and neighbor-search cost set by $W$ and $E$. It needs no model access once the record is available. Acquisition is separate: a parameter or gradient norm requires an $O(P)$ reduction, and probe loss requires a forward pass.

'''+s[end:]
write('sections/cost_discussion.tex',s)
# Repair earlier wording substitutions without touching filenames or labels.
p=H/'sections/appendix.tex';s=p.read_text(encoding='utf-8-sig')
for a,b in [('required reference of distances','required bounds on distance distortion'),('PPO policies reference six actuators','PPO policies command six actuators'),('reference steps','environment steps'),('Seed 100 chooses thresholds; seeds 101--109 confirm them.','We select thresholds on seed 100 and evaluate on seeds 101--109.'),('a separate held-out development gave','a separate development run gave'),('The shaded region marks the fixed regime','The shaded region marks the tested regime')]:s=s.replace(a,b)
# Define occupation measure explicitly in the theory appendix.
anchor='The reconstruction theorem alone does not guarantee accurate estimation; the estimator alone does not establish an embedding.'
s=s.replace(anchor,anchor+r'''

For a stationary ergodic trajectory, the occupation measure can be written as $\mu(A)=\lim_{N\to\infty}N^{-1}\sum_{t=1}^{N}\mathbf{1}\{s_t\in A\}$ on sets whose boundary has zero measure. It records long-run occupancy, rather than the dimension of a finite list of observations. In Eq.~\eqref{eq:active}, $\epsilon\to0^+$ and $\epsilon\downarrow0$ denote the same right-hand limit. We use the former to emphasize that $\epsilon$ is a positive distance scale, distinct from embedding dimension $E$. Nonstationary training need not have such a limiting measure; the applied experiments therefore use finite-window scores and external reference outcomes.
''')
# Clarify that proof belongs with geometric interpretation, before implementation.
start=s.index(r'\begin{proposition}[Bi-Lipschitz');end=s.index(r'\subsection{Affine invariance',start)
proof=s[start:end];s=s[:start]+s[end:]
idx=s.index(r'\subsection{Recurrent, transient');s=s[:idx]+proof+'\n'+s[idx:]
write('sections/appendix.tex',s)
p=H/'sections/practical_appendix.tex';s=p.read_text(encoding='utf-8-sig').replace('Seeds 0--2 choose the rules; seeds 10--17 evaluate them.','We choose rules on seeds 0--2 and evaluate them on seeds 10--17.')
write('sections/practical_appendix.tex',s)
p=H/'preamble.tex';s=p.read_text(encoding='utf-8-sig').replace(r'\usepackage{etoolbox,float}',r'\usepackage{etoolbox}').replace(r'\usepackage[hidelinks]{hyperref}',r'\usepackage{float}'+'\n'+r'\usepackage[hidelinks]{hyperref}')
write('preamble.tex',s)
p=H/'appendix_part.tex';s=p.read_text(encoding='utf-8-sig');s+=r'\input{sections/campaign_appendix}'+'\n';write('appendix_part.tex',s)
