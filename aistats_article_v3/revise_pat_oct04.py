"""Apply the documented PAT editorial corrections; no experiments are modified."""
from pathlib import Path
H=Path(__file__).resolve().parent
p=H/'sections/appendix.tex'
s=p.read_text(encoding='utf-8')
def rep(old,new):
    global s
    assert old in s,old[:100]
    s=s.replace(old,new)
rep('and a warm-up through update 4,000', 'and excludes feature windows ending at or before update 1,500 from the detection history')
rep('ordinal order five;', 'ordinal order five, unit delay, stable sorting for ties, and normalization by $\\log(5!)$;')
rep('template length two and tolerance $0.2$ times the window standard deviation;', 'template length two, unit delay, Chebyshev distance, tolerance $0.2$ times the window standard deviation, and self-match exclusion;')
rep('and its calibrated thresholds are reported in Table~\\ref{tab:cusum}.',r'''and the parameters in Table~\ref{tab:baseline_rules}. The block score is $s(1-\operatorname{median}_{M}/\operatorname{median}_{B})$, with direction $s\in\{-1,1\}$; MG fixes $s=1$. Calibration searches $M\in\{2,3,4\}$, $B\in\{3,4,6\}$. Its threshold exceeds the largest control score by 0.02, and selection maximizes event detections, then minimizes median delay. A baseline ratio is skipped when the reference magnitude is at most $10^{-12}$.

CUSUM fixes its mean $\mu_0$ and scale $\sigma_0=\max(\operatorname{std}_0,0.01|\mu_0|,10^{-10})$ from feature windows ending at 2,000--3,500. Starting at zero, $C_j=\max\{0,C_{j-1}+s(\mu_0-v_j)/\sigma_0-\kappa\}$; a detection occurs when $C_j>h$. Both directions and the stated drift grid are calibrated, with $h$ exceeding the largest calibration-control score by 0.5. These rules are causal but warm-up must finish before scoring; they are not detectors available from the first training step.

\begin{table*}[t]
\centering\small
\caption{\claude{Selected detector parameters. The block sign is listed; all selected CUSUM signs are recorded in \texttt{new\_results/rules.json}. Full-precision values in that file define the implemented thresholds.}}
\label{tab:baseline_rules}
\input{tables/baseline_rules.tex}
\end{table*}''')
rep('On this benchmark, normalized increments and the two entropy features detect all strong events, while MG has no false positives on any control type.', 'On this benchmark, normalized increments detect 28/28 events with 0/16 control false positives, compared with 27/28 and 0/16 for MG. Absolute increments match MG in detections and false positives with shorter median delay. Both entropy features detect all events but produce control false positives.')
rep('The MG horizon sensitivity on the original held-out records is',r'''\begin{table*}[t]
\centering\small
\caption{\claude{Post-hoc horizon sensitivity with thresholds and first detections unchanged. Entries are detections within the specified number of updates; FP uses each complete control record and is independent of the event horizon.}}
\label{tab:horizons}
\input{tables/horizon_sensitivity.tex}
\end{table*}
The MG horizon sensitivity on the original held-out records (Table~\ref{tab:horizons}) is''')
rep('the pretrained setting uses $64\\times64$ images and ImageNet normalization.', 'the pretrained setting uses $64\\times64$ images and ImageNet normalization. The pretrained network retains the standard $7\\times7$, stride-two stem and max pool and replaces its classifier by a ten-class head. The scratch network uses a $3\\times3$, stride-one stem, no initial max pool, and $32\\times32$ CIFAR-normalized inputs.')
rep('A channel is classified as dead if it is zero on all 500 diagnostic images.', 'The tested learning-rate multipliers are 20, 30, 50, and 100 for one update; 15 for three updates; and 20 for two updates, followed by restoration of the base rate. A channel is classified as dead if it is zero on all 500 diagnostic images.')
rep('This gives 21 eligible events and 31 non-event records.', 'The 42 runs comprise seven arms on six seeds: 23 meet the event criterion and 19 do not. Removing two catastrophic events leaves 21 eligible events; six rescaled and six smoothed base logs augment the 19 non-events to 31. Thus $42-2+12=52$ scored records, not 52 independent training runs.')
rep('a budget of 6,12,or 24 extra checks.', 'a budget of 6, 12, or 24 extra checks.')
rep('thresholds are selected from the same fourteen-value grid on the pilot.', 'thresholds are selected on the pilot from $\\{0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1,1.5,2,3\\}$. Features are evaluated every 64 updates in this cyclic experiment; the earlier preservation experiment uses stride 128.')
rep('Rescaling and smoothing transform the base log without retraining.',r'''Rescaling multiplies the base log by ten after update 4,000. Smoothing replaces that suffix by the causal 16-sample moving average $\bar x_t=\sum_{j=0}^{15}x_{t-j}/16$, using the preceding original samples at the boundary; neither transformation retrains the network.

The CNN IAAFT comparison uses two surrogates per window and reports their median MG, with RNG seed equal to the window start. Each surrogate uses 100 alternating spectrum/rank projections. Endpoint matching can trim up to 15\% of the window before randomization, so spectrum and marginal preservation refer to the retained segment; it is a descriptive control, not a matched-length significance test.''')
rep('The original modular-arithmetic studies include regularized mini-batch and unregularized full-batch settings.',r'''The four positive sketch runs comprise modular addition $y=(a+b)\bmod113$ on seeds 42--44 and permutation composition in $S_5$ on seed 42. Training fractions are 0.30 and 0.50, respectively; zero-decay controls use the corresponding seed-42 tasks. A one-layer transformer has width 128, four 32-dimensional heads, MLP width 512, no layer normalization, and float64 arithmetic. AdamW uses learning rate $10^{-3}$, betas $(0.9,0.98)$, batch 256, and decay 1.0 (addition) or 0.2 ($S_5$). Training lasts 20,000 or 15,000 loop iterations. The historical $S_5$ implementation performs two optimizer updates per loop iteration; this difference is retained when reporting its times. The unregularized full-batch comparison uses a width-500 quadratic perceptron at modulus 97, a 50\% training split, and plain gradient descent on mean squared error.''')
rep('We retain this as a complementary example rather than the sole applied validation.',r'''We retain this as a complementary example rather than the sole applied validation. Linear detrending removes the scalar separation in the matched-window analysis. Covariance PR can change with drift, correlation time, or coherent motion even without a change in active dimension; random-walk and Ornstein--Uhlenbeck trajectory PCA illustrate this distinction \citep{antognini2018pca}. We have not established that the grokking effects persist against nulls matched jointly in temporal correlations and nonstationary drift. The observations therefore identify a change in trajectory and scalar statistics, not its unique dimensional cause.''')
rep('Moderate action-smoothing penalties give the most reproducible scalar result:',r'''The smoothing objective is
\[
L=L_{\rm PPO}+\frac{\lambda}{6}\mathbb E\|\bar a_\theta(s_{t+1})-\bar a_\theta(s_t)\|_2^2,
\]
where $\bar a_\theta(s)$ is the policy mean clipped to $[-1,1]$ per actuator. Episode boundaries are excluded. The environment reward and executed commands receive no extra smoothing. Separate tracking variants change the reward to $r'_t=r_t-\lambda(1-\exp[-e_t^2/(2\sigma^2)])$, where $e_t$ measures distance to the recorded reference cycle, either at its nearest point or at the prescribed phase. The reference has 152 samples; the external phase period is approximately 152.215 control steps. The initial $\sigma^2=0.116779$ saturated; a separate validation calibration gave 38.477 for the broader tracking experiment.

Actor and critic use separate 64--64 tanh networks, eight environments, 512-step rollouts per environment, batches of 256, five PPO epochs, learning rate decreasing from $3\cdot10^{-5}$ to $3\cdot10^{-6}$, clip 0.1, target KL 0.01, discount 0.99, GAE 0.95, entropy coefficient zero, value coefficient 0.5, and gradient clipping 0.5. Normalizations are frozen. Fine-tuning branches use a common initial checkpoint after 655,360 transitions, then 1,048,576 additional transitions per branch; their seeds do not represent independently learned initial policies.

Moderate action-smoothing penalties give the most reproducible scalar result:''')
rep(r'\paragraph{Window length and recurrence requirements.}',r'''\paragraph{Exact adaptive rules.}
For centered samples $z_t$, define $a(\ell)=\sum_{t=1}^{W-\ell}z_tz_{t+\ell}/\sum_{t=1}^Wz_t^2$. The autocorrelation time $t_a$ is the first lag with $a(\ell)<e^{-1}$, or $W$ if none exists. The adaptive generator uses
\[
\tau=\min\{\max(1,\operatorname{round}(t_a/4)),\max(1,\lfloor W/[8(E-1)]\rfloor)\}.
\]
The implementation uses nearest-even rounding. Theiler settings resolve to $T=\min(T_{\max},T_0)$, where $T_0=(E-1)\tau$ for the embedding rule and $T_0=\max((E-1)\tau,t_a)$ for the autocorrelation rule. An explicit integer is also capped. Thus a cap can shorten the requested exclusion below the embedding span; report both the rule and its resolved value. The primary digits configuration uses the embedding rule (76 samples), not an uncapped autocorrelation rule.

The known-mode table uses endpoint windows; its additional switch trace uses $W=4,096$ and stride 2,048. Walker uses stride 512 in the early repair series and 1,024 in later three-window summaries. Generator windows are disjoint. The cyclic VAE uses stride 64. These variations are fixed per protocol, not selected after comparing outcomes.

\paragraph{Window length and recurrence requirements.}''')
s=s.replace('neighbours','neighbors').replace('neighbour','neighbor').replace('1,000-unit E9 generator','1,000-unit generator')
p.write_text(s,encoding='utf-8')
p=H/'references.bib';s=p.read_text(encoding='utf-8')
s=s.replace('by E. Levina and P. Bickel}', 'by {E. Levina} and {P. Bickel}}').replace('MacKay, David J C','MacKay, David J. C.')
s=s.replace('Corpus of English: The Penn Treebank}', 'Corpus of {English}: The {Penn Treebank}}').replace('Machine Learning in Python}', 'Machine Learning in {Python}}')
s=s.replace('year={2017}, eprint={1707.06347}, archivePrefix={arXiv}', 'year={2017}, note={arXiv:1707.06347}, eprint={1707.06347}, archivePrefix={arXiv}')
s=s.replace('title={PCA of High', 'title={{PCA} of High')
p.write_text(s,encoding='utf-8')
print('Applied appendix and bibliography corrections.')
