from pathlib import Path
import json,shutil,hashlib
import pandas as pd
H=Path(__file__).resolve().parent;R=H.parent/'research_mg_deployment'
d=pd.read_csv(R/'checkpoint_summary.csv').set_index('method')
assert abs(d.loc['MG','reward']/d.loc['reward','reward']-1-.076900102)<1e-5
p=H/'sections/applications.tex';s=p.read_text(encoding='utf-8')
start=s.index('We also analyze behavior learned by proximal policy optimization')
end=s.index('\n}',start)
s=s[:start]+r'''PPO policies in Walker2d \citep{schulman2017proximal,todorov2012mujoco} show lower control-log MG under moderate action smoothing in two five-seed series. This does not imply simpler whole-body motion. We additionally select among 16 saved policies per seed using three nominal rollouts and a shared reward/survival gate. On four held-out fine-tuning seeds, minimum action-norm MG gives deployment return 4.313 versus 4.005 for nominal-reward selection under hidden actuator lag/noise; completion rises from 80.0\% to 90.8\%. Spectral entropy performs better (4.480, 97.5\%), and a pilot-selected fixed configuration nearly matches MG (4.310). This is exploratory policy selection, not demonstrated curriculum or training acceleration; Appendix~\ref{app:rl_selection} reports all comparisons and shared-initial-policy limitations.
'''+s[end:];p.write_text(s,encoding='utf-8')
p=H/'sections/appendix.tex';s=p.read_text(encoding='utf-8');pos=s.index(r'\section{Timing protocols')
s=s[:pos]+r'''\subsection{Selecting policies for actuator shifts}\label{app:rl_selection}
We evaluated historical PPO policies on new nominal and deployment episodes. The 16 candidates per fine-tuning seed cross smoothing strengths $0,0.25,1,4$ with checkpoints at 262,144, 524,288, 786,432 and 1,048,576 transitions. Development seed 231 is excluded from the summary; seeds 232--235 provide the four confirmation units. All policies share one initial learned skill. Earlier results informed the setting, so this is new-outcome validation on existing policies, not an independent sample of newly learned skills.

Each selector receives three nominal episodes (reset seeds 83001--83003), with 256 warm-up and 2,048 measured control steps per episode. Eligible candidates complete at least two episodes and have mean zero-padded return at least 90\% of the largest nominal mean. MG selects the lowest median estimate from the scalar action norm over complete episodes, using $E=20$, $\tau=8$, $k=20$, $T=312$, and dither seed 123. Entropy, normalized squared increments, and minimum scalar return error over lags 20--250 use the same log. Action smoothness $J_1$ uses all six commands. Reward selection maximizes the mean over all episodes, including falls. Every method has the same eligibility gate. The fixed pilot choice is smoothing strength 0.25 at the final checkpoint, falling back to nominal reward if ineligible.

Deployment applies $v_t=(1-\alpha)u_t+\alpha v_{t-1}+\epsilon_t$, with $v_{-1}=0$, commanded action $u_t$, and independent Gaussian components of $\epsilon_t$ with standard deviation $\sigma$. The environment receives $v_t$ clipped to $[-1,1]$; the recursion retains the unclipped value. The six conditions cross $\alpha\in\{0.05,0.1,0.2\}$ and $\sigma\in\{0,0.02\}$ and apply also during warm-up. Five new reset seeds 84001--84005 per condition give 30 target episodes per candidate. Return is the measured reward sum divided by 2,048, with a zero remainder after falls. Selectors never use confirmation target outcomes; the oracle does. The control interval is 0.008 seconds.

\begin{table}[ht]
\centering\small
\caption{\claude{Policy selection under hidden actuator shifts: mean over four fine-tuning seeds. Completion is over all 120 selected target episodes. Random denotes expected uniform choice among eligible policies; the oracle uses target returns.}}
\label{tab:rl_selection}
\input{tables/rl_selection.tex}
\end{table}

MG exceeds nominal-reward selection on three of four seeds, with paired mean difference 0.308 (7.7\%); a two-sided sign-flip test gives $p=0.375$. It does not outperform spectral entropy, and its mean advantage over the fixed pilot is only 0.0025. Reset repetitions and candidate checkpoints are correlated and are not independent statistical units. MG-based selection therefore provides a descriptive benefit over one common rule, not established superiority over competing selectors.

On an Intel i5-12500H CPU with one numerical thread, a 2,048-point MG calculation takes 0.516 seconds, compared with 0.00014 for entropy and 0.0062 for scalar recurrence. Three nominal episodes take 5.52 seconds, while 30 target episodes take 44.7 seconds on the benchmark policy. Nominal acquisition plus three MG calculations is approximately 7.06 seconds; these component medians do not establish a universal speed advantage, and entropy uses the same acquisition budget. Training all candidate policies is a shared sunk cost. Choosing an earlier stored checkpoint does not demonstrate causal early stopping or curriculum improvement.

An earlier stress test selected among four final policies using delay-free observations and applied delays of 1--3 entire control steps after warm-up. On five seeds, MG and the pilot-selected fixed strength 1 choose the same policies, with return 0.541 versus 0.319 for nominal reward, but only 2/150 target episodes finish without falling. This severe-shift test is retained as a negative deployment result.

'''+s[pos:];p.write_text(s,encoding='utf-8')
labels={'MG':'MG','reward':'Nominal reward','entropy':'Spectral entropy','recurrence':'Scalar recurrence','J1':'Action smoothness','increments':'Norm. increments','fixed_pilot':'Fixed pilot','random_eligible':'Random eligible','oracle_eligible':'Oracle (eligible)'}
lines=[r'\begin{tabular}{lrr}',r'\toprule',r'Selector & Return & Completion \\',r'\midrule']
for key,name in labels.items():lines.append(f"{name} & {d.loc[key,'reward']:.3f} & {100*d.loc[key,'survival']:.1f}\\% "+r'\\')
lines +=[r'\bottomrule',r'\end{tabular}'];(H/'tables/rl_selection.tex').write_text('\n'.join(lines))
dest=H/'evidence/rl_selection';dest.mkdir(exist_ok=True)
names=['checkpoint_summary.csv','checkpoint_selections.csv','checkpoint_comparisons.csv','checkpoint_records.csv','delay_summary.csv','delay_selections.csv','selection_manifest.json','audit.json','timing_summary.csv','CHECKPOINT_PROTOCOL.md','PROTOCOL.md','analyze.py','checkpoint_selection.py','run.py','README.md']
manifest=[]
for name in names:
    shutil.copy2(R/name,dest/name);manifest.append(dict(path=name,sha256=hashlib.sha256((dest/name).read_bytes()).hexdigest()))
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Added audited RL selection results and evidence.')
