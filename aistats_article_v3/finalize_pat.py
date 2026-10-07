"""One-time source corrections and provenance snapshots for the October 4 revision."""
from pathlib import Path
import hashlib,json,shutil
H=Path(__file__).resolve().parent;R=H.parent
def edit(file,old,new):
    p=H/file;s=p.read_text(encoding='utf-8');assert old in s,(file,old[:60]);p.write_text(s.replace(old,new),encoding='utf-8')
edit('sections/appendix.tex',r'\begin{tabular}{llllllll}',r'\begin{tabular}{llrrrrll}')
edit('sections/appendix.tex','Driven digits head & state-based observers','Digits head & state-based')
edit('sections/appendix.tex',' & span; cap 150 & one window', ' & span; cap 150 & 3,000')
edit('sections/appendix.tex','Known-mode regression & loss / probe','Regression & loss / probe')
edit('sections/appendix.tex','1,000-unit generator & activity of neuron zero','Generator & neuron zero')
edit('sections/appendix.tex','autocorrelation; cap 320 & 8,192','ACF; cap 320 & 8,192')
edit('sections/appendix.tex','fixed-probe reconstruction NLL & 512','probe NLL & 512')
edit('sections/appendix.tex',r'\tau=\min\{\max(1,\operatorname{round}(t_a/4)),\max(1,\lfloor W/[8(E-1)]\rfloor)\}.',r'''\begin{aligned}
\tau_0&=\max(1,\operatorname{round}(t_a/4)),\\
\tau&=\min\{\tau_0,\max(1,\lfloor W/[8(E-1)]\rfloor)\}.
\end{aligned}''')
edit('sections/appendix.tex',r'$\{0.025,0.05,0.075,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1,1.5,2,3\}$',r'0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 1.5, 2, and 3')
edit('sections/appendix.tex','The known-mode table uses endpoint windows;', 'The historical digits sweep uses stride 3,000 on the 26,000 post-burn samples ($\\max(500,\\lfloor(N-W)/6\\rfloor)$); calibration used stride 2,000. The known-mode table uses endpoint windows;')
edit('sections/appendix.tex','Quasiperiodic modulation of selected group weights drives the recurrent motion.',r'''For the historical digits experiment, weights are $w_i(t)=\max\{0.05,1+0.8\sum_{j=1}^r G_{ji}\sum_{\ell=1}^r A_{j\ell}\sin(2\pi f_\ell t+\phi_\ell)\}$. Here $G$ identifies the twelve fixed data groups and $A$ equalizes their linearized forcing directions and gains. For $r>1$, $f_j=2^{a_j}/16$, with $a_j=0.03+0.94(b_j-\min b)/(\max b-\min b)$ and $b_j=\operatorname{frac}(\sqrt{p_j})$ for the first $r$ primes; for $r=1$, $f_1=\sqrt2/16$. Phases are uniform on $[0,2\pi)$ from NumPy's generator seeded by $31+\text{seed}$; group assignment uses $7717+\text{seed}$. The drive amplitude is 0.8 and the preconditioned head step is 0.15. These describe the archived implementation behind the reported scores; the newer code organizes random streams differently.''')
edit('sections/appendix.tex','The short-window parameter-norm analysis reports a coincident decline after matching temporal bandwidth.',r'''The short-window parameter-norm analysis uses 60 logged samples at the centers of the trajectory windows. Its grid crosses $E\in\{4,6,10\}$, $\tau\in\{1,2,4\}$, $k\in\{5,20\}$ and embedding/ACF exclusion. The headline cell maximizes $E$ subject to $(E-1)\tau\le60/4$, retaining the frozen $k$ and exclusion rule, and breaking ties by smaller $\tau$: $E=10$, $\tau=1$, $k=20$, embedding-span exclusion. This rule does not select the largest observed effect. The scalar decline is a change statistic at that resolution.''')
# Remove repeated explanation while preserving numerical comparisons and application descriptions.
p=H/'sections/applications.tex';s=p.read_text(encoding='utf-8')
a=s.index('We also evaluated natural scalar competitors');b=s.index('\\begin{figure*}',a)
s=s[:a]+s[b:]
a=s.index('MG detects 27/28 strong interventions');b=s.index('\\begin{table}',a)
s=s[:a]+r'''Table~\ref{tab:detectors} includes every tested scalar feature. MG detects 27/28 strong interventions with 0/16 control false positives and median delay 1,000 updates; normalized increments give 28/28 and 0/16, while absolute increments match MG with shorter delay. At horizons 500, 1,000, and 2,000, MG detects 8/28, 16/28, and 22/28 events (Appendix~\ref{app:baselines}). Thus its advantage here is not universal detection superiority. On ten additional seeds with shifted intervention times, its frozen threshold gives 19/30 detections and false positives in five of ten controls, compared with 30/30 and zero for absolute increments. Weak interventions and threshold transfer remain limitations. These paired records share seeds; we test detection, not the benefit of acting on it.

'''+s[b:]
s=s.replace('$W=1,000$', '$W=1{,}000$')
p.write_text(s,encoding='utf-8')
edit('sections/controlled.tex','MG of the training loss falls from 4.97 to 2.56.', 'Across three coordinate/probe seeds, loss-based MG falls from 4.97 to 2.56; these seeds share the same loss dynamics (Appendix~\\ref{app:controlled}).')
edit('sections/cost_discussion.tex','Each 1,000-point window takes about 9.5\\,ms to process (Table~\\ref{tab:cost}). This makes repeated analysis inexpensive in the tested implementation. A streaming covariance calculation or trajectory sketch would reduce the reference\'s storage requirement; the comparison is with the saved full trajectory.', 'Table~\\ref{tab:cost} reports analysis times. Streaming covariance or sketches would reduce the reference storage; the comparison uses the saved full trajectory.')
edit('sections/abstract.tex','These results establish a low-cost diagnostic of several forms of learned simplification,', 'These results support a diagnostic with low analysis cost when an informative scalar is available,')
edit('sections/introduction.tex',r'and rank-based grokking analyses \citep{gao2017theory,cohen2021gradient,yunis2024rank,humayun2024deep}',r'and studies of representation, function complexity, and weight rank at grokking \citep{gao2017theory,cohen2021gradient,liu2022omnigrok,yunis2024rank,humayun2024deep}')
# Bundle the unchanged estimator and experiment sources in their original relative layout.
manifest=[]
def copy(src,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    manifest.append(dict(path=dest.relative_to(H).as_posix(),source=src.relative_to(R).as_posix(),sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
for rel in ['code/actdim/__init__.py']+[p.relative_to(R).as_posix() for p in (R/'code/actdim/estimator').glob('*.py')]:
    copy(R/rel,H/'measurement_code'/Path(rel).relative_to('code'))
roots=['code/actdim','research_trajectory_reference','research_text_vae','research_generator','research_force_motion']
roots +=[p.name for p in R.glob('research_walker*') if p.is_dir()]
for root in roots:
    for src in (R/root).rglob('*.py'):
        if '__pycache__' not in src.parts:copy(src,H/'experiment_code'/src.relative_to(R))
copy(R/'research_known_modes.py',H/'experiment_code/research_known_modes.py')
for src in (R/'archived_code/active_dimension').glob('*.py'):copy(src,H/'experiment_code'/src.relative_to(R))
for rel in ['research_text_vae/cyclical/selection.json','research_known_modes_results/metadata.json','code/data/calib.e8/frozen_config.json']:
    copy(R/rel,H/'evidence'/rel)
(H/'source_snapshot_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(f'Final text edits applied; {len(manifest)} source files snapshotted.')
