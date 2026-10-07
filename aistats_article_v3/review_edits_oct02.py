from pathlib import Path
H=Path(__file__).resolve().parent
def edit(name, changes):
    p=H/name;s=p.read_text(encoding="utf-8")
    for old,new in changes:
        if old not in s:
            if new not in s: print("NOT FOUND",name,old[:90])
        else:s=s.replace(old,new)
    p.write_text(s,encoding="utf-8")
edit("sections/abstract.tex",[
("Analysis takes 9.5\\,ms per CNN log window and is 14--19 times faster than the full Lyapunov spectrum in the specified recurrent-network benchmark. ",""),
("Delay-embedding theory motivates","Delay-embedding theory from dynamical systems motivates"),
])
edit("sections/method.tex",[
(r"\section{Estimating complexity from a scalar time series}",r"\section{Estimating the complexity of neural network dynamics}"),
(r"\section{Dynamical-systems basis for scalar-log analysis}",r"\section{Estimating the complexity of neural network dynamics}"),
(r"\subsection{The geometric information in delayed observations}",r"\subsection{Delay reconstruction and dynamical-systems theory}"),
(r"\subsection{Delay-coordinate reconstruction from dynamical-systems theory}",r"\subsection{Delay reconstruction and dynamical-systems theory}"),
(r"\subsection{Computing MG}",r"\subsection{Pooled local-dimension estimation}"),
("and test their usefulness as a relative complexity measure in applied training.","and test changes in the estimate against independent measures of simplification in applied training."),
("How a scalar record enters MG. Delayed samples reconstruct a trajectory in a coordinate space; the estimator compares the radii of local neighbour shells after excluding temporally adjacent points. The figure illustrates the geometry used by the method, not an additional experiment.",
r"Delay reconstruction and local dimension estimation for a one-phase signal. (a) First 400 scalar observations. (b) Two delay coordinates form a closed loop. (c) Neighbour distances around a reference point enter Eq.~\eqref{eq:mg}. Temporally adjacent samples are excluded; neighbours correspond to later or earlier returns. In the diagram, $m=k$ and $W_T=T$."),
("Geometric construction used by MG. (a) A scalar log; (b) its delay-coordinate reconstruction; (c) local neighbour shells used by the estimator after temporally adjacent points are excluded.",
r"Delay reconstruction and local dimension estimation for a one-phase signal. (a) First 400 scalar observations. (b) Two delay coordinates form a closed loop. (c) Neighbour distances around a reference point enter Eq.~\eqref{eq:mg}. Temporally adjacent samples are excluded; neighbours correspond to later or earlier returns. In the diagram, $m=k$ and $W_T=T$."),
(r"\paragraph{Scale invariance.}",r"\paragraph{Invariance to affine rescaling.}"),
(r"\paragraph{Scale and offset invariance.}",r"\paragraph{Invariance to affine rescaling.}"),
(r"\paragraph{Windowing and interpretation.}",r"\paragraph{Window length and temporal resolution.}"),
(r"\paragraph{Window length and interpretation.}",r"\paragraph{Window length and temporal resolution.}"),
])
edit("sections/applications.tex",[
("This task tests whether MG can reveal when the decoder stops using information in the code.","This task tests whether MG detects a reduction in the dependence of the decoder's output distribution on the latent code."),
("Free bits reduces pressure to suppress low-KL latent coordinates.","The free-bits objective excludes low-KL latent coordinates from further KL penalization."),
(r"\paragraph{MG adds information beyond simple scalar statistics.}",r"\paragraph{Scalar baselines and spectral surrogates.}"),
(r"\paragraph{Detecting changes using only past observations.}",r"\paragraph{Causal detection of training interventions.}"),
("We count the first threshold crossing as an alarm and score it as a detection if it occurs within 5,000 updates after the intervention. An earlier alarm is false.","The first threshold crossing defines a detected change. It is correct if it occurs within 5,000 updates after the intervention; an earlier crossing is a false positive."),
("The CNN benchmark provides the clearest detection advantage over simple scalar statistics.",""),
(r"\paragraph{Grokking.}",r"\paragraph{Generalization transitions in modular arithmetic.}"),
(r"\paragraph{Learned locomotion.}",r"\paragraph{Action regularity in Walker2d policies.}"),
])
edit("sections/cost_discussion.tex",[
(r"\paragraph{Analysis from a compact record.}",r"\paragraph{Storage and analysis of scalar records.}"),
(r"\paragraph{Savings relative to full Lyapunov spectra.}",r"\paragraph{Runtime relative to full Lyapunov spectra.}"),
(r"\paragraph{When logging costs dominate.}",r"\paragraph{Observation costs in VAE monitoring.}"),
(r"\paragraph{Where the method adds value.}",r"\paragraph{Geometric information and computational savings.}"),
(r"\paragraph{Choosing a useful observable.}",r"\paragraph{Observable sensitivity and detector transfer.}"),
("We developed and evaluated a method for monitoring simplification from scalar logs. Delay-embedding theory motivates its active-component interpretation, and controlled experiments test that interpretation against known phase structure.",
"We developed and evaluated a scalar-log method for monitoring simplification of neural network dynamics, motivated by delay-embedding theory from dynamical systems. Under suitable reconstruction and sampling assumptions, local dimension estimation provides an active-component interpretation; controlled experiments test this interpretation against known phase structure."),
])
edit("sections/controlled.tex",[
(r"\paragraph{Known phases.}",r"\paragraph{Oscillatory signals with known dimension.}"),
(r"\paragraph{Learning on MLP features.}",r"\paragraph{Classifier training on fixed MLP features.}"),
])
edit("sections/appendix.tex",[
(r"\subsection{What follows from an embedding}",r"\subsection{Dimension preservation under delay reconstruction}"),
(r"\subsection{What the scalar geometry looks like}",r"\subsection{Recurrent, transient, and stochastic regimes}"),
(r"\subsection{The estimator as an explicit algorithm}",r"\subsection{Pooled estimator: implementation steps}"),
(r"\paragraph{Selecting a window.}",r"\paragraph{Window length and recurrence requirements.}"),
("The result changes the claim we make: simple features can be better event detectors on this particular CNN, whereas MG supplies a geometric quantity and a common protocol that also applies to the controlled generator, VAE, RL, and full-state timing comparisons.","Simple features achieve higher detection rates in this CNN benchmark. MG provides a geometric statistic that is also evaluated on the controlled generator, VAE, RL, and full-state timing benchmarks."),
("We also ran a preregistered-style confirmation","We also ran a confirmation with the protocol fixed before training"),
("Normalized increments produced 30/30 hits with no control alarms.","Mean absolute increments produced 30/30 detections with no control false positives. Normalized increments produced 29/30 detections and false positives in two of ten controls."),
("five of ten ordinary-training control runs alarmed, and six intervention prefixes alarmed before the assigned intervention time","five of ten ordinary-training controls contained threshold crossings, and six intervention branches had crossings before the assigned intervention time"),
("We retain this result as a scope check:","This confirmation indicates that"),
])
for p in list((H/"sections").glob("*.tex"))+list((H/"tables").glob("*.tex")):
    s=p.read_text(encoding="utf-8")
    changes=[
      ("alarm threshold","detection threshold"),("control-alarm trade-off","detection--false-positive trade-off"),
      ("Control alarms","False positives"),("control alarms","false positives"),
      ("false-alarm probabilities","false-positive probabilities"),("false alarms","false positives"),
      ("raises alarms","produces false positives"),("no alarms","no false positives"),
      ("with an alarm","with a threshold crossing"),("on an alarm","on a detected change"),
      ("its alarms","its detections"),("Alarms","False positives"),("alarms","detections"),
      ("alarm","detected change"),
      ("model's parameter count","number of model parameters"),("network's parameter count","number of network parameters"),
      ("trainable parameter count","number of trainable parameters"),("parameter count alone","the number of parameters alone"),
      ("An exact component count","An exact component dimension"),
      ("neighbour count $k$","number of neighbours $k$"),
      ("tracks this count","tracks this dimension"),
      ("Their target phase counts are","Their target phase dimensions are"),
      ("identical spectral peak counts","the same number of spectral peaks"),
      ("spectral peak counts","the number of spectral peaks"),
      ("peak counts and covariance summaries","spectral peak statistics and covariance summaries"),
      ("higher phase counts","higher phase dimensions"),
      ("the count of near-threshold Hessian modes","the number of near-threshold Hessian modes"),
      ("ratios, medians, counts","ratios, medians, detection totals"),
      ("We present counts and paired contrasts","We report detection totals and paired contrasts"),
      ("Its count agrees with the task phase count","The number of near-zero exponents agrees with the target phase dimension"),
      ("is overcounted","is overestimated"),
      ("The primary 27/28 count","The primary 27/28 result"),
      ("these counts describe","these totals describe"),("these counts are","these results are"),
      ("A channel is counted as dead","A channel is classified as dead"),
      ("the exact request count per seed","the number of requests for each seed"),
      ("uses request counts","uses request totals"),("actual operation counts","actual numbers of operations"),
      ("Hits count the seven strong intervention types across four seeds; detections count the four control types across four seeds.",
       "Successful detections cover seven strong intervention types across four seeds; false positives cover four control types across four seeds."),
    ]
    for a,b in changes:s=s.replace(a,b)
    p.write_text(s,encoding="utf-8")
edit("prepare_assets.py",[("'Alarms'","'False positives'")])
edit("make_baseline_tables.py",[("Control alarms","False positives")])
edit("claim_sources.md",[("while normalized increments give 30/30 with no control alarms","while mean absolute increments give 30/30 with no control false positives; normalized increments give 29/30 with two control false positives")])
print("Editorial changes applied.")

