"""One-time editorial integration of the audited practical-utility results."""
from pathlib import Path
H=Path(__file__).resolve().parent
def change(file,old,new):
 p=H/file;s=p.read_text(encoding='utf-8');assert old in s,(file,old[:60]);p.write_text(s.replace(old,new),encoding='utf-8')
(H/'sections/abstract.tex').write_text(r'''\begin{abstract}
\claude{Neural network training can restrict parameter motion, reduce latent information, or produce more regular behavior. We propose a protocol for detecting such changes from one scalar time series using delay reconstruction, pooled local-dimension estimation, and calibrated decision rules. Dynamical-systems theory motivates an active-component interpretation under suitable recurrence and observation conditions; controlled experiments distinguish independent phases from harmonics. Applied tests track training restrictions in CNNs, information loss and preservation in a text VAE, and changes in learned control. Beyond diagnosis, the scalar estimate selects forecasting models for recurrent-network activity and policies for changed actuator dynamics. A 128-query approximation preserves all 160 full-estimator forecast choices on a separate 20-seed cohort. These results connect scalar geometry to useful decisions with low data requirements and measured computational savings. Absolute dimension recovery requires the stated assumptions; practical accuracy and cost depend on the observable, decision task, and competing baseline.}
\end{abstract}
''',encoding='utf-8')
change('sections/introduction.tex',r'\item \textbf{Independent validation in applied learning tasks.} MG detects 27/28 strong CNN interventions with no false positives, transfers to backbone freezing in ResNet-18, tracks both information loss and preservation in a text VAE, and responds to changes in learned Walker2d behavior and grokking trajectories. We compare it with simple scalar statistics using the same records and calibration procedure.',r'\item \textbf{Independent validation and useful decisions.} We validate changes against training restrictions, latent information in a text VAE, and learned behavior. MG also selects forecast models from one neuron and RL policies under actuator shifts; strong scalar and validation-based competitors remain in the comparisons.')
change('sections/introduction.tex',r'\item \textbf{Measured savings in analysis and storage.} We separate the cost of obtaining a log from processing it. A CNN window takes 9.5\,ms to analyze; in the specified learned recurrent benchmark, MG is 14--19 times faster than the full Lyapunov spectrum while detecting the same transition. Simpler statistics can be faster, so this is a comparison with a full-state reference, not a universal speed claim.',r'\item \textbf{A fast sampled-query implementation.} On 20 new generator seeds, 128 queries preserve all 160 full-MG forecast choices with a 16.3-fold measured speedup. We separately report acquisition cost and comparisons with full-state diagnostics, including a 14--19-fold speedup over full Lyapunov spectra in the specified benchmark.')
change('sections/applications.tex',r'\section{Applied experiments: monitoring learned simplification}',r'\section{Applications: monitoring and decisions}')
p=H/'sections/applications.tex';s=p.read_text(encoding='utf-8');a=s.index('We evaluate three practical uses:');b=s.index(r'\subsection{CNN training',a)
s=s[:a]+r'''We test scalar geometry against independent measurements of restricted training, latent information loss, and learned behavior, then use it to select policies and forecasters. Predictive utility is evaluated separately from absolute dimension interpretation.

'''+s[b:]
a=s.index(r'\paragraph{Scalar baselines and spectral surrogates.}');b=s.index(r'\paragraph{Causal detection',a)
s=s[:a]+r'''MG separates interventions from unchanged-training and increased-batch controls with AUC 0.93, versus 0.48--0.57 for trend crossings, lag-one correlation, and detrended standard deviation. IAAFT spectral surrogates yield 0.68. Appendix~\ref{app:baselines} specifies the descriptive surrogate check, all scalar competitors, and alternative comparison sets.

'''+s[b:]
a=s.index('Table~\ref{tab:detectors} includes every tested scalar feature.') if False else s.index(r'Table~\ref{tab:detectors} includes every tested scalar feature.')
b=s.index(r'\begin{table}',a)
s=s[:a]+r'''MG detects 27/28 strong interventions with 0/16 control false positives, while normalized increments give 28/28 and 0/16 (Table~\ref{tab:detectors}). MG detects 8/28, 16/28, and 22/28 within 500, 1,000, and 2,000 updates. On ten additional seeds with shifted intervention times, it gives 19/30 detections and false positives in five of ten controls, versus 30/30 and zero for absolute increments. Thus detector transfer is task-specific; these paired records share seeds (Appendix~\ref{app:baselines}).

'''+s[b:]
s=s.rstrip();assert s.endswith('}');s=s[:-1]+r'\input{sections/forecast_utility}'+'\n}\n';p.write_text(s,encoding='utf-8')
p=H/'sections/cost_discussion.tex';s=p.read_text(encoding='utf-8')
a=s.index(r'\paragraph{Runtime relative');b=s.index(r'\paragraph{Observable sensitivity',a)
s=s[:a]+r'''\paragraph{When analysis saves computation.}
In the 256-unit FORCE benchmark, MG and the full Lyapunov spectrum identify the learned transition toward a cycle. The measured speedup is 14--19-fold, or 5--10-fold including an embedding-doubling check. A 128-oscillator test gives 22--23-fold savings. These comparisons depend on window length and implementation; elementary scalar features and the largest exponent can be faster. MG does not recover individual exponents (Appendix~\ref{app:cost}).

In forecast selection, sampled-query MG reduces computation while retaining full-MG decisions on the held-out cohort (Section~\ref{sec:forecast_utility}). Conversely, VAE probe acquisition dominates analysis: cyclic monitoring costs an estimated 54.68\,s versus 2.72\,s for periodic checks; reusing the probe reduces MG's total to 3.54\,s. Low analysis cost is most useful when the informative scalar is already available.

'''+s[b:]
a=s.index(r'\section{Conclusion}')
s=s[:a]+r'''\section{Conclusion}
Delay-embedding theory motivates our scalar-log protocol; controlled tests connect its estimates to recurrent phase structure. Applied experiments validate signals of training restriction, latent information loss, and behavioral change. Policy and forecast selection turn these measurements into decisions. Sampled-query MG preserves full-MG forecast choices at substantially lower measured cost. These benefits require a suitable observable and task-specific validation; neither a decline in MG nor a successful decision alone proves an exact change in active dimension.
}
''';p.write_text(s,encoding='utf-8')
change('aistats2027.tex',r'\input{sections/appendix}',r'\input{sections/appendix}'+'\n'+r'\input{sections/practical_appendix}')
change('sections/practical_appendix.tex','crossed with tasks T1, T2, H2, and T4, seven interleaved repetitions per input after warm-up. The realized records in the saved timing table are T1, T2, H4, and T4; H4 is the harmonic task used in all reported comparisons.', 'crossed with tasks T1, T2, H4, and T4, seven interleaved repetitions per input after warm-up.')
print('Integrated practical results into abstract, contributions, applications, discussion and appendix.')
