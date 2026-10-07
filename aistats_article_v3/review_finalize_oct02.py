from pathlib import Path
import re, json, hashlib, subprocess
H=Path(__file__).resolve().parent
def subfile(name,changes):
 p=H/name;s=p.read_text(encoding="utf-8")
 for a,b in changes:
  assert a in s,(name,a[:80])
  s=s.replace(a,b)
 p.write_text(s,encoding="utf-8")
subfile("sections/applications.tex",[
("an detection threshold","a detection threshold"),
("counts control records with a threshold crossing","denotes control records with a threshold crossing"),
("MG tracks changes across learning tasks. Left: CNN estimates after an intervention divided by those before it, on four held-out seeds. Bars show medians; the dashed line means no change. Center: the same comparison for ResNet-18 trained from scratch or pretrained weights, with all seeds shown. Right: late MG in the suppressed and free-bits VAE branches relative to their shared low-regularization control, across nine test seeds. Independent update and information measurements validate the changes.",
"MG under training interventions and latent-information suppression. Left: post/pre-intervention ratios on four CNN seeds; vertical marks indicate medians. Center: corresponding ResNet-18 ratios for scratch and pretrained initialization. Right: late VAE estimates relative to the shared low-regularization control, for suppression and free-bits branches on nine seeds. Points represent seeds; dashed lines indicate a ratio of one. Independent update and information measurements provide the references.")
])
subfile("sections/introduction.tex",[("An exact component dimension is not required","Exact recovery of active dimension is not required")])
subfile("sections/abstract.tex",[("one neuron's activity distinguishes independent phases from harmonics that the number of spectral peaks cannot separate","one neuron's activity distinguishes phase structures with identical numbers of spectral peaks")])
subfile("sections/cost_discussion.tex",[
("Time to analyze an available record. The FORCE reference is the full Lyapunov spectrum; the VAE reference combines latent-information and code-shuffling measurements. FORCE ranges cover before and after training. Compare methods within a row: tasks use different records and configurations. Scalar acquisition is excluded.",
r"Analysis time for an available record, excluding scalar acquisition. FORCE is compared with the full Lyapunov spectrum; its ranges cover pre/post-training medians. The VAE reference combines latent-information and code-shuffling measurements. Hardware, thread settings, and timing protocols are documented in Appendix~\ref{app:hardware}; comparisons are within each row.")
])
hardware=r"""
\subsection{Computing infrastructure and provenance}\label{app:hardware}
The FORCE timing report identifies an Intel Core i5-12500H CPU. Its saved environment specifies Windows 10 (build 18363), Python 3.13.12, NumPy 2.4.3, SciPy 1.17.1, and one BLAS thread; the benchmark uses float64 arithmetic and three warmed sequential repetitions. No GPU is used in this benchmark.

The cyclic-VAE benchmark records CPU execution on Windows 10 (build 18363), Python 3.13.12, and PyTorch 2.10.0+cpu, with two PyTorch threads, one estimator thread, and forty interleaved repetitions. Its metadata do not identify the CPU model. The Walker timing metadata record the same operating system and Python version, an Intel64 Family 6 Model 154 Stepping 3 processor, one thread, and five repetitions.

For the new ten-seed CNN confirmation, the local machine identifies an Intel Core i5-12500H (12 physical cores, 16 logical processors); the training script uses CPU execution with six PyTorch threads. This machine identification was collected during manuscript revision on October 2, 2026 and does not retrospectively establish the hardware of every earlier experiment. The historical CNN timing output does not record a CPU model. Absolute runtimes and speed ratios therefore describe the documented local implementations; a complete hardware inventory for all historical runs is unavailable.

\subsection{Benchmark-specific timing and storage}
"""
subfile("sections/appendix.tex",[
("We also ran a confirmation with the protocol fixed before training on ten new seeds with the historical rules frozen before reading the new logs.","We evaluated ten additional seeds using a protocol and historical detector rules fixed before training."),
("This harder transfer check produced 19/30 intervention hits","This transfer test produced 19/30 intervention detections"),
(r"\paragraph{CNN.}",hardware+"\n"+r"\paragraph{CNN.}")
])
subfile("sections/checklist.tex",[
(r"\section*{Paper Checklist}",r"\section*{Checklist}"),
("[No] The manuscript bundle contains figure/table reconstruction scripts and numerical evidence; a complete anonymized training-software release is separate.","[No] The bundle includes analysis scripts and numerical evidence, but not the complete training software with all dependencies."),
("[Yes] The source bundle includes saved numerical outputs and scripts to reproduce the plotted and tabulated results. It does not include all weights or raw corpora needed to repeat every training run.","[No] Saved outputs and scripts reproduce tables and figures; the materials required to repeat every training run are not fully bundled."),
("[Yes] All figures show individual records or seeds; ratios, medians, detection totals and the VAE bootstrap are defined explicitly.",r"[Yes] Captions and Appendix~\ref{app:settings} specify statistics, replication units, and the VAE bootstrap."),
("[No] CPU/thread settings are specified for the main timing comparisons, but a complete hardware inventory for every legacy run is not available in this manuscript.",r"[No] Appendix~\ref{app:hardware} documents recovered CPU, software, and thread settings; hardware metadata are incomplete for some historical runs.")
])
# Record only public hardware/software facts, not account names or absolute paths.
dest=H/"evidence"/"compute_provenance";dest.mkdir(exist_ok=True)
sources={}
for name,rel in {
 "force_environment":"research_force_motion/environment.json",
 "vae_environment":"research_text_vae/environment.json",
 "walker_environment":"research_walker_smooth_lambda/timing_environment.json",
}.items():
 p=H.parent/rel
 d=json.loads(p.read_text(encoding="utf-8"))
 d.pop("source",None)
 (dest/(name+".json")).write_text(json.dumps(d,indent=2),encoding="utf-8")
 sources[name]={"source":rel,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
report=H.parent/"research_force_motion/report_ru.md"
for line in report.read_text(encoding="utf-8").splitlines():
 if "i5-12500H" in line:
  (dest/"force_hardware_report.md").write_text(line+"\n",encoding="utf-8")
snapshot={"recorded_on":"2026-10-02","purpose":"Revision-time hardware identification; not a retrospective per-run log",
 "processor":"Intel Core i5-12500H","physical_cores":12,"logical_processors":16,
 "historical_sources":sources,
 "official_template_sha256":"aac31ecf2e41f5a2b7f21d00094bfc2fc1207c34d9f955e991b66f3dc98fdb9b",
 "checklist_rule":"Yes, No, Not Applicable; 1-2 sentence justifications encouraged; do not change questions"}
(dest/"provenance.json").write_text(json.dumps(snapshot,indent=2),encoding="utf-8")
print("Caption, checklist, hardware and numerical attribution fixed.")

