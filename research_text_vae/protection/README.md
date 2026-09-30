# Protected third branch of the text VAE experiment

Read `report_ru.pdf` / `report_ru.md` for the Russian report. The parent report includes the original two-arm experiment and this control. `PROTOCOL.md` records the choices fixed before the protected branches were trained.

The original paired runs and their archived branch checkpoints are reused, not retrained. These are new continuations of the same ten initializations, not ten additional independent datasets.

1. `encoder5/seed0` is the failed protection pilot: five extra encoder-only steps at beta=1. Its failure is assessed by independent reference diagnostics, without MG selection.
2. `freebits/seed0` tests the prespecified fallback: beta=1, per-coordinate batch-averaged KL floored at 0.5 nats. This changes the loss shape; it is not a pure control for beta.
3. `selection.json` records selection based only on the pilot reference diagnostics, before calculating its protected MG. `confirm_protection.py` then runs all seeds 1–9 regardless of MG outcomes, with at most two concurrent jobs.
4. `freebits/seed*/comparison.json` contains all three branches, primary MG and sensitivity settings. `three_arm_windows.csv` retains individual windows. Each directory includes final weights, scalar logs, reference diagnostics, and checkpoint/RNG audit metadata.
5. `all_seeds.csv` and `summary.json` include every seed, with failed independent protections flagged rather than hidden. The pilot is excluded from confirmation statistics.

From the project root, after all branches finish:

```powershell
python research_text_vae/protection/summarize_protection.py
python research_text_vae/protection/verify_protection.py
python research_text_vae/protection/build_report.py
python research_text_vae/make_report.py
python research_text_vae/build_report.py
python research_text_vae/package_results.py
```

Existing complete training branches are not overwritten. To regenerate analyses, run `analyze_protection.py --mode freebits --seed N` for each N=0,...,9. Retraining requires a new output directory and the original cached PTB data (see parent README / `run.py`). The package excludes raw corpus files but includes their provenance and the training code.

Ideas used: He et al., *Lagging Inference Networks and Posterior Collapse in Variational Autoencoders*, ICLR 2019, arXiv:1901.05534; Kingma et al., *Improved Variational Inference with Inverse Autoregressive Flow*, NeurIPS 2016, arXiv:1606.04934, Appendix C. The fixed five-step variant is an adaptation, not an exact reproduction of He et al.
