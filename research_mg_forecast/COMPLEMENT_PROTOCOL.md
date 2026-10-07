# Incremental value of MG above cheap forecasting selectors

Third exploratory hypothesis after seeing that recurrence wins earlier comparisons. A low-capacity decision tree may combine MG and recurrence. The last temporal block is not used by previous evaluations.

Training remains the original pilot seeds1--2, first4096 samples. Target for tree regression is AR minus periodic prediction loss; positive predicted difference selects periodic. All trees identical fixed max_depth2, min_samples_leaf3, seed0, no feature scaling or hyperparameter search. Feature sets: recurrence; MG; recurrence+MG; all cheap (entropy,increments,recurrence); all cheap+MG. Include original frozen MG/recurrence thresholds, prefix-holdout winner, fixed AR, fixed periodic and oracle.

Test seeds3--5 all8 arms, prefix[24576:28672], forecast[28672:29184]. Choices only inspect prefix. All rules fitted/exported before reading final-block targets. Previous outcomes motivated the comparison; the final temporal block is fresh but belongs to the same networks, so this is temporal transfer, not independent-seed confirmation. Do not replace primary tables or claim dozens of independent runs. Keep every result regardless of sign. Record error and feature/forecast selection computation; no saving based on future test outcomes.
