# Frozen temporal transfer check

Written after first confirmation. All selectors retain pilot-only thresholds. Evaluate seeds3--5, all8 arms at prefix starts8192 and16384, each4096 samples and512-step forecast. Do not alter predictors or thresholds. This is a sensitivity analysis, not six additional independent seeds. Report the primary result unchanged even if this is more favorable. Cost metric is squared normalized error (MSE divided by prefix variance); the original protocol's NRMSE wording was inaccurate.
