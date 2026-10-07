# E12 protocol: early stopping without validation data under label noise

Fixed before the confirmatory runs. The pilots (seed 99) looked only at clean validation
accuracy and the share of fitted noisy labels; no log statistic was computed on them.

Training: label_noise.py, width 16, SGD lr 0.002, momentum 0.9, batch 128, 25 000 steps,
10 000 CIFAR-10 images with symmetric label noise. Clean accuracy on 1 000 held-out images
every 250 steps (validation, truth only) and on another 1 000 (test, for scoring).

Runs. Calibration: noise 0.4, seeds 0-3. Test: noise 0.2, 0.4, 0.6, seeds 10-13 (12 runs).

Stopping rules. Each returns a step; its score is the clean test accuracy at the nearest
evaluation step; regret = best clean test accuracy of the run minus that.
- fixed: the median best-validation step of the calibration runs;
- loss plateau: first step where the 1 000-step mean mini-batch loss fell by less than a
  fraction f over the previous 1 000 steps (f calibrated);
- train accuracy: first step where the 1 000-step mean mini-batch accuracy on the noisy
  labels exceeds a threshold a (calibrated);
- log statistic s (MG and every competitor of cnn_competitors.EXTRA plus crossings,
  lag1, det_std) on parameter norm, mini-batch loss or gradient norm, windows of 1 000
  steps, stride 500: the E6 change detector D_k with (M, B) in {2,3,4} x {3,4,6}, sign
  +-1 and threshold delta from a grid; stop at the first alarm after step 1 500, or at
  the end of training if none.
All free parameters (f, a, log, M, B, sign, delta) are chosen on the calibration runs to
minimise mean regret, then frozen.

Prediction: the MG rule has lower mean test regret than fixed, loss plateau and train
accuracy, averaged over the 12 test runs.
