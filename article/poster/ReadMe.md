# ICOMP poster source

Edit [main.tex](main.tex) for the 120 × 90 cm poster. The [figure directory](figures/ReadMe.md) contains the vector panels. The ICOMP logo and white MIPT logo are the two PNGs next to the TeX source. The original layout includes the ASCOMP 2026 badge beneath the ICOMP logo.

## Build

From this directory, run:

```bash
latexmk -pdf '-outdir=.build' -interaction=nonstopmode -halt-on-error main.tex
```

The build writes `.build/main.pdf`. A full TeX Live installation provides the required LaTeX packages and fonts.

[Reference poster PDF](../poster_120x90.pdf)

## Poster blocks and experiments

### Block 04 · Synthetic tests with a known active dimension

- [RNNs learning four-tone signals](../../code/synthetic/poster_examples/four_tone_rnn/ReadMe.md).
- [Linear learning](../../code/synthetic/poster_examples/known_linear_modes/ReadMe.md).

### Block 05 · Detect restricted training in ResNet

- [ResNet-18 on CIFAR-10](../../code/real/scalar_log_diagnostics/resnet_freeze/ReadMe.md).

### Block 06 · Diagnose code loss and cycle formation

- [FORCE-trained recurrent network](../../code/real/scalar_log_diagnostics/force_rnn_cycle/ReadMe.md).
- [Text VAE on Penn Treebank](../../code/real/scalar_log_diagnostics/text_vae_latent_code/ReadMe.md).

### Block 07 · Select models and stop training

- [MNIST classification with 40% noisy labels](../../code/real/model_selection_and_stopping/mnist_early_stopping/ReadMe.md).
- [Forecasting one neuron in a trained RNN](../../code/real/model_selection_and_stopping/rnn_forecast_selection/ReadMe.md).
- [Walker2d policy selection](../../code/real/model_selection_and_stopping/walker2d_policy_selection/ReadMe.md).

[Article and poster index](../ReadMe.md)
