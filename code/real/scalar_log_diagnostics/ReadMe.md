# Diagnostics of model dynamics

These experiments use scalar logs to detect backbone freezing, loss of latent information, or a learned cycle.

**In the article** · [Section 3.2 · FORCE training from chaos to periodic output](../../../article/manuscript/ReadMe.md#section-32--force-training-from-chaos-to-periodic-output) · [Section 4.1 · Detecting restricted parameter updates](../../../article/manuscript/ReadMe.md#section-41--detecting-restricted-parameter-updates) · [Section 4.2 · Tracking sentence information in a text VAE](../../../article/manuscript/ReadMe.md#section-42--tracking-sentence-information-in-a-text-vae)

- [ResNet-18 backbone freeze](resnet_freeze/ReadMe.md). Use a weight-norm log to detect backbone freezing on CIFAR-10. Compare scratch and ImageNet initializations.
- [Sentence information in a text VAE](text_vae_latent_code/ReadMe.md). Compare KL penalties with and without free bits on Penn Treebank. Track reconstruction loss, rolling MG and mutual information.
- [FORCE RNN cycle](force_rnn_cycle/ReadMe.md). Compare one neuron's autonomous activity before and after learning a figure-eight reference. Evaluation holds weights fixed.

[Recorded model runs](../ReadMe.md)
