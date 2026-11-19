# Diagnostics of model dynamics

Poster experiments that use scalar logs to detect backbone freezing, loss of latent information, or a learned cycle.

- [ResNet-18 backbone freeze](resnet_freeze/ReadMe.md). Use a weight-norm log to detect backbone freezing on CIFAR-10. Compare scratch and ImageNet initializations.
- [Text VAE code loss](text_vae_code_loss/ReadMe.md). Compare baseline, code suppression and free bits on Penn Treebank. Plot scalar reconstruction loss, rolling MG and independently measured mutual information.
- [FORCE RNN cycle](force_rnn_cycle/ReadMe.md). Compare one neuron's autonomous activity before and after learning a figure-eight reference. Evaluation holds weights fixed.

[Recorded model runs](../ReadMe.md)
