"""The scalar observables written to the log.

The analysis package reconstructs the optimizer's phase space from *one* of these
columns at a time (``edm.sliding_dimension(target_metric=...)``), so each one has
to be a clean 1-D series sampled on a fixed step grid.  ``weight_norm`` is the
non-generic observable of Figs. 1-2; ``val_loss`` / ``train_loss`` are the generic
ones of Fig. 3 and App. B.
"""

import numpy as np
import torch


def weight_norm(model):
    """``||w||_2`` over every parameter, flattened into one vector.

    Accumulates in Python floats before the square root, exactly as the original
    ``grokking_utils.get_weight_norm`` did, so the column is comparable digit for
    digit with the published logs.
    """
    return float(np.sqrt(sum(p.detach().pow(2).sum().item() for p in model.parameters())))


def accuracy(logits, targets):
    return float((logits.argmax(dim=1) == targets).float().mean())


class GradientProbe:
    """Per-step gradient diagnostics: norm, embedding norm, step-to-step cosine.

    Call :meth:`update` after ``loss.backward()`` and before ``zero_grad()``;
    ``optimizer.step()`` does not touch ``.grad``, so either side of it works.
    The cosine between consecutive gradients is 0.0 on the first step, matching
    the original notebooks.
    """

    KEYS = ("grad_norm", "embed_grad_norm", "grad_cosine")

    def __init__(self):
        self._previous = None
        self.values = dict.fromkeys(self.KEYS, 0.0)

    @torch.no_grad()
    def update(self, model):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads:
            return self.values

        flat = torch.cat([g.reshape(-1) for g in grads])
        embed_grad = getattr(model, "embedding_weight", None)
        embed_grad = None if embed_grad is None else embed_grad.grad

        self.values = {
            "grad_norm": float(torch.sqrt(sum(g.pow(2).sum() for g in grads))),
            "embed_grad_norm": 0.0 if embed_grad is None else float(embed_grad.norm(2)),
            "grad_cosine": 0.0 if self._previous is None else float(
                torch.nn.functional.cosine_similarity(
                    flat.unsqueeze(0), self._previous.unsqueeze(0)
                )
            ),
        }
        self._previous = flat.clone()
        return self.values
