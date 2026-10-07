"""Ground truth: loss-spike onsets from the mini-batch loss log only (fixed in pilots 3-4, seed 100).

Work on l_t = log10(mini-batch loss).
  b_t   = median of l over the trailing reference [t-REF, t-1]
  s_t   = mean of l over [t, t+PERSIST-1]        (the jump must persist, not one bad batch)
  onset at t if s_t - b_t > JUMP (a >= 10x rise of the loss) and s_t - b_t > K * sigma_t,
        sigma_t = 1.4826 * MAD of the reference window.
  Each onset opens a spike period [onset, onset+COOL); the search resumes after it with a
  fresh trailing reference. Non-finite loss (divergence) counts as an onset at the first
  non-finite step.
"""
import numpy as np

REF, PERSIST, JUMP, K, COOL = 200, 10, 1.0, 6.0, 300


def spikes(loss, start=REF, jump=JUMP, k=K):
    raw = np.asarray(loss, float)
    fin = np.isfinite(raw)
    nf = int(np.argmin(fin)) if not fin.all() else len(raw)
    x = np.log10(np.maximum(raw[:nf], 1e-30))
    out = []
    t = start
    while t < len(x) - PERSIST:
        ref = x[t - REF:t]
        b = np.median(ref)
        sig = 1.4826 * np.median(np.abs(ref - b))
        d = x[t:t + PERSIST].mean() - b
        if d > jump and d > k * sig:
            out.append(t)
            t += COOL
        else:
            t += 1
    if nf < len(raw):
        out.append(nf)
    return out
