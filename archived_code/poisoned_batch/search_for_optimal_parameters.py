import numpy as np
from sklearn.neighbors import KDTree
from sklearn.metrics import mutual_info_score

def delayed_mutual_information(x, max_tau=50, bins=50):
    x = np.asarray(x)
    dmi_values =[]
    taus = np.arange(1, max_tau + 1)
    
    hist, bin_edges = np.histogram(x, bins=bins)
    x_binned = np.digitize(x, bin_edges[:-1]) 
    
    for tau in taus:
        x_t = x_binned[:-tau]
        x_t_minus_tau = x_binned[tau:]
        
        mi = mutual_info_score(x_t, x_t_minus_tau)
        dmi_values.append(mi)
        
    return taus, np.array(dmi_values)

import numpy as np

def get_first_local_minimum(y, abs_eps=0.01, drop_fraction=0.01):
    y = np.asarray(y)
    
    drop_eps = y[0] * drop_fraction
    
    for i in range(len(y) - 1):
        if y[i] < abs_eps:
            return i
            
        drop = y[i] - y[i+1]
        if drop < drop_eps:
            return i
            
    return np.argmin(y)





def delay_embedding(x, m, tau):
    N = len(x) - (m - 1) * tau
    if N <= 0:
        raise ValueError(f"Time series length {len(x)} is too short for m={m}, tau={tau}")
    return np.column_stack([x[i : i + N] for i in range(0, m * tau, tau)])

def false_nearest_neighbors(x, tau=1, max_m=10, Rtol=10.0, Atol=2.0):
    x = np.asarray(x)
    sigma = np.std(x)
    fnn_percent =[]

    for m in range(1, max_m + 1):
        X_m = delay_embedding(x, m, tau)
        tree = KDTree(X_m)
        dists, idx = tree.query(X_m, k=2)

        R_d = dists[:, 1]
        nn_idx = idx[:, 1]

        max_idx = len(x) - m * tau
        valid_mask = (np.arange(len(X_m)) < max_idx) & (nn_idx < max_idx)

        if np.sum(valid_mask) == 0:
            fnn_percent.append(0.0)
            continue

        R_d = R_d[valid_mask]
        nn_idx = nn_idx[valid_mask]
        current_indices = np.arange(len(X_m))[valid_mask]

        x_next = x[current_indices + m * tau]
        x_next_nn = x[nn_idx + m * tau]

        dist_increase = np.abs(x_next - x_next_nn)
        R_d1 = np.sqrt(R_d**2 + dist_increase**2)

        epsilon = 1e-10
        is_false_1 = (dist_increase / (R_d + epsilon)) > Rtol
        is_false_2 = (R_d1 / sigma) > Atol

        is_false = is_false_1 | is_false_2
        pct = 100.0 * np.sum(is_false) / len(is_false)
        fnn_percent.append(pct)

    return np.array(fnn_percent)