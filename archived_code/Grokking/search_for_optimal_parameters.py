import numpy as np
from sklearn.neighbors import KDTree
from sklearn.metrics import mutual_info_score
from causal_ccm.causal_ccm import ccm
from scipy.spatial.distance import cdist

def delayed_mutual_information(x, max_tau=50, bins=50):
    x = np.asarray(x)
    dmi_values =[]
    taus = np.arange(1, max_tau + 1)
    
    if np.std(x) < 1e-8:
        return taus, np.zeros(len(taus))
        
    hist, bin_edges = np.histogram(x, bins=bins)
    x_binned = np.digitize(x, bin_edges[:-1]) 
    
    for tau in taus:
        x_t = x_binned[:-tau]
        x_t_minus_tau = x_binned[tau:]
        
        mi = mutual_info_score(x_t, x_t_minus_tau)
        dmi_values.append(mi)
        
    return taus, np.array(dmi_values)

def get_first_local_minimum(y, abs_eps=0.01, drop_fraction=0.01):
    y = np.asarray(y)
    if len(y) == 0 or np.all(y == 0):
        return 0
        
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
    x = x + np.random.normal(0, 1e-10, size=len(x))
    
    sigma = np.std(x)
    fnn_percent =[]

    for m in range(1, max_m + 1):
        try:
            X_m = delay_embedding(x, m, tau)
        except ValueError:
            fnn_percent.append(0.0)
            continue
            
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
        is_false_2 = (R_d1 / (sigma + epsilon)) > Atol

        is_false = is_false_1 | is_false_2
        pct = 100.0 * np.sum(is_false) / len(is_false)
        fnn_percent.append(pct)

    return np.array(fnn_percent)


def find_optimal_E_simplex(series, tau=1, max_E=15, Tp=1):
    series = np.asarray(series)
    E_skills = []
    
    for E in range(1, max_E + 1):
        N = len(series)
        len_embed = N - (E - 1) * tau - Tp
        if len_embed < 10:
            break
            
        X = np.array([series[i : i + E*tau : tau] for i in range(len_embed)])
        Y = np.array([series[i + (E-1)*tau + Tp] for i in range(len_embed)])
        
        predictions = []
        actuals = []
        
        split = int(len_embed * 0.7)
        X_lib, Y_lib = X[:split], Y[:split]
        X_test, Y_test = X[split:], Y[split:]
        
        for i, target_point in enumerate(X_test):
            distances = np.linalg.norm(X_lib - target_point, axis=1)
            
            k = E + 1
            nearest_indices = np.argsort(distances)[:k]
            nearest_dists = distances[nearest_indices]
            
            weights = np.exp(-nearest_dists / (nearest_dists[0] + 1e-8)) 
            weights /= np.sum(weights)
            
            pred_y = np.sum(weights * Y_lib[nearest_indices])
            predictions.append(pred_y)
            actuals.append(Y_test[i])
            
        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
        E_skills.append(rmse)
        
    if not E_skills:
        return 1, []
        
    optimal_E = np.argmin(E_skills) + 1 
    
    return optimal_E, E_skills


def cao_method(series, tau, max_E=15):
    series = np.asarray(series)
    series = series + np.random.normal(0, 1e-10, size=len(series))
    
    E_values =[]
    
    for d in range(1, max_E + 1):
        try:
            emb_d_plus_1 = delay_embedding(series, d + 1, tau)
            length = len(emb_d_plus_1)
            emb_d = delay_embedding(series, d, tau)[:length]
        except ValueError:
            break
            
        if length < 10:
            break
            
        tree = KDTree(emb_d, metric='chebyshev')
        distances_d, indices = tree.query(emb_d, k=2)
        
        R_d = distances_d[:, 1]
        nn_indices = indices[:, 1]
        
        R_d = np.maximum(R_d, 1e-10)
        
        x_next = series[np.arange(length) + d * tau]
        x_next_nn = series[nn_indices + d * tau]
        
        diff_d_plus_1 = np.abs(x_next - x_next_nn)
        R_d_plus_1 = np.maximum(R_d, diff_d_plus_1)
        
        a_i_d = R_d_plus_1 / R_d
        E_values.append(np.mean(a_i_d))
        
    E1_real =[]
    for i in range(1, len(E_values)):
        E1_real.append(E_values[i] / E_values[i-1])
            
    if not E1_real:
        return 1, [1.0]

    diff = np.abs(np.diff(E1_real))
    plateau_indices = np.where(diff < 0.05)[0]
    
    optimal_E = plateau_indices[0] + 2 if len(plateau_indices) > 0 else max_E
    E1_plot =[1.0] + E1_real
    
    return optimal_E, E1_plot


def mle_intrinsic_dimension(series, tau, max_E=15, k_neighbors=5):
    series = np.asarray(series)
    series = series + np.random.normal(0, 1e-9, size=len(series))
    
    try:
        embedded_data = delay_embedding(series, max_E, tau)
    except ValueError:
        return np.nan
        
    if len(embedded_data) < k_neighbors + 2:
        return np.nan
        
    tree = KDTree(embedded_data)
    distances, _ = tree.query(embedded_data, k=k_neighbors + 1)
    distances = distances[:, 1:] 
    
    distances = np.maximum(distances, 1e-8)
    R_k = distances[:, -1:]
    
    log_ratios = np.log(R_k / distances[:, :-1])
    sum_log_ratios = np.sum(log_ratios, axis=1)
    sum_log_ratios = np.maximum(sum_log_ratios, 1e-5)
    
    local_id = (k_neighbors - 1) / sum_log_ratios
    local_id = local_id[np.isfinite(local_id)]
    
    if len(local_id) == 0:
        return np.nan
        
    global_id = np.mean(local_id)
    
    if global_id > max_E * 2: 
        return float(max_E)
        
    return global_id