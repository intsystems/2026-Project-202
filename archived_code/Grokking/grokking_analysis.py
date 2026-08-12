import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from search_for_optimal_parameters import (
    delayed_mutual_information, get_first_local_minimum,
    false_nearest_neighbors, find_optimal_E_simplex,
    cao_method, mle_intrinsic_dimension
)



def get_tau_dmi(series, max_tau=15):
    if np.std(series) < 1e-6:
        return 1
    taus, dmi_vals = delayed_mutual_information(series, max_tau=max_tau, bins=20)
    min_idx = get_first_local_minimum(dmi_vals, abs_eps=0.01, drop_fraction=0.01)
    return taus[min_idx]

def get_tau_fixed(series):
    return 1

def get_E_fnn(series, tau):
    if np.std(series) < 1e-6: return 1.0
    fnn_vals = false_nearest_neighbors(series, tau=tau, max_m=15)
    below_thresh = np.where(fnn_vals < 1.0)[0]
    return float(below_thresh[0] + 1 if len(below_thresh) > 0 else np.argmin(fnn_vals) + 1)

def get_E_simplex(series, tau):
    if np.std(series) < 1e-6: return 1.0
    opt_E, _ = find_optimal_E_simplex(series, tau, max_E=15)
    return float(opt_E)

def get_E_cao(series, tau):
    if np.std(series) < 1e-6: return 1.0
    opt_E, _ = cao_method(series, tau, max_E=15)
    return float(opt_E)

def get_E_mle(series, tau):
    if np.std(series) < 1e-6: return 1.0
    return mle_intrinsic_dimension(series, tau, max_E=15, k_neighbors=5)

def analyze_grokking_dimensionality(
    csv_path, 
    tau_func, 
    E_func, 
    method_name="Method",
    window_size=1000, 
    step_size=200,
    target_metric='train_loss',
    add_name=''
):
    print(f"\n--- Running Grokking Analysis: {method_name} ---")
    
    df = pd.read_csv(csv_path)
    
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df['train_acc'] = pd.to_numeric(df['train_acc'], errors='coerce')
    df['val_acc'] = pd.to_numeric(df['val_acc'], errors='coerce')
    df[target_metric] = pd.to_numeric(df[target_metric], errors='coerce')
    
    df = df.dropna(subset=['step', 'train_acc', 'val_acc', target_metric])
    
    all_steps = np.array(df['step'].values, dtype=np.float64)
    series = np.array(df[target_metric].values, dtype=np.float64)
    
    steps_E = []
    E_history = []
    tau_history =[]
    
    for i in tqdm(range(0, len(series) - window_size, step_size), desc="Sliding Window"):
        window_data = series[i : i + window_size]
        
        if np.isnan(window_data).any():
            continue
            
        tau = tau_func(window_data)
        E = E_func(window_data, tau)
        
        center_step = all_steps[i + window_size // 2]
        
        if E is not None and not np.isnan(E) and not np.isinf(E):
            clean_E = max(1.0, min(float(E), 30.0))
        else:
            clean_E = np.nan
            
        steps_E.append(center_step)
        E_history.append(clean_E)
        tau_history.append(tau)

    x_e = np.array(steps_E, dtype=np.float64)
    y_e = np.array(E_history, dtype=np.float64)
    
    x_full = all_steps
    y_tr_acc = np.array(df['train_acc'].values, dtype=np.float64)
    y_val_acc = np.array(df['val_acc'].values, dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_E = 'tab:purple'
    ax1.set_xlabel('Optimization Steps', fontsize=12)
    ax1.set_ylabel(f'Effective Dimensionality $E$ ({method_name})', color=color_E, fontsize=12, fontweight='bold')
    
    line_E, = ax1.plot(x_e, y_e, color=color_E, linewidth=2.5, marker='o', markersize=5, label=f'Dimensionality $E$')
    ax1.tick_params(axis='y', labelcolor=color_E)
    
    valid_E = y_e[~np.isnan(y_e)]
    if len(valid_E) > 0:
        ax1.set_ylim(min(valid_E) - 0.05, max(valid_E) + 0.05)

    ax2 = ax1.twinx()
    color_tr = 'tab:blue'
    color_val = 'tab:red'
    ax2.set_ylabel('Accuracy', color='black', fontsize=12)
    
    line_tr, = ax2.plot(x_full, y_tr_acc, color=color_tr, alpha=0.3, label='Train Acc')
    line_val, = ax2.plot(x_full, y_val_acc, color=color_val, linewidth=2.5, label='Val Acc')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(-0.05, 1.05)

    ax1.set_xlim(x_full.min(), x_full.max())
    ax2.set_xlim(x_full.min(), x_full.max())

    try:
        grok_mask = y_val_acc >= 0.95
        if np.any(grok_mask):
            grok_step = x_full[grok_mask][0]
            ax2.axvline(x=grok_step, color='black', linestyle=':', linewidth=2)
            
            text_x = min(grok_step + 500, x_full.max() - 2000)
            ax2.annotate('Grokking Point', xy=(grok_step, 0.2), xytext=(text_x, 0.2),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                         fontsize=11, fontweight='bold')
    except Exception as e:
        print(f"Warning: Could not plot grokking point - {e}")

    lines =[line_E, line_tr, line_val]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', framealpha=0.9)
    
    plt.title(f'Collapse of Dimensionality during Grokking\nMethod: {method_name} (Metric: {target_metric}, W={window_size})', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    add_str = f"_{add_name}" if add_name else ""
    save_path = f'grokking_dim_collapse_{method_name.replace(" ", "_")}{add_str}.pdf'
        
    plt.savefig(save_path, format='pdf', dpi=300)
    plt.show()
    
    return x_e, y_e, tau_history