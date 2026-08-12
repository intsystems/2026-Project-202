import os
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots as plt_subplots
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import warnings

from search_for_optimal_parameters import delayed_mutual_information, get_first_local_minimum, false_nearest_neighbors
from visualisation_ccm import viz_ccm
from causal_ccm.causal_ccm import ccm

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 14, 'font.family': 'serif',
    'axes.labelsize': 16, 'axes.titlesize': 18,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 12, 'figure.figsize': (10, 6),
    'lines.linewidth': 2.0, 'axes.grid': True, 'grid.alpha': 0.3,
})

def find_and_plot_optimal_tau(
    ts_data, exp_name, out_dir, 
    max_tau_search=50, strict_max_tau=2,
    bins=50, abs_eps=0.02, drop_fraction=0.02,
    **dmi_kwargs
):
    dmi_results = {}
    optimal_taus = {}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'val_loss': 'tab:blue', 'val_accuracy': 'tab:orange', 'poison_fraction': 'tab:green'}
    text_outline = [path_effects.withStroke(linewidth=2, foreground="white")]

    for name, series in ts_data.items():
        taus, dmi_vals = delayed_mutual_information(series, max_tau=max_tau_search, bins=bins, **dmi_kwargs)
        dmi_results[name] = dmi_vals
        
        min_idx = get_first_local_minimum(dmi_vals, abs_eps=abs_eps, drop_fraction=drop_fraction)
        found_tau = taus[min_idx]
        
        opt_tau = min(found_tau, strict_max_tau)
        optimal_taus[name] = opt_tau

        line, = ax.plot(taus, dmi_vals, label=name, color=colors.get(name, 'black'), alpha=0.8)
        opt_dmi = dmi_vals[opt_tau - 1]
        
        ax.plot(opt_tau, opt_dmi, marker='v', markersize=10, 
                color=colors.get(name, 'black'), markeredgecolor='white', markeredgewidth=1.5, zorder=10)
        
        ax.annotate(f"$\\tau={opt_tau}$", 
                    xy=(opt_tau, opt_dmi), xytext=(0, 15), textcoords="offset points",
                    ha='center', fontsize=12, color=colors.get(name, 'black'), fontweight='bold',
                    path_effects=text_outline, zorder=20)

    ax.set_xlabel(r"Time Delay $\tau$ (steps)")
    ax.set_ylabel("Mutual Information $I(\tau)$")
    ax.set_title(f"[{exp_name}] Delayed Mutual Information (Max $\\tau={strict_max_tau}$)")
    ax.legend()
    plt.tight_layout()
    
    save_path = os.path.join(out_dir, f"{exp_name}_DMI_tau.pdf")
    plt.savefig(save_path, format='pdf', dpi=300)
    plt.close()
    
    return optimal_taus


def find_and_plot_optimal_E(
    ts_data, optimal_taus, exp_name, out_dir, 
    max_m_search=15, fnn_threshold=1.0,
    **fnn_kwargs
):
    
    m_data = {}
    colors = {'val_loss': 'tab:blue', 'val_accuracy': 'tab:orange', 'poison_fraction': 'tab:green'}
    
    for name, series in ts_data.items():
        tau = optimal_taus[name]
        fnn_vals = false_nearest_neighbors(series, tau=tau, max_m=max_m_search, **fnn_kwargs)
        m_data[name] = fnn_vals

    fig, ax = plt.subplots(figsize=(12, 6))
    text_groups = {}
    dims = np.arange(1, max_m_search + 1)

    optimal_E = {}

    for name, values in m_data.items():
        vals = np.asarray(values)
        
        below_threshold = np.where(vals < fnn_threshold)[0]
        opt_m_idx = below_threshold[0] if len(below_threshold) > 0 else np.argmin(vals)
        opt_m = dims[opt_m_idx]
        optimal_E[name] = int(opt_m)
        min_val = vals[opt_m_idx]

        line_color = colors.get(name, 'black')
        ax.plot(dims, vals, label=f"{name} ($\\tau$={optimal_taus[name]})", 
                color=line_color, marker='o', markersize=6, alpha=0.8)

        ax.plot(opt_m, min_val, marker='s', markersize=10, color=line_color,
                 markeredgecolor='white', markeredgewidth=1.5, zorder=10)

        txt_label = f"E={opt_m}"
        if opt_m not in text_groups:
            text_groups[opt_m] =[]
        text_groups[opt_m].append({'y': min_val, 'text': txt_label, 'color': line_color})

    Y_PROXIMITY_THRESHOLD = 5.0
    text_outline = [path_effects.withStroke(linewidth=2, foreground="white")]

    for x_val, items in text_groups.items():
        items.sort(key=lambda item: item['y'])
        base_start_offset = 15 if (int(x_val) % 2 == 0) else 35
        current_stack_level = 0
        last_y = -9999

        for item in items:
            if abs(item['y'] - last_y) < Y_PROXIMITY_THRESHOLD:
                current_stack_level += 1
            else:
                current_stack_level = 0

            vertical_offset = base_start_offset + (current_stack_level * 18)
            ax.annotate(item['text'], xy=(x_val, item['y']), xytext=(0, vertical_offset),
                textcoords="offset points", ha='center', fontsize=11, color=item['color'],
                fontweight='bold', zorder=20, path_effects=text_outline,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6))
            last_y = item['y']

    ax.set_xlabel("Embedding Dimension $E$")
    ax.set_ylabel("False Nearest Neighbors (%)")
    ax.set_title(f"[{exp_name}] FNN for Embedding Dimension $E$")
    ax.legend(loc='upper right')
    
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{exp_name}_FNN_E.pdf")
    plt.savefig(save_path, format='pdf', dpi=300)
    plt.close()

    E_global = max(optimal_E.values())
    return optimal_E, E_global


def run_all_ccm(
    ts_data, optimal_taus, E_global, exp_name, out_dir, 
    step_divisor=60, **viz_kwargs
):
    N_points = len(ts_data['val_loss'])
    max_tau = max(optimal_taus.values())

    safe_min_L = (E_global - 1) * max_tau + 10
    if safe_min_L >= N_points:
        safe_min_L = N_points // 2 

    step = max(1, (N_points - safe_min_L) // step_divisor)
    L_range = list(range(safe_min_L, N_points, step))

    print(f"[{exp_name}] Running CCM. Global E={E_global}, L_range: {len(L_range)} steps.")

    def get_title(x_n, y_n, tx, ty):
        return rf"CCM: {x_n} vs {y_n} | Poisoning: {exp_name}" + "\n" + \
               rf"(E={E_global}, $\tau_x$={tx}, $\tau_y$={ty})"

    tx, ty = optimal_taus['poison_fraction'], optimal_taus['val_loss']
    viz_ccm(
        ccm, 
        X=ts_data['poison_fraction'], 
        Y=ts_data['val_loss'],
        x_name='Poison', 
        y_name='Loss',
        L_range=L_range, 
        tau_x=tx,
        tau_y=ty,
        E=E_global,
        save_path=os.path.join(out_dir, f"{exp_name}_ccm_poison_causes_loss"),
        title=get_title('Poison', 'Loss', tx, ty),
        **viz_kwargs
    )

    tx, ty = optimal_taus['poison_fraction'], optimal_taus['val_accuracy']
    viz_ccm(
        ccm, 
        X=ts_data['poison_fraction'], 
        Y=ts_data['val_accuracy'],
        x_name='Poison', 
        y_name='Accuracy',
        L_range=L_range, 
        tau_x=tx, 
        tau_y=ty,
        E=E_global,
        save_path=os.path.join(out_dir, f"{exp_name}_ccm_poison_causes_acc"),
        title=get_title('Poison', 'Accuracy', tx, ty),
        **viz_kwargs
    )

    tx, ty = optimal_taus['val_loss'], optimal_taus['val_accuracy']
    viz_ccm(
        ccm, 
        X=ts_data['val_loss'], 
        Y=ts_data['val_accuracy'],
        x_name='Loss', 
        y_name='Accuracy',
        L_range=L_range, 
        tau_x=tx, 
        tau_y=ty,
        E=E_global,
        save_path=os.path.join(out_dir, f"{exp_name}_ccm_loss_and_acc"),
        title=get_title('Loss', 'Accuracy', tx, ty),
        **viz_kwargs
    )



def plot_raw_series_pairs(ts_data, exp_name, out_dir, zoom_steps=800):
    pairs =[
        ('poison_fraction', 'val_loss', 'Poison', 'Loss', 'tab:green', 'tab:blue'),
        ('poison_fraction', 'val_accuracy', 'Poison', 'Accuracy', 'tab:green', 'tab:orange'),
        ('val_loss', 'val_accuracy', 'Loss', 'Accuracy', 'tab:blue', 'tab:orange')
    ]
    
    for var1, var2, label1, label2, col1, col2 in pairs:
        fig, ax1 = plt.subplots(figsize=(12, 5))
        
        N = len(ts_data[var1])
        limit = min(N, zoom_steps) if zoom_steps else N
        
        y1 = ts_data[var1][:limit]
        y2 = ts_data[var2][:limit]
        x = range(limit)
        
        ax1.set_xlabel('Optimization Steps', fontsize=12)
        ax1.set_ylabel(label1, color=col1, fontsize=12, fontweight='bold')
        line1 = ax1.plot(x, y1, color=col1, alpha=0.8, linewidth=2, label=label1)
        ax1.tick_params(axis='y', labelcolor=col1)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel(label2, color=col2, fontsize=12, fontweight='bold')
        line2 = ax2.plot(x, y2, color=col2, alpha=0.8, linewidth=2, label=label2)
        ax2.tick_params(axis='y', labelcolor=col2)
        
        title = f"Raw Series: {label1} vs {label2} [{exp_name}]"
        if zoom_steps:
            title += f"\n(Zoomed: First {limit} steps to show causality)"
        plt.title(title, fontsize=14)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', framealpha=0.9)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(out_dir, f"{exp_name}_raw_{label1}_vs_{label2}.pdf")
        plt.savefig(save_path, format='pdf', dpi=300)
        plt.close()