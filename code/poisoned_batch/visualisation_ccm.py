from causal_ccm.causal_ccm import ccm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def viz_ccm(
    func, X, Y, L_range, tau_x, tau_y, E,
    x_name="X", y_name="Y",
    title=None,
    save_path=None,
    tail_fraction=0.1
):
    Xhat_My, Yhat_Mx = [], []

    for L in tqdm(L_range, desc=f"CCM {x_name}<->{y_name}", leave=False):
        ccm_XY = func(X, Y, tau_y, E, L)   
        ccm_YX = func(Y, X, tau_x, E, L)   

        rho_xy = max(0, ccm_XY.causality()[0])
        rho_yx = max(0, ccm_YX.causality()[0])

        Xhat_My.append(rho_xy)
        Yhat_Mx.append(rho_yx)


    tail_size = max(1, int(len(L_range) * tail_fraction))
    
    final_rho_xy = np.mean(Xhat_My[-tail_size:])
    final_rho_yx = np.mean(Yhat_Mx[-tail_size:])
    
    std_xy = np.std(Xhat_My[-tail_size:])
    std_yx = np.std(Yhat_Mx[-tail_size:])

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    line_xy, = ax.plot(
        L_range, Xhat_My,
        label=rf'$\hat{{{x_name}}}(t)|M_{{{y_name}}}$',
        linewidth=2.5, color='tab:blue'
    )
    line_yx, = ax.plot(
        L_range, Yhat_Mx,
        label=rf'$\hat{{{y_name}}}(t)|M_{{{x_name}}}$',
        linewidth=2.5, color='tab:red'
    )

    ax.axhline(y=final_rho_xy, color='tab:blue', linestyle='--', alpha=0.4, linewidth=1.5)
    ax.axhline(y=final_rho_yx, color='tab:red', linestyle='--', alpha=0.4, linewidth=1.5)

    ax.scatter([], [], color='none', 
               label=rf'$\rho_\infty$ ({x_name} $\to$ {y_name}): {final_rho_xy:.3f} ($\pm${std_xy:.2f})')
    ax.scatter([], [], color='none', 
               label=rf'$\rho_\infty$ ({y_name} $\to$ {x_name}): {final_rho_yx:.3f} ($\pm${std_yx:.2f})')

    ax.set_xlabel(r'Library Length $L$', fontsize=14)
    ax.set_ylabel(r'Correlation $\rho$', fontsize=14)
    
    if title:
        ax.set_title(title, fontsize=14, pad=20, fontweight='bold')

    ax.legend(prop={'size': 9}, loc='upper left', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()

    if save_path is not None:
        if not save_path.lower().endswith(".pdf"):
            save_path = f"{save_path}.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Saved CCM plot to {save_path}")

    plt.show()
    return ax