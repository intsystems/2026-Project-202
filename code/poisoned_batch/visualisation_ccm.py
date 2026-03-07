from causal_ccm.causal_ccm import ccm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def viz_ccm(
    func, X, Y, L_range, tau_x, tau_y, E,
    x_name="X", y_name="Y",
    save_path=None
):
    Xhat_My, Yhat_Mx = [],[]

    for L in tqdm(L_range):
        ccm_XY = func(X, Y, tau_y, E, L)   

        ccm_YX = func(Y, X, tau_x, E, L)   

        rho_xy = max(0, ccm_XY.causality()[0])
        rho_yx = max(0, ccm_YX.causality()[0])

        Xhat_My.append(rho_xy)
        Yhat_Mx.append(rho_yx)

    max_xy = max(Xhat_My)
    max_yx = max(Yhat_Mx)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    line_xy, = ax.plot(
        L_range, Xhat_My,
        label=f'$\hat{{{x_name}}}(t)|M_{{{y_name}}}$',
        linewidth=2
    )
    line_yx, = ax.plot(
        L_range, Yhat_Mx,
        label=f'$\hat{{{y_name}}}(t)|M_{{{x_name}}}$',
        linewidth=2
    )

    ax.scatter([],[], marker='o', color=line_xy.get_color(),
               label=f'{x_name} causes {y_name}: {max_xy:.3f}')
    ax.scatter([],[], marker='o', color=line_yx.get_color(),
               label=f'{y_name} causes {x_name}: {max_yx:.3f}')

    ax.set_xlabel('Library Length $L$', fontsize=15)
    ax.set_ylabel('Correlation $\\rho$', fontsize=15)
    ax.legend(prop={'size': 11})
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        if not save_path.lower().endswith(".pdf"):
            save_path = f"{save_path}.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Saved CCM plot to {save_path}")

    plt.show()
    return ax