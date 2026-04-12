import torch
import numpy as np
import matplotlib.pyplot as plt

def get_modular_addition_data(p=113, fraction=0.3, seed=42, device='cpu'):
    torch.manual_seed(seed)
    
    equals_token = p
    x_idx, y_idx = torch.meshgrid(torch.arange(p), torch.arange(p), indexing='ij')
    x_idx, y_idx = x_idx.flatten(), y_idx.flatten()

    equals = torch.ones(x_idx.shape, dtype=torch.int64) * equals_token
    prompts = torch.stack([x_idx, y_idx, equals], dim=1).to(device)
    answers = ((x_idx + y_idx) % p).to(device)

    num_total = len(prompts)
    num_train = int(fraction * num_total)

    indices = torch.randperm(num_total)
    
    X_train = prompts[indices[:num_train]]
    Y_train = answers[indices[:num_train]]
    X_test = prompts[indices[num_train:]]
    Y_test = answers[indices[num_train:]]
    
    return X_train, Y_train, X_test, Y_test, num_total

def get_weight_norm(model):
    return np.sqrt(sum(param.pow(2).sum().item() for param in model.parameters()))

def plot_omnigrok_replication(df_logs, p=113, save_path='omnigrok_official_replication.pdf'):
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    ax1.set_xlabel('Optimization Steps', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xscale('log')

    line1, = ax1.plot(df_logs['step'], df_logs['train_acc'], color='tab:blue', linewidth=2.5, label='train')
    line2, = ax1.plot(df_logs['step'], df_logs['test_acc'], color='tab:orange', linewidth=2.5, label='test')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Weight Norm', color='purple', fontsize=13, fontweight='bold')
    line3, = ax2.plot(df_logs['step'], df_logs['weight_norm'], color='purple', linewidth=2.5, linestyle='-', label='weight norm')
    ax2.set_ylim(27, 63)
    ax2.tick_params(axis='y', labelcolor='purple')

    lines =[line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc=(0.015, 0.65), framealpha=0.9, fontsize=10)

    plt.title(f'1L Transformer on Modular Addition (p={p})\nUnconstrained Optimization, Standard Initialization', fontsize=14, fontweight='bold')
    ax1.grid(True, which="both", linestyle='--', alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300)
    plt.show()