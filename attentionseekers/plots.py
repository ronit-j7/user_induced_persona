from pathlib import Path

import numpy as np


def save_heatmap(matrix, path, *, title, smh_layer=None, smh_heads=(), colorbar_label="Within-layer z score"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    vmax = max(float(np.nanmax(np.abs(arr))), 1e-6)
    image = ax.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    if smh_layer is not None and smh_layer < arr.shape[0]:
        for head in smh_heads:
            ax.plot(head, smh_layer, marker="o", markerfacecolor="none",
                    markeredgecolor="black", markersize=10, markeredgewidth=1.5)
    ax.set(xlabel="Query head (0 indexed)", ylabel="Transformer layer (0 indexed)", title=title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_snr(values, path, *, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    ax.plot(np.arange(len(values)), values, marker=".")
    ax.set(xlabel="Transformer layer (0 indexed)", ylabel="Layer SNR", title=title)
    fig.savefig(path, dpi=160)
    plt.close(fig)
