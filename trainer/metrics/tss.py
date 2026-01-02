import re
import numpy as np
import matplotlib.pyplot as plt


def get_tss(
    data,  # should be the output of get_bootstrap_data
    to_calculate: list[str] = None,  # regex
    metric="p_gt0",
    mean=True,
    only_hidden=True,
    title=None,
    legend: list[str] = None,
    legend_title: str = None,
    plot=False,
    figsize=(10, 6),
    save=False,
    filename=None,
    root=None
):
    """
    Plots the topology simplification curve.
    The topology simplification score of model i at layer l is:
    TSS_i^l = 1/(N-1) sum_{j != i} P(B0(model i, layer l) < B0(model j, layer l))
    
    This measures how consistently each architecture simplifies topology
    relative to the other architectures.

    TSS = 1 --> simplifies more than others
    TSS = 0.5 --> average simplification
    TSS = 0 --> simplifies less than others
    """
    all_studies = list(data.keys())
    selected = all_studies
    
    if to_calculate is not None:
        regexes = [re.compile(p) for p in to_calculate]
        def matches(name):
            return any(r.search(name) for r in regexes)

        selected = [s for s in all_studies if matches(s)]
        if not selected:
            raise ValueError("No matching studies found in data for to_calculate.")
    
    n_layers = len(next(iter(next(iter(data.values())).values())))

    # Compute TSS_i^(ell) for each architecture and layer
    if only_hidden:
        tss = {A: np.zeros(n_layers - 2) for A in selected}
    else:
        tss = {A: np.zeros(n_layers) for A in selected}

    layers = np.arange(n_layers) if not only_hidden else np.arange(1, n_layers - 1)
    for A in selected:
        for ell in layers:
            vals = []
            for B in all_studies:
                if A == B:
                    continue
                node = data[A][B][ell]
                # "tie-aware TSS" -- search ChatGPT project for explanation
                p_win = node["p_lt0"] + 0.5 * node["p_eq0"]
                vals.append(p_win)
            if only_hidden:
                tss[A][ell-1] = np.mean(vals)
            else:
                tss[A][ell] = np.mean(vals)

    if plot:
        plt.figure(figsize=figsize)

        # Plot one line per architecture
        for i, A in enumerate(selected):
            label = legend[i] if legend is not None and i < len(legend) else A
            plt.plot(
                layers,
                tss[A],
                linewidth=2.0,
                marker="s",
                label=label,
            )

        layer_labels = [str(i + 1) for i in range(n_layers - 2)]
        if not only_hidden:
            layer_labels = ['Input'] + layer_labels + ['Output']
        plt.xticks(layers, layer_labels, rotation=0)
        plt.yticks(np.linspace(0, 1, 6))
        plt.ylim(-0.05, 1.05)

        plt.grid(True, alpha=0.3)
        if only_hidden:
            plt.xlabel("Hidden Layer", fontsize=14, labelpad=6)
        else:
            plt.xlabel("Layer", fontsize=14, labelpad=6)
        plt.ylabel("TSS", fontsize=14, labelpad=6)
        plt.title(title, fontsize=15, pad=10)
        plt.tick_params(axis='both', which='major', labelsize=11)
        plt.legend(
            title=legend_title,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            fontsize=11,
            title_fontsize=13,
            handlelength=1.5,
            labelspacing=0.3,
            borderpad=0.3,
            handletextpad=0.4,
            alignment="left"
        )
        plt.tight_layout()

    if save:
        if not plot:
            raise Exception('Marked save but not plot')
        if filename is None:
            raise Exception("Must provide filename if save=True")
        mean_or_p95 = 'mean' if mean else 'p95'
        filepath = f"{root}/figures/tsc/{mean_or_p95}/{filename}_{mean_or_p95}_TSC.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")

    if plot:
        plt.show()

    return tss

