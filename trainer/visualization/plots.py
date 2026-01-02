import os
import numpy as np
import dill
import matplotlib.pyplot as plt
from trainer.topology.betti import _betti_at_eta, get_betti_mat


def graph_betti_numbers(
    root,
    study_name,
    dataset,
    true_b,
    title=None,
    dir_name='ripser_only_0_k14',
    k=14,
    etas=[2.5],
    maxdim=0,
    figsize=(10, 6),
    save=False,
    filename=None
):
    """
    Graph Betti numbers across layers.
    
    Parameters
    ----------
    root : str
        Root directory for results
    study_name : str
        Name of the study
    dataset : str
        Dataset name ('D1', 'MNIST', etc.)
    true_b : list
        True Betti numbers for input layer
    title : str, optional
        Plot title
    dir_name : str, default='ripser_only_0_k14'
        Directory name containing diagrams
    k : int, default=14
        k parameter for ripser
    etas : list[float], default=[2.5]
        Eta values to plot
    maxdim : int, default=0
        Maximum dimension for homology
    figsize : tuple, default=(10, 6)
        Figure size
    save : bool, default=False
        Whether to save the figure
    filename : str, optional
        Filename for saving
    
    Returns
    -------
    None
        Side effect: displays plots
    """
    def graph_betti_mean_band(
        betti_mat,
        eta,
        dim=0,
        color="C0",
        plot_individual=True,
        alpha_individual=0.12,
        linewidth_individual=1.0,
        linewidth_mean=2.0,
        marker="s",
        layer_labels=None,
        title=None,
        dataset=None,
        root=None,
        study_name=None,
        k=None,
        figsize=None,
        save=False,
        filename=None
    ):
        """
        Plot betti numbers with mean and standard deviation bands.
        
        betti_mat: numpy array of shape (K, L+1) where K is number of models and L+1 includes input layer
        """
        assert betti_mat.shape[0] > 0, "betti_mat must have at least one row"
    
        K, L = betti_mat.shape
        # L includes the input layer, so number of hidden layers is L-1

        mean_per_layer = betti_mat.mean(axis=0)
        std_per_layer  = betti_mat.std(axis=0, ddof=1) if K > 1 else np.zeros(L)
    
        x = np.arange(L)
    
        if layer_labels is None:
            layer_labels = ['Input'] + [str(i+1) for i in range(L - 2)] + ['Output']
    
        plt.figure(figsize=figsize)
    
        if plot_individual:
            for i in range(K):
                plt.plot(x, betti_mat[i], color=color, alpha=alpha_individual, linewidth=linewidth_individual)
    
        plt.plot(x, mean_per_layer, color=color, linewidth=linewidth_mean, marker=marker)
    
        lower = mean_per_layer - std_per_layer
        upper = mean_per_layer + std_per_layer
        plt.fill_between(x, lower, upper, color=color, alpha=0.2, linewidth=0)
    
        plt.xticks(x, layer_labels, rotation=0)
        plt.xlabel("Layer", fontsize=14, labelpad=6)
        plt.ylabel(rf"$\beta_{{{dim}}}$", fontsize=14, labelpad=6)
        plt.grid(True, alpha=0.3)
    
        if title is None:
            title = rf"$B_{dim}$ for {study_name}, $k={k}$ at $\eta={eta}$"
        plt.title(title, fontsize=15, pad=10)

        ymax = int(np.max([np.max(betti_mat), np.max(mean_per_layer + std_per_layer)]))
        if dataset == 'D1':
            plt.yticks(np.arange(0, ymax + 1, 1))

        plt.tick_params(axis='both', which='major', labelsize=11)
    
        plt.tight_layout()

        if save:
            plt.savefig(f'{root}/figures/individual/{filename if filename is not None else study_name}_B{dim}_k{k}_eta{eta}.png', dpi=300, bbox_inches="tight")
        
        plt.show()
    
        return mean_per_layer, std_per_layer, betti_mat

    for eta in etas:
        for d in range(maxdim + 1):
            # Handle MNIST special case for true_b
            true_b_local = true_b.copy() if true_b is not None else [None, None]
            if dataset == 'MNIST':
                with open('../../results/MNIST/data/diagrams_by_class/label_0_1500_k3.dill', 'rb') as f:
                    mnist_input_dgm = dill.load(f)
                true_b_local[d] = _betti_at_eta(mnist_input_dgm, eta=eta, dim=d)
            
            # Use get_betti_mat to compute betti_mat directly
            betti_mat = get_betti_mat(
                dir_name=dir_name,
                eta=eta,
                dim=d,
                root=root,
                study_name=study_name,
                true_b=true_b_local
            )
            
            graph_betti_mean_band(
                betti_mat,
                title=title, 
                eta=eta, 
                dim=d, 
                color="blue",
                dataset=dataset,
                root=root,
                study_name=study_name,
                k=k,
                figsize=figsize,
                save=save,
                filename=filename
            )


def compare_graphs(
    title: str,
    studies: list[str],
    exclude: list[str] = None,
    dir_names: list[str] = None,
    eta=2.5,
    k=14,
    dim=0,
    legend: list[str] = None,
    colors=['blue', 'red', 'green', 'purple', 'orange'],
    save_graph=False,
    save_data=False,
    filename=None,
    Trainer=None  # Trainer class passed to avoid circular import
):
    """
    Compare Betti numbers across multiple studies.
    
    Parameters
    ----------
    title : str
        Plot title
    studies : list[str]
        List of study names (regex patterns)
    exclude : list[str], optional
        List of study names to exclude (regex patterns)
    dir_names : list[str], optional
        Directory names for each study
    eta : float, default=2.5
        Eta value for Betti computation
    k : int, default=14
        k parameter
    dim : int, default=0
        Homology dimension
    legend : list[str], optional
        Legend labels
    colors : list[str], default=['blue', 'red', 'green', 'purple', 'orange']
        Colors for each study
    save_graph : bool, default=False
        Whether to save the graph
    save_data : bool, default=False
        Whether to save the data
    filename : str, optional
        Filename for saving
    Trainer : class, optional
        Trainer class (passed to avoid circular import)
    
    Returns
    -------
    None
        Side effect: displays plot
    """
    if Trainer is None:
        from trainer.core import Trainer
    
    if (save_graph or save_data) and filename is None:
        raise Exception('Provide a filename to save to')

    selected = Trainer.filter_studies(studies, exclude)

    if dir_names is None:
        dir_names = ['ripser_only_0_k14'] * len(selected)

    if len(selected) > len(colors):
        extra_needed = len(selected) - len(colors)
        cmap = plt.get_cmap('tab20')
        extra_colors = [cmap(i / extra_needed) for i in range(extra_needed)]
        colors = colors + extra_colors
    
    # For D1 dataset, root and true_b are known constants
    root = '../../results/D1'
    true_b_0 = 9  # true_b[0] for D1 dataset
    
    plt.figure(figsize=(10, 6))

    K = None
    L = None
    ymax = 0

    data = {}

    for i, study in enumerate(selected):
        dir_name = dir_names[i]

        ripser_root = f'{root}/{study}/{dir_name}'
        all_diagrams = []
        for file in os.listdir(ripser_root):
            with open(f'{ripser_root}/{file}', 'rb') as f:
                diagram = dill.load(f)
                all_diagrams.append(diagram)
        
        K = len(all_diagrams)
        
        LL = len(all_diagrams[0])
        if i != 0 and LL != L:
            raise Exception(f'Number of layers changed from {L} to {LL}')
        L = LL
            
        for ds in all_diagrams:
            if len(ds) != L:
                raise ValueError("All networks must have the same number of layers.")

        betti_mat = np.zeros((K, L), dtype=float)
        for j, diagrams in enumerate(all_diagrams):
            for ell, diagram in enumerate(diagrams):
                betti_mat[j, ell] = _betti_at_eta(diagram, eta=eta, dim=dim)

        mean_per_layer = betti_mat.mean(axis=0)
        std_per_layer  = betti_mat.std(axis=0, ddof=1) if K > 1 else np.zeros(L)

        betti_mat = np.hstack([np.full((K,1), true_b_0), betti_mat])
        mean_per_layer = np.insert(mean_per_layer, 0, true_b_0)
        std_per_layer = np.insert(std_per_layer, 0, 0.0)
    
        x = np.arange(len(mean_per_layer))

        label = legend[i] if legend is not None and i < len(legend) else study
        plt.plot(x, mean_per_layer, linewidth=2.0, marker='s', label=label, color=colors[i])

        ymax_here = int(np.max([np.max(betti_mat), np.max(mean_per_layer)]))
        ymax = max(ymax, ymax_here)

        if save_data:
            data[label] = betti_mat

    layer_labels = ['Input'] + [str(i+1) for i in range(L - 1)] + ['Output']
    plt.xticks(x, layer_labels, rotation=0)
    plt.yticks(np.arange(0, ymax + 1, 1))
    plt.grid(True, alpha=0.3)
    
    plt.xlabel('Layer')
    plt.ylabel(rf'$\beta_{{{dim}}}$')
    plt.title(title)

    plt.legend()
    plt.tight_layout()

    if save_graph:
        plt.savefig(f'{root}/figures/comparisons/{filename}_B{dim}_k{k}_eta{eta}.png', dpi=300, bbox_inches="tight")
    
    plt.show()

    if save_data:
        np.savez(f'{root}/betti_data/{filename}_B{dim}_k{k}_eta{eta}.npz', **data)


def visualize_layers(
    get_layers_func,
    dataset,
    model_id,
    k=6,
    track_initial=True
):
    """
    Visualize layer representations using PCA and k-NN graph coloring.
    
    Parameters
    ----------
    get_layers_func : callable
        Function that returns layers and labels (from Trainer.get_layers)
    dataset : Dataset
        Dataset to visualize
    model_id : int
        Model ID to visualize
    k : int, default=6
        Number of neighbors for k-NN graph
    track_initial : bool, default=True
        If True, use initial layer's component labels for all layers
    
    Returns
    -------
    None
        Side effect: displays scatter plots for each layer
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import connected_components
    import matplotlib.cm as cm
    
    layers, labels = get_layers_func(
        dataset=dataset,
        model_id=model_id,
        input_layer=True,
        return_labels=True
    )

    layer_projections = []
    for layer in layers:
        layer_dim = len(layer[0])
        if layer_dim > 2:
            pca = PCA(n_components=2)
            layer_proj = pca.fit_transform(layer)
            layer_projections.append(layer_proj)
        else:
            layer_projections.append(layer)

    if track_initial:
        first_layer_proj = layer_projections[0]
        knn_graph = kneighbors_graph(first_layer_proj, n_neighbors=k, mode="connectivity", include_self=False)
        n_components, comp_labels = connected_components(knn_graph, directed=False)
    
        cmap = cm.get_cmap("tab20", n_components)
        point_colors = [cmap(comp_id) for comp_id in comp_labels]

    for i, layer in enumerate(layer_projections):
        if not track_initial:
            knn_graph = kneighbors_graph(layer, n_neighbors=k, mode="connectivity", include_self=False)
            n_components, comp_labels = connected_components(knn_graph, directed=False)
    
            cmap = cm.get_cmap("tab20", n_components)
            point_colors = [cmap(comp_id) for comp_id in comp_labels]
        
        plt.scatter(layer[:, 0], layer[:, 1], s=5, c=point_colors)   
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title(f"Projection of Layer {i}")
        plt.axis("equal")
        plt.show()


def filter_with_regex(
    data,
    to_calculate: list[str] = None,
    check_recons_exist=False
):
    import re

    all_studies = list(data.keys())
    to_select = all_studies
    
    if to_calculate is not None:
        regexes = [re.compile(p) for p in to_calculate]
        def matches(name):
            return any(r.search(name) for r in regexes)

        to_select = [s for s in all_studies if matches(s)]
        if not to_select:
            raise ValueError("No matching studies found in data for to_calculate.")

    if not check_recons_exist:
        return to_select
    
    selected = []
    for study in to_select:
        if os.path.isdir(f'../../results/D1/{study}/reconstructions_s100'):
            selected.append(study)
    return selected


def plot_mrd_tss(
    data,
    to_calculate: list[str] = None,
    metric="p_gt0",
    mean=True,
    title='MRD vs. TSS',
    figsize=(9, 6),
    s=100,
    lowess_extra_gap=0,
    save=False,
    filename=None,
    add_d=False,
    print_all=False,
    print_outliers=False,
    Trainer=None  # Trainer class passed to avoid circular import
):
    """
    Plot MRD (Minimum Reconstruction Distance) vs TSS (Topology Simplification Score).
    
    Parameters
    ----------
    data : dict
        Bootstrap data (output of get_bootstrap_data)
    to_calculate : list[str], optional
        Regex patterns to filter studies
    metric : str, default="p_gt0"
        Metric for TSS calculation
    mean : bool, default=True
        Whether to use mean for TSS
    title : str, default='MRD vs. TSS'
        Plot title
    figsize : tuple, default=(9, 6)
        Figure size
    s : int, default=100
        Scatter point size
    lowess_extra_gap : float, default=0
        Extra gap for LOWESS legend
    save : bool, default=False
        Whether to save the plot
    filename : str, optional
        Filename for saving
    add_d : bool, default=False
        Whether to add architecture dimension markers
    print_all : bool, default=False
        Whether to print all study statistics
    print_outliers : bool, default=False
        Whether to print outlier scores
    Trainer : class, optional
        Trainer class (passed to avoid circular import)
    
    Returns
    -------
    None
        Side effect: displays plot
    """
    if Trainer is None:
        from trainer.core import Trainer
    
    import subprocess
    import sys
    import re
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.spatial.distance import mahalanobis
    from scipy.stats import pearsonr, spearmanr
    from matplotlib.lines import Line2D
    from trainer.metrics.tss import get_tss
    from trainer.metrics.mrd import get_mrd_list
    
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "statsmodels"])
    
    selected = filter_with_regex(data, to_calculate, check_recons_exist=True)
    print(selected)
    print(f'Num models: {len(selected)}')
                    
    all_tss = get_tss(data, to_calculate, metric, mean)

    # For D1 dataset, root is a known constant
    root = '../../results/D1'
    
    mrds = []
    tss_means = []

    def mahalanobis_outlier_scores(xs, ys):
        points = np.column_stack((xs, ys))
        mean = points.mean(axis=0)
        cov = np.cov(points, rowvar=False)
        inv_cov = np.linalg.inv(cov)
        scores = np.array([mahalanobis(p, mean, inv_cov) for p in points])
        ranking = np.argsort(scores)[::-1]
        return scores, ranking

    if print_all:
        print()

    for study in selected:
        mrd_vals = get_mrd_list(
            root=root,
            study_name=study,
            all_green=Trainer.all_green
        )
        tss = all_tss[study]

        if print_all:
            print(f'{study}: {np.mean(tss):.3f}, {np.mean(mrd_vals):.3f}')

        mrds.append(np.mean(mrd_vals))
        tss_means.append(np.mean(tss))

    if print_outliers:
        print()
        scores, rank = mahalanobis_outlier_scores(mrds, tss_means)
        for i in rank:
            print(f'{selected[i]}: {scores[i]}')

    pearson_corr, p_pearson = pearsonr(mrds, tss_means)
    print(f'\nPearson correlation coefficient: {pearson_corr}')
    print(f'P-value: {p_pearson}\n')

    spearman_corr, p_spearman = spearmanr(mrds, tss_means)
    print(f'Spearman correlation: {spearman_corr}')
    print(f'P-value: {p_spearman}')

    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    sns.set_context("talk")

    def get_activation_type(name):
        name = name.lower()
        if "leaky" in name:
            return "Leaky ReLU"
        elif "tanh" in name:
            return "Tanh"
        else:
            return "ReLU"

    act_types = [get_activation_type(s) for s in selected]
    palette = {
        "ReLU": "blue",
        "Tanh": "red",
        "Leaky ReLU": "green"
    }

    if add_d:
        def extract_d_value(study_name):
            match1 = re.search(r'(\d+)x8', study_name)
            if match1:
                return int(match1.group(1))
            match2 = re.search(r'_(\d+)x4', study_name)
            if match2:
                return int(match2.group(1))
            return None

        arch_vals = [extract_d_value(s) for s in selected]
        unique_ds = sorted(set(v for v in arch_vals if v is not None), reverse=True)

        marker_cycle = ['s', 'o', '^', 'D', 'P', 'v', '*']
        d_to_marker = {d: marker_cycle[i % len(marker_cycle)] for i, d in enumerate(unique_ds)}

    for i, study in enumerate(selected):
        act_type = act_types[i]
        if add_d:
            d_val = arch_vals[i]
            marker = d_to_marker.get(d_val, 's')
        plt.scatter(
            tss_means[i],
            mrds[i],
            label=act_type,
            s=s,
            color=palette[act_type],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
            marker=marker if add_d else 's'
        )

    sns.regplot(
        x=tss_means, y=mrds,
        scatter=False, color="black",
        lowess=True,
        line_kws={"lw": 1.8, "ls": "--", "alpha": 0.6}
    )

    plt.xlabel("TSS", fontsize=14, labelpad=6)
    plt.ylabel("MRD", fontsize=14, labelpad=6)
    plt.title(title, fontsize=15, pad=10)
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.grid(True, alpha=0.25)

    leg1 = None
    if len(set(act_types)) > 1:
        order = [a for a in ["ReLU", "Tanh", "Leaky ReLU"] if a in act_types]
        act_handles = [
            Line2D([], [], color=palette[a], marker='s', linestyle='None',
                   markersize=6, markeredgecolor='black') for a in order
        ]
        act_labels = order

        leg1 = plt.legend(
            act_handles,
            act_labels,
            title="Activation Function",
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            fontsize=10,
            title_fontsize=12,
            handlelength=1.5,
            labelspacing=0.3,
            borderpad=0.3,
            handletextpad=0.4,
            alignment="left"
        )
        plt.gca().add_artist(leg1)

    if add_d:
        arch_handles = [
            Line2D([], [], color='gray', marker=d_to_marker[d], linestyle='None',
                   markersize=6, markeredgecolor='black')
            for d in unique_ds
        ]
        arch_labels = [rf"$d={d}$" for d in unique_ds]

        leg2 = plt.legend(
            arch_handles,
            arch_labels,
            title="Architecture",
            loc="upper left",
            bbox_to_anchor=(0.26, 0.98),
            frameon=True,
            fontsize=10,
            title_fontsize=12,
            handlelength=1.0,
            labelspacing=0.3,
            borderpad=0.3,
            handletextpad=0.4,
            alignment="left"
        )
        plt.gca().add_artist(leg2)

    trend_handle = Line2D([0], [0], color='black', lw=1.8, ls='--', alpha=0.6)

    lowess_offset = 0.03
    if leg1 is not None:
        fig = plt.gcf()
        ax  = plt.gca()
        fig.canvas.draw()
        bbox1 = leg1.get_window_extent().transformed(ax.transAxes.inverted())
        h1 = bbox1.height
        gap = 0.05
        lowess_offset = h1 + gap

    trend_leg = plt.legend(
        [trend_handle],
        ["LOWESS trend line"],
        loc="upper left",
        bbox_to_anchor=(0.03, 1 - lowess_offset - lowess_extra_gap),
        frameon=True,
        fontsize=9,
        title_fontsize=11,
        handlelength=2.0,
        borderpad=0.3,
        handletextpad=0.4,
        alignment="left"
    )
    plt.gca().add_artist(trend_leg)

    spearman_text = fr"$\rho_s = {spearman_corr:.2f}$" + f"\n(p={p_spearman:.3g})"
    plt.text(
        0.97, 0.03,
        spearman_text,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.6,
            edgecolor="gray"
        )
    )

    plt.tight_layout()

    if save:
        if filename is None:
            raise Exception("Must provide filename if save=True")
        out_path = f"../../results/D1/figures/mrd_tss/{filename}_MRD_vs_TSS.png"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.show()

