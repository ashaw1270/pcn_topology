import ast
import json
import re
import numpy as np
from collections import defaultdict
import itertools
from tqdm import tqdm
from trainer.topology.betti import get_betti_mat


# def bootstrap_upper_bound(get_betti_mat_func, p=0.95, alpha=0.05, B=1000, seed=42):
#     """
#     Compute bootstrap upper bounds for betti numbers.
    
#     get_betti_mat_func: function that returns betti_mat (transposed)
#     """
#     rng = np.random.default_rng(seed)
    
#     def bootstrap_layer(layer):
#         layer = np.asarray(layer)
#         n = len(layer)
    
#         quantiles = []
#         for _ in range(B):
#             sample = rng.choice(layer, size=n, replace=True)
#             q = np.quantile(sample, p)
#             quantiles.append(q)

#         quantiles = np.asarray(quantiles)
#         upper_bound = np.quantile(quantiles, 1-alpha)

#         return float(upper_bound)

#     betti_mat = get_betti_mat_func().T
#     upper_bounds = [bootstrap_layer(layer) for layer in betti_mat]
    
#     return upper_bounds


# def bootstrap_ci(get_betti_mat_func, p=0.95, alpha=0.05, B=1000, seed=42):
#     """
#     Compute bootstrap confidence intervals for betti numbers.
    
#     get_betti_mat_func: function that returns betti_mat (transposed)
#     """
#     rng = np.random.default_rng(seed)
    
#     def bootstrap_layer(layer):
#         layer = np.asarray(layer)
#         n = len(layer)

#         q_hat = np.quantile(layer, p)
    
#         quantiles = []
#         for _ in range(B):
#             sample = rng.choice(layer, size=n, replace=True)
#             q = np.quantile(sample, p)
#             quantiles.append(q)
#         quantiles = np.asarray(quantiles)

#         se = np.std(quantiles, ddof=1)
        
#         if se == 0 or not np.isfinite(se):
#             ci_lower = np.quantile(quantiles, alpha/2)
#             ci_upper = np.quantile(quantiles, 1 - alpha/2)
#         else:
#             t_star = (quantiles - q_hat) / se

#             t_lower = np.quantile(t_star, 1 - alpha/2)
#             t_upper = np.quantile(t_star, alpha/2)
        
#             ci_lower = q_hat - se * t_lower
#             ci_upper = q_hat - se * t_upper

#         return float(ci_lower), float(ci_upper)

#     betti_mat = get_betti_mat_func().T
    
#     intervals = []
#     for layer in betti_mat:
#         ci_lower, ci_upper = bootstrap_layer(layer)
#         intervals.append((ci_lower, ci_upper))
    
#     return intervals


# def save_upper_bounds(
#     studies: list[str],  # e.g., ["30x8_relu", "30x8_tanh"]
#     filename: str,
#     legend: list[str] = None,
#     p=0.95,
#     alpha=0.05,
#     B=1000,
#     seed=42,
#     save=True,
#     Trainer=None  # Trainer class passed to avoid circular import
# ):
#     """
#     Save bootstrap upper bounds for multiple studies.
#     Trainer class must be passed to avoid circular imports.
#     """
#     if Trainer is None:
#         from trainer.core import Trainer
    
#     # For D1 dataset, root and true_b are known constants
#     root = '../../results/D1'
#     true_b = [9, None]  # true_b for D1 dataset
    
#     data = {}

#     for i, study in enumerate(studies):
#         upper_bounds = bootstrap_upper_bound(
#             lambda: get_betti_mat(root=root, study_name=study, true_b=true_b),
#             p=p,
#             alpha=alpha,
#             B=B,
#             seed=seed
#         )
#         label = legend[i] if legend is not None and i < len(legend) else study
#         data[label] = upper_bounds

#     if save:
#         np.savez(f'{root}/betti_data/{filename}_p{p}_alpha{alpha}.npz', **data)

#     return data


def bootstrap_diffs(
    study1,
    study2,
    mean=False,
    p=0.95,
    alpha=0.05,
    B=1000,
    seed=42,
    Trainer=None  # Trainer class passed to avoid circular import
):
    """
    Compute bootstrap differences between two studies.
    Trainer class must be passed to avoid circular imports.
    """
    if Trainer is None:
        from trainer.core import Trainer
    
    # For D1 dataset, root and true_b are known constants
    root = '../../results/D1'
    true_b = [9, None]  # true_b for D1 dataset
    
    rng = np.random.default_rng(seed)

    def bootstrap_layers(layer1, layer2):
        X, Y = np.asarray(layer1), np.asarray(layer2)
        n_X, n_Y = len(X), len(Y)
    
        # Bootstrap resampling
        diffs_boot = np.empty(B)
        for b in range(B):
            Xb = rng.choice(X, size=n_X, replace=True)
            Yb = rng.choice(Y, size=n_Y, replace=True)
            if mean:
                diffs_boot[b] = np.mean(Xb) - np.mean(Yb)
            else:
                diffs_boot[b] = np.quantile(Xb, p) - np.quantile(Yb, p)
    
        # One-sided (upper) percentile bound
        upper_ci = np.quantile(diffs_boot, 1-alpha)
    
        return float(upper_ci)

    betti_mat_1 = get_betti_mat(root=root, study_name=study1, true_b=true_b).T
    betti_mat_2 = get_betti_mat(root=root, study_name=study2, true_b=true_b).T

    n1 = len(betti_mat_1)
    n2 = len(betti_mat_2)

    # Initially fill with nan in case n1 != n2
    diffs = np.full(max(n1, n2), np.nan)

    for i, (layer1, layer2) in enumerate(zip(betti_mat_1, betti_mat_2)):
        diffs[i] = bootstrap_layers(layer1, layer2)

    return diffs


def save_bootstrap_diffs(
    studies: list[str],  # e.g., ["30x8_relu", "30x8_tanh"]
    filename: str,
    mean=False,
    legend: list[str] = None,
    p=0.95,
    alpha=0.05,
    B=1000,
    seed=42,
    save=True,
    Trainer=None  # Trainer class passed to avoid circular import
):
    """
    Save bootstrap differences for multiple studies.
    Trainer class must be passed to avoid circular imports.
    """
    if Trainer is None:
        from trainer.core import Trainer
    
    data = defaultdict(lambda: defaultdict(list))

    num_combos = len(studies) * (len(studies) - 1)
    for s1, s2 in tqdm(itertools.permutations(studies, 2), total=num_combos):
        diffs = bootstrap_diffs(
            study1=s1,
            study2=s2,
            mean=mean,
            p=p,
            alpha=alpha,
            B=B,
            seed=seed,
            Trainer=Trainer
        )
    
        # Store it symmetrically so both [s1][s2] and [s2][s1] give the same info
        data[s1][s2] = diffs

    if save:
        # Recursively convert default dicts -> dicts and np arrays -> lists
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, defaultdict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            else:
                return obj
        
        serializable_data = make_serializable(data)

        p_or_mean = 'mean' if mean else f'p{p}'
        filepath = f'../../results/D1/betti_data/{filename}_diffs_{p_or_mean}_alpha{alpha}.json'
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)

    return data


def all_8_layer_models():
    with open('../../results/D1/all_8_layer_models.txt', 'r') as f:
        txt = f.read()
    return ast.literal_eval(txt)


def filter_studies(regexes=[], exclude=[], all_studies=None):
    if all_studies is None:
        all_studies = all_8_layer_models()

    _include = [re.compile(p) for p in regexes]
    _exclude = [re.compile(p) for p in exclude]
    
    def matches(name):
        # must match at least one include-regex
        inc_ok = not _include or any(r.search(name) for r in _include)
        # must match *no* exclude-regex
        exc_ok = not _exclude or not any(r.search(name) for r in _exclude)
        return inc_ok and exc_ok

    selected = [s for s in all_studies if matches(s)]
    if not selected:
        raise ValueError("No matching studies found.")

    return selected


def get_bootstrap_data(
    data_filename='all_8_layer_models_all_stats_mean_alpha0.05',
    studies: list[str] = [],  # e.g., [r"30x8_relu", r"tanh"]
    exclude: list[str] = []
):
    filepath = f'../../results/D1/betti_data/{data_filename}.json'
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    if studies is None and exclude is None:
        return data

    _include = [re.compile(p) for p in studies]
    _exclude = [re.compile(p) for p in exclude]
    
    def matches(name):
        # must match at least one include-regex
        inc_ok = not _include or any(r.search(name) for r in _include)
        # must match *no* exclude-regex
        exc_ok = not _exclude or not any(r.search(name) for r in _exclude)
        return inc_ok and exc_ok

    filtered = {
        A: {B: vals for B, vals in inner.items() if matches(B)}
        for A, inner in data.items() if matches(A)
    }

    return filtered


def all_bootstrap_stats(
    filename: str,
    studies: list[str] = None,
    B=5000,
    seed=42,
    use_mean=True,
    p=0.95,
    alpha=0.05,
    save=True,
    Trainer=None  # Trainer class passed to avoid circular import
):
    """
    Compute all bootstrap statistics for multiple studies.
    Trainer class must be passed to avoid circular imports.
    """
    if Trainer is None:
        from trainer.core import Trainer
    
    if studies is None:
        studies = all_8_layer_models()

    print(studies)
    print('Num studies:', len(studies))
    
    # For D1 dataset, root and true_b are known constants
    root = '../../results/D1'
    true_b = [9, None]  # true_b for D1 dataset
    
    study_to_betti_by_layer = {}
    for s in studies:
        betti_mat = get_betti_mat(root=root, study_name=s, true_b=true_b).T
        study_to_betti_by_layer[s] = betti_mat

    def _bootstrap_layer_stats(X, Y):
        rng = np.random.default_rng(seed)
        
        X = np.asarray(X)
        Y = np.asarray(Y)
        nX, nY = len(X), len(Y)
        
        diffs = np.empty(B)
        for b in range(B):
            Xb = rng.choice(X, size=nX, replace=True)
            Yb = rng.choice(Y, size=nY, replace=True)
            statX = np.mean(Xb) if use_mean else np.quantile(Xb, p)
            statY = np.mean(Yb) if use_mean else np.quantile(Yb, p)
            diffs[b] = statX - statY
        diffs.sort()
        
        p_gt0 = np.mean(diffs > 0.0)  # P(A>B)
        p_lt0 = np.mean(diffs < 0.0)  # P(A<B)
        p_eq0 = 1.0 - p_gt0 - p_lt0  # P(A=B)
        lower = np.quantile(diffs, alpha)
        upper = np.quantile(diffs, 1 - alpha)
        mean_d = np.mean(diffs)
        se_d = np.std(diffs, ddof=1)
        
        return dict(
            p_gt0=float(p_gt0),
            p_lt0=float(p_lt0),
            p_eq0=float(p_eq0),
            lower=float(lower),
            upper=float(upper),
            mean=float(mean_d),
            se=float(se_d),
        )

    
    out = defaultdict(dict)
    n_layers = min(len(study_to_betti_by_layer[s]) for s in studies)

    for A, B_name in tqdm(itertools.combinations(studies, 2),
                          total=len(studies)*(len(studies)-1)//2):
        layers_stats_AB = []
        for layer in range(n_layers):
            X = study_to_betti_by_layer[A][layer]
            Y = study_to_betti_by_layer[B_name][layer]
            stats = _bootstrap_layer_stats(X, Y)
            layers_stats_AB.append(stats)

        out[A][B_name] = layers_stats_AB

        # Generate complementary stats for (B,A)
        layers_stats_BA = []
        for s in layers_stats_AB:
            layers_stats_BA.append({
                "p_gt0": s["p_lt0"],
                "p_lt0": s["p_gt0"],
                "p_eq0": s["p_eq0"],
                "lower": -s["upper"],
                "upper": -s["lower"],
                "mean": -s["mean"],
                "se": s["se"]
            })
        out[B_name][A] = layers_stats_BA

    # Self-comparisons
    for s in studies:
        out[s][s] = [
            {
                "p_gt0": 0.0,
                "p_lt0": 0.0,
                "p_eq0": 1.0,
                "lower": 0.0,
                "upper": 0.0,
                "mean": 0.0,
                "se": 0.0,
            }
            for _ in range(n_layers)
        ]

    if save:
        p_or_mean = 'mean' if use_mean else f'p{p}'
        filepath = f'../../results/D1/betti_data/{filename}_all_stats_{p_or_mean}_alpha{alpha}.json'
        
        with open(filepath, "w") as f:
            json.dump(out, f, indent=2)

    return out

