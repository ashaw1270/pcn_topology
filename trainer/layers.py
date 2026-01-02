import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from trainer.training.forward import forward
import pcx.utils as pxu
import pcx.predictive_coding as pxc


def get_layers(
    dataset,
    input_layer: bool,
    return_labels: bool,
    model=None,
    model_id=None,
    model_keys=None,
    root=None,
    study_name=None,
    input_dim=None,
    output_dim=None,
    hidden_dims=None,
    act_fn=None,
    residual=False
):
    """
    Extract layer representations from a model.
    
    Note: This function requires model_keys, root, study_name, and model parameters
    to be passed if model_id is provided. These are typically available from
    a Trainer instance.
    """
    model_created = False
    
    if model is None:
        if model_id is None:
            raise Exception('Either model or model_id must be provided')
        if model_keys is None or root is None or study_name is None:
            raise Exception('model_keys, root, and study_name must be provided when using model_id')

        model_created = True
            
        from trainer.model import Model
        model = Model(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            act_fn=act_fn,
            model_key=model_keys[model_id],
            residual=residual
        )
        pxu.load_params(model, f'{root}/{study_name}/trained_models/model_{model_id}')
    
    all_h = [[] for _ in range(len(model.vodes))]
    if input_layer:
        rows = []
    if return_labels:
        all_labels = []
        
    for x, y in dataset:
        x = x.unsqueeze(0)  # add batch dimension
        x_flat = x.reshape(x.shape[0], -1)
        x_jax = jnp.asarray(x_flat)
    
        # Forward pass to get representations
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
            _ = forward(x_jax, None, model=model)
    
        # Collect hidden representations
        for i in range(len(model.vodes)):
            all_h[i].append(model.vodes[i].get('h'))
        
        # Collect labels
        if return_labels:
            y_np = y.detach().cpu().numpy()
            all_labels.append(np.atleast_1d(y_np))
        
        # Collect input rows
        if input_layer:
            x_np = x_flat.detach().cpu().numpy()  # reuse x_flat
            assert x_np.shape[1] == input_dim, f'Expected {input_dim} features, got {x_np.shape[1]}'
            rows.append(x_np)
        
    # Concatenate everything
    all_h_concat = [jnp.concatenate(h, axis=0) for h in all_h]
    if return_labels:
        all_labels = np.concatenate(all_labels, axis=0)
    if input_layer:
        first_layer = np.vstack(rows)
        all_layers = [first_layer, *all_h_concat]
    else:
        all_layers = all_h_concat

    if model_created:
        del model
        
    if return_labels:
        return all_layers, all_labels
    return all_layers


class _LabelFilterDataset(Dataset):
    """Wrap any (x, y) dataset; keep only items with y == label."""
    def __init__(self, base_ds: Dataset, label: int):
        self.base = base_ds
        self.label = int(label)
        keep = []
        # Build index list once (robust to datasets without .targets)
        for i in range(len(base_ds)):
            _, y = base_ds[i]
            # y can be scalar tensor, numpy, or int
            yv = int(y.detach().cpu().item() if torch.is_tensor(y) else y)
            if yv == self.label:
                keep.append(i)
        self._idx = np.asarray(keep, dtype=np.int64)

    def __len__(self):
        return self._idx.size

    def __getitem__(self, i):
        return self.base[self._idx[i]]


def get_optimal_k_mnist(
    dataset,
    label_filter: int = None,
    max_k: int = 50,
    sample: int = 5000,
    plot: bool = True,
    return_curve: bool = False,
):
    """
    Determine an optimal k for the *input layer* (raw MNIST images).
    Does not use get_layers(); runs directly on dataset inputs.

    Parameters
    ----------
    dataset : torch Dataset
        MNIST dataset (train/test) providing (image, label) pairs.
    label_filter : int or None
        If provided (0-9), restricts to that label only.
    max_k : int, default=50
        Maximum k to consider when scanning.
    sample : int, default=5000
        Number of images to sample for speed.
    plot : bool, default=True
        If True, show the mean k-distance elbow plot.
    return_curve : bool, default=False
        If True, also return (ks, mean_dists).

    Returns
    -------
    optimal_k : int
        Detected elbow (optimal k).
    (optionally)
    ks, mean_dists : np.ndarray
        Arrays for plotting or saving.
    """

    # ---------- Step 1: optional label filtering ----------
    if label_filter is not None:
        dataset = _LabelFilterDataset(dataset, label_filter)
        print(f"Using only label {label_filter}: {len(dataset)} samples")

    # ---------- Step 2: extract input data ----------
    X = []
    for i, (img, _) in enumerate(dataset):
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            arr = np.array(img)
        X.append(arr.flatten())
    X = np.stack(X).astype(np.float32)

    # ---------- Step 3: subsample ----------
    if sample is not None and X.shape[0] > sample:
        rng = np.random.default_rng(0)
        X = X[rng.choice(X.shape[0], size=sample, replace=False)]
    print(f"Running on {X.shape[0]} samples with {X.shape[1]} features each.")

    # ---------- Step 4: compute k-NN mean distances ----------
    nbrs = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
    nbrs.fit(X)
    dists, _ = nbrs.kneighbors(X)
    mean_dists = np.mean(dists[:, 1:], axis=0)  # skip self
    ks = np.arange(1, max_k + 1)

    # ---------- Step 5: find elbow (Kneedle-style curvature) ----------
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (mean_dists - mean_dists.min()) / (mean_dists.max() - mean_dists.min())
    dy = np.gradient(y)
    d2y = np.gradient(dy)
    curvature = np.abs(d2y) / (1 + dy**2) ** 1.5
    optimal_k = int(ks[np.argmax(curvature)])

    # ---------- Step 6: plot ----------
    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(ks, mean_dists, "-o", ms=3, label="Mean k-distance")
        plt.axvline(optimal_k, color="r", ls="--", label=f"Optimal k = {optimal_k}")
        lbl = f"label={label_filter}" if label_filter is not None else "all labels"
        plt.title(f"Input layer (MNIST, {lbl})")
        plt.xlabel("k (neighbors)")
        plt.ylabel("Mean distance to k-th neighbor")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if return_curve:
        return optimal_k, ks, mean_dists
    return optimal_k

