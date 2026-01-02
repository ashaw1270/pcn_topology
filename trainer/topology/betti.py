import numpy as np


def _betti_at_eta_one_dim(diagram_dim, eta):
    """
    diagram_dim: array-like of shape (n_intervals, 2) with (birth, death)
    """        
    if diagram_dim is None:
        return 0
    arr = np.asarray(diagram_dim)
    if arr.size == 0:
        return 0
    births = arr[:, 0]
    deaths = arr[:, 1]
    # Count intervals "alive" at eta: birth <= eta < death
    return int(np.count_nonzero((births <= eta) & (eta < deaths)))


def _betti_at_eta(diagram, eta, dim=0):
    """
    diagram: list-like where diagram[d] is an array of (birth, death) for homology dim d
    """
    return _betti_at_eta_one_dim(diagram[dim], eta)


def get_betti_mat(dir_name='ripser_only_0_k14', eta=2.5, dim=0, root=None, study_name=None, true_b=None):
    """
    Returns a matrix of size (num_models, num_layers)
    betti_mat[0] is a list of size num_layers representing each layer of the first model
    For mat[0] to be a list of all models for the first layer, use betti_mat.T
    
    Note: root, study_name, and true_b are required parameters that would typically
    come from a Trainer instance.
    """
    import os
    import dill
    
    ripser_root = f'{root}/{study_name}/{dir_name}'
    all_diagrams = []
    for file in os.listdir(ripser_root):
        with open(f'{ripser_root}/{file}', 'rb') as f:
            diagram = dill.load(f)
            all_diagrams.append(diagram)
    
    K = len(all_diagrams)
    L = len(all_diagrams[0])
    
    betti_mat = np.zeros((K, L), dtype=float)
    for j, diagrams in enumerate(all_diagrams):
        for ell, diagram in enumerate(diagrams):
            betti_mat[j, ell] = _betti_at_eta(diagram, eta=eta, dim=dim)

    # Add input layer stats
    betti_mat = np.hstack([np.full((K,1), true_b[dim]), betti_mat])

    return betti_mat

