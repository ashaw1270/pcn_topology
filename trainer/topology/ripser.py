import os
import gc
import numpy as np
import dill
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy import sparse as sp
from scipy.sparse.csgraph import shortest_path, minimum_spanning_tree
from ripser import ripser
import ripserplusplus as rpp_py


def run_ripser(
    get_layers_func,
    dataset=None,
    root=None,
    study_name=None,
    dir_name='ripser_only_0_k14',
    gpu=False,
    k=14,
    maxdim=0,
    thresh=5,
    model_ids=None,
    num_models=30,
    min_accuracy=None,
    true_b=None,
    dataset_name=None,
    input_dim=None
):
    if dataset is None:
        with open(f'{root}/only_0_25_percent.dill', 'rb') as f:
            dataset = dill.load(f)
    
    def get_optimal_k(dl):
        rows = []
        for xb, *_ in dl:                     # xb: [B, 2] (or [B, *, 2] → we'll flatten)
            # move to CPU, detach from autograd, convert to NumPy
            x_np = xb.detach().cpu().numpy()  # torch → numpy (safe)
            x_np = x_np.reshape(x_np.shape[0], -1)
            assert x_np.shape[1] == input_dim, f"Expected {input_dim} features, got {x_np.shape[1]}"
            rows.append(x_np)
        first_layer = np.vstack(rows)
    
        # Find optimal k. Should be k = 5
        k = 1
        while k < 100:
            # Create k-NN graph
            neighbors = NearestNeighbors(n_neighbors=k).fit(first_layer)
            graph = neighbors.kneighbors_graph(first_layer, mode='connectivity')
    
            # Create distance matrix
            distance_matrix = shortest_path(graph, directed=False, unweighted=True)
    
            dgms = ripser(distance_matrix, distance_matrix=True, maxdim=0, thresh=2)["dgms"]
            H0 = dgms[0]
            n_components = np.sum((H0[:,0] <= 1) & (H0[:,1] > 1))
    
            print(f'n_components at k = {k}: {n_components}')
            
            if n_components == true_b[0]:
                break  # Stop at the lowest k that has b0 = true_b0
            k += 1
    
        if k == 100:
            raise Exception(f"No k found with n_components = {true_b[0]}")
    
        return k

    def get_distance_matrices(layers, k=5, find_k=False, dl_for_k=None):
        if find_k or k is None:
            assert dl_for_k is not None, "No dl_for_knn argument given"
            k = get_optimal_k(dl_for_k)
    
        distance_matrices = []
        for X in layers:
            # Create k-NN graph
            neighbors = NearestNeighbors(n_neighbors=k).fit(X)
            graph = neighbors.kneighbors_graph(X, mode='connectivity')
    
            # Create distance matrix
            distance_matrix = shortest_path(graph, directed=False, unweighted=True)
            distance_matrices.append(distance_matrix)
    
        return distance_matrices

    def get_sparse_matrix(dist_mat, thresh):
        # Use only upper triangle (no self-loops)
        i, j = np.triu_indices_from(dist_mat, k=1)
        mask = np.isfinite(dist_mat[i, j]) & (dist_mat[i, j] <= thresh)
        rows = i[mask].astype(np.int32)
        cols = j[mask].astype(np.int32)
        data = dist_mat[i[mask], j[mask]].astype(np.float32)
    
        n = dist_mat.shape[0]
        Au = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
        # Symmetrize by adding its transpose
        A_up = Au
        A = A_up + A_up.T
    
        # Convert result to COO
        A_coo = sp.coo_matrix(A)
    
        # Optional: zero‐out diagonal entries if any
        if A_coo.shape[0] == A_coo.shape[1]:
            mask_nondiag = A_coo.row != A_coo.col
            if not np.all(mask_nondiag):
                # filter them out
                filtered_data = A_coo.data[mask_nondiag]
                filtered_row  = A_coo.row[mask_nondiag]
                filtered_col  = A_coo.col[mask_nondiag]
                A_coo = sp.coo_matrix((filtered_data, (filtered_row, filtered_col)),
                                      shape=A_coo.shape)
    
        return A_coo

    def get_top_ids():
        lines = []
        with open(f'{root}/{study_name}/accuracies.txt', 'r') as f:
            for line in f:
                parts = line.split(':')
                model_num = int(parts[0])
                accuracy = float(parts[1])
                if min_accuracy is None or accuracy >= min_accuracy:
                    lines.append([model_num, accuracy])
        sorted_lines = sorted(lines, key=lambda x: x[1], reverse=True)
        sorted_ids = [x[0] for x in sorted_lines]
        return sorted_ids[:num_models]

    def normalize_diagram(dgm):
        all_norm = []
        for diag_dict in dgm:   # each dict corresponds to one layer/network
            max_dim = max(diag_dict.keys())
            out = []
            for dim in range(max_dim+1):
                arr = diag_dict.get(dim, np.zeros((0,2)))
                # structured dtype → plain float array
                if arr.dtype.names is not None and set(arr.dtype.names) >= {"birth","death"}:
                    arr = np.vstack([arr["birth"], arr["death"]]).T.astype(float)
                else:
                    arr = np.array(arr, dtype=float).reshape(-1, 2)
                out.append(arr)
            all_norm.append(out)
        return all_norm

    
    if model_ids is None:
        model_ids = get_top_ids()

    print(f'Model_ids: {model_ids}')

    ripser_root = f'{root}/{study_name}/{dir_name}'

    os.makedirs(ripser_root, exist_ok=True)

    for i, model_id in enumerate(tqdm(model_ids, desc='Running ripser')):
        all_layers = get_layers_func(
            dataset=dataset,
            model_id=model_id,
            input_layer=False,
            return_labels=False
        )
        distance_matrices = get_distance_matrices(all_layers, k=k)
        
        diagrams = []
        for dist_mat in distance_matrices:
            if gpu:
                sparse = get_sparse_matrix(dist_mat, thresh)
                # diagrams.append(rpp_py.run(f"--format distance --dim {maxdim} --threshold {thresh}", sparse))
                dgm = rpp_py.run(f'--format sparse --sparse --dim {maxdim} --threshold {thresh}', sparse)
                diagrams.append(normalize_diagram(dgm))
            else:
                dgm = ripser(dist_mat, distance_matrix=True, maxdim=maxdim, thresh=thresh)['dgms']
                diagrams.append(dgm)
                
        with open(f'{ripser_root}/model_{model_id}.dill', 'wb') as f:
            dill.dump(diagrams, f)

        # clean up memory usage
        del all_layers, distance_matrices, diagrams, dist_mat, dgm
        if gpu:
            del sparse
        gc.collect()

