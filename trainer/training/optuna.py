import os
import gc
import socket
import threading
import time
import jax
import jax.numpy as jnp
import numpy as np
import dill
import optuna
from optuna_dashboard import run_server
from torch.utils.data import DataLoader

from trainer.model import Model
from trainer.training.forward import forward
from trainer.training.loops import train, eval
from trainer.training.optimizers import get_opts
import pcx.utils as pxu
import pcx.predictive_coding as pxc


def run_optuna(
    num_models,
    T_func,
    start_lr_w_func,
    start_lr_h_func,
    trans_mult_func,
    decay_rate_func,
    # Trainer instance attributes
    dataset,
    root,
    study_name,
    batch_size,
    num_epochs,
    input_dim,
    output_dim,
    hidden_dims,
    act_fn,
    residual,
    model_keys,
    get_epoch_dataloader_func,
    early_stopping_within_key=lambda epoch, test_acc: epoch > 100 and test_acc < 0.8,
    early_stopping_whole_trial=lambda trial_num, test_acc: trial_num == 0 and test_acc == 0.61328125,
    prune_after_num_trials=8,
    prune_after_keys_tried=2,
    add_trials: list[dict] = None
):
    """
    Run Optuna hyperparameter optimization.
    
    Parameters
    ----------
    num_models : int
        Number of models to train per trial
    T_func : callable
        Function that takes an Optuna trial and returns T (inference steps)
    start_lr_w_func : callable
        Function that takes an Optuna trial and returns initial learning rate for weights
    start_lr_h_func : callable
        Function that takes an Optuna trial and returns initial learning rate for hidden states
    trans_mult_func : callable
        Function that takes an Optuna trial and returns transition multiplier
    decay_rate_func : callable
        Function that takes an Optuna trial and returns decay rate
    dataset : str
        Dataset name ('D1', 'D2', 'D3', or 'MNIST')
    root : str
        Root directory for results
    study_name : str
        Name of the study
    batch_size : int
        Batch size for training
    num_epochs : int
        Number of epochs to train
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    hidden_dims : list[int]
        Hidden layer dimensions
    act_fn : callable
        Activation function
    residual : bool
        Whether to use residual connections
    model_keys : array-like
        Array of model keys for initialization
    get_epoch_dataloader_func : callable
        Function that takes (model_id, epoch, train_dataset) and returns DataLoader
    early_stopping_within_key : callable, optional
        Early stopping function for individual models
    early_stopping_whole_trial : callable, optional
        Early stopping function for entire trial
    prune_after_num_trials : int, default=8
        Number of trials to run before pruning starts
    prune_after_keys_tried : int, default=2
        Number of keys to try before pruning starts
    add_trials : list[dict], optional
        Additional trials to enqueue
    
    Returns
    -------
    None
        Side effect: creates Optuna study and runs optimization
    """
    filetype = 'pkl' if dataset == 'D1' else 'dill'
    with open(f'{root}/train_test_split_25_percent.{filetype}', 'rb') as f:
        train_sub, test_sub = dill.load(f)

    test_loader = DataLoader(test_sub, batch_size=batch_size, shuffle=False, drop_last=True)
    
    def objective(trial):
        T = T_func(trial)
        start_lr_w = start_lr_w_func(trial)
        start_lr_h = start_lr_h_func(trial)
        trans_mult = trans_mult_func(trial)
        decay_rate = decay_rate_func(trial)
    
        results = []
        for i in range(num_models):
            model_key = model_keys[i]
            model = Model(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                act_fn=act_fn,
                model_key=model_key,
                residual=residual
            )
        
            # Dummy forward pass to initialize Vodes
            with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
                forward(jax.numpy.zeros((batch_size, model.input_dim.get())), None, model=model)
        
            optim_w, optim_h = get_opts(
                model=model,
                init_w=start_lr_w,
                init_h=start_lr_h,
                transition_steps=(len(train_sub) // batch_size) * trans_mult,
                decay_rate=decay_rate,
                T=T,
            )
        
            best_test_acc = 0
            for epoch in range(num_epochs):
                train_loader = get_epoch_dataloader_func(i, epoch, train_sub)
                train(train_loader, T=T, model=model, optim_w=optim_w, optim_h=optim_h)
        
                test_acc, _ = eval(test_loader, model=model)
                best_test_acc = max(best_test_acc, test_acc)
        
                if test_acc == 1 or (early_stopping_within_key is not None and early_stopping_within_key(epoch, test_acc)):
                    break

            results.append(best_test_acc)

            trial.report(np.mean(results), i+1)

            del model, optim_w, optim_h, train_loader
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if early_stopping_whole_trial is not None and early_stopping_whole_trial(i, best_test_acc):
                return np.mean(results)

        mean_acc = np.mean(results)

        gc.collect()
        jax.clear_caches()
        
        return mean_acc

    
    home_dir = os.path.expanduser("~")
    storage_url = f"sqlite:///{home_dir}/optimization.db"
    
    study = optuna.create_study(
        study_name=f'{dataset}_{study_name}',
        storage=storage_url,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=prune_after_num_trials,
            n_warmup_steps=prune_after_keys_tried,
            interval_steps=1
        )
    )

    if add_trials is not None:
        for trial in add_trials:
            study.enqueue_trial(trial)
    
    study.optimize(objective)


def start_optuna_dashboard():
    """
    Start the Optuna dashboard in a separate thread.
    
    This function finds an available port, starts the Optuna dashboard server,
    and prints instructions for accessing it via SSH tunnel.
    """
    def find_available_port(start_port=8080):
        for port in range(start_port, start_port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
        raise Exception("No available ports found")
    
    def start_dashboard(storage_url, port=8080):
        """Start Optuna dashboard in a separate thread"""
        try:
            print(f"Starting Optuna dashboard on http://localhost:{port}")
            run_server(storage_url, host="0.0.0.0", port=port)
        except Exception as e:
            print(f"Error starting dashboard: {e}")
            
    
    home_dir = os.path.expanduser("~")
    storage_url = f"sqlite:///{home_dir}/optimization.db"
    port = find_available_port(8080)
    
    # Start dashboard
    dashboard_thread = threading.Thread(
        target=start_dashboard, 
        args=(storage_url, port),
        daemon=True
    )
    dashboard_thread.start()
    time.sleep(3)  # Give it time to start
    
    print('Run the following command in your local terminal on YOUR MACHINE, not in the cluster:')
    print(f'ssh -L {port}:{socket.gethostname()}:{port} {{YOUR USERNAME}}@discovery.usc.edu')
    print(f'Once you enter the cluster from your terminal, navigate to http://localhost:{port} in a web browser')

