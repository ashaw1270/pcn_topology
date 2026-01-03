# This file will contain the Trainer class that imports from all other modules
# Due to size constraints, this is a placeholder that will be completed
# The Trainer class maintains backward compatibility by importing functions
# from the refactored modules

# Import setup first to ensure paths are set
import trainer.setup  # noqa: F401

# Now import all necessary modules
import os
import gc
import ast
import json
import re
import sys
import subprocess
import threading
import time
import socket
from typing import Callable, Dict, Tuple
from collections import defaultdict
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import dill
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import optuna
from optuna_dashboard import run_server

import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.functional as pxf
import pcx.utils as pxu

# Import from our refactored modules
from trainer.data import load_dataset
from trainer.model import Model
from trainer.training.forward import forward, energy
from trainer.training.loops import train, eval, train_on_batch, eval_on_batch
from trainer.training.optimizers import get_opts
from trainer.training.optuna import run_optuna as _run_optuna_func, start_optuna_dashboard as _start_optuna_dashboard_func
from trainer.layers import get_layers as _get_layers_func, get_optimal_k_mnist, _LabelFilterDataset
from trainer.topology.ripser import run_ripser as _run_ripser_func
from trainer.topology.betti import _betti_at_eta, get_betti_mat as _get_betti_mat_func
from trainer.bootstrap.stats import (
    bootstrap_diffs as _bootstrap_diffs_func,
    save_bootstrap_diffs as _save_bootstrap_diffs_func,
    all_8_layer_models,
    filter_studies,
    get_bootstrap_data,
    all_bootstrap_stats as _all_bootstrap_stats_func
)
from trainer.inversion.reconstruction import (
    invert_output as _invert_output_func,
    _invert_once,
    _energy_given_xy,
    _rand_init
)
from trainer.metrics.mrd import mrd, get_mrd_list as _get_mrd_list_func, svm_accuracy
from trainer.metrics.tss import get_tss
from trainer.visualization.plots import (
    graph_betti_numbers as _graph_betti_numbers_func,
    compare_graphs as _compare_graphs_func,
    visualize_layers as _visualize_layers_func,
    plot_mrd_tss as _plot_mrd_tss_func
)


class Trainer:
    root = '../results'
    
    with open(f'{root}/D1/train_test_split.pkl', 'rb') as f:
        _data = dill.load(f)
        train_dataset_d1 = _data[0]
        test_dataset_d1 = _data[1]

    _all_data = np.load(f'{root}/D1/full_dataset.npz')
    _all_points = _all_data['points']
    _all_labels = _all_data['labels']

    full_dataset_d1 = (_all_points, _all_labels)

    _green_indices = np.where(_all_labels == 0)
    all_green = (_all_points[_green_indices], _all_labels[_green_indices])

    _red_indices = np.where(_all_labels == 1)
    all_red = (_all_points[_red_indices], _all_labels[_red_indices])

    with open(f'{root}/D1/svm.dill', 'rb') as f:
        svm = dill.load(f)  # this is trained on the full dataset

    with np.load(f'{root}/keys.npz') as f:
        model_keys = f['model_keys']
        epoch_keys_per_model = f['epoch_keys_per_model']
    
    def __init__(
        self,
        dataset: str,  # 'D1', 'D2', 'D3', or 'MNIST'
        hidden_dims: list[int],
        act_fn,
        study_name: str,
        root: str = None,
        residual=False,
        num_models=50,
        num_epochs=200,
        batch_size=32
    ):
        if dataset not in ['D1', 'D2', 'D3', 'MNIST']:
            raise Exception('dataset argument must be D1, D2, D3, or MNIST')
        
        self.dataset = dataset
        self.hidden_dims = hidden_dims
        self.act_fn = act_fn
        self.study_name = study_name
        self.residual = residual
        self.num_models = num_models
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.root = root if root is not None else f'../results/{dataset}'

        if dataset == 'D1':
            self.train_dataset = Trainer.train_dataset_d1
            self.test_dataset = Trainer.test_dataset_d1
            self.input_dim = 2
            self.output_dim = 2
            self.true_b = [9, None]
        else:
            # Standard datasets like MNIST/FashionMNIST/CIFAR10
            self.train_dataset, self.test_dataset, self.input_dim, self.output_dim = load_dataset(dataset)
            self.true_b = [None, None]  # Not meaningful for real image datasets

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(f'{self.root}/{self.study_name}', exist_ok=True)
        os.makedirs(f'{self.root}/{self.study_name}/trained_models', exist_ok=True)
            
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)

    def get_epoch_dataloader(self, model_id: int, epoch: int, train_dataset=None) -> DataLoader:
        """Get a DataLoader with epoch-specific seeding"""
        if train_dataset is None:
            train_dataset = self.train_dataset
        
        epoch_key = Trainer.epoch_keys_per_model[model_id][epoch]
        epoch_seed = int(jax.random.randint(epoch_key, (), 0, 2**31 - 1))
    
        # Set seeds for reproducibility
        torch.manual_seed(epoch_seed)
        np.random.seed(epoch_seed)
    
        # Create generator for this epoch
        generator = torch.Generator()
        generator.manual_seed(epoch_seed)
    
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            generator=generator
        )
    
        return train_dataloader

    def train_model(
        self,
        model_id,
        T,
        start_lr_w,
        start_lr_h,
        trans_mult,
        decay_rate,
        l2_w,
        l2_x,
        l2_h,
        early_stopping
    ):
        model_key = Trainer.model_keys[model_id]
        model = Model(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            act_fn=self.act_fn,
            model_key=model_key,
            residual=self.residual,
            l2_w=l2_w,
            l2_x=l2_x,
            l2_h=l2_h
        )
    
        # Dummy forward pass to initialize Vodes
        with pxu.step(model, pxc.STATUS.INIT, clear_params=pxc.VodeParam.Cache):
            forward(jax.numpy.zeros((self.batch_size, model.input_dim.get())), None, model=model)
    
        optim_w, optim_h = get_opts(
            model=model,
            init_w=start_lr_w,
            init_h=start_lr_h,
            transition_steps=(len(self.train_dataset) // self.batch_size) * trans_mult,
            decay_rate=decay_rate,
            T=T
        )
    
        best_test_acc = 0
        for epoch in range(self.num_epochs):
            train_loader = self.get_epoch_dataloader(model_id, epoch)
            train(train_loader, T=T, model=model, optim_w=optim_w, optim_h=optim_h)
    
            test_acc, _ = eval(self.test_loader, model=model)
            if test_acc > best_test_acc:
                pxu.save_params(model, f'{self.root}/{self.study_name}/trained_models/model_{model_id}')
            
            best_test_acc = max(best_test_acc, test_acc)
            
            if test_acc == 1 or early_stopping(epoch, test_acc):
                break
    
        with open(f'{self.root}/{self.study_name}/accuracies.txt', 'a') as f:
            f.write(f'{model_id}: {best_test_acc}\n')

        # clean up memory usage
        del model, train_loader
        gc.collect()
        jax.clear_caches()

    def train_all_models(
        self,
        T,
        start_lr_w,
        start_lr_h,
        trans_mult,
        decay_rate,
        l2_w=0.0,
        l2_x=0.0,
        l2_h=0.0,
        model_ids=None,
        early_stopping=lambda epoch, test_acc: epoch > 100 and test_acc < 0.8
    ):
        if model_ids is None:
            model_ids = range(self.num_models)

        for i in tqdm(model_ids):
            self.train_model(
                model_id=i,
                T=T,
                start_lr_w=start_lr_w,
                start_lr_h=start_lr_h,
                trans_mult=trans_mult,
                decay_rate=decay_rate,
                l2_w=l2_w,
                l2_x=l2_x,
                l2_h=l2_h,
                early_stopping=early_stopping
            )

    @staticmethod
    def start_optuna_dashboard():
        return _start_optuna_dashboard_func()

    # Continue with remaining methods...
    # Due to size, I'll need to add the rest of the methods
    # For now, this is the structure

    def get_layers(self, dataset, input_layer: bool, return_labels: bool, model=None, model_id=None):
        return _get_layers_func(
            dataset=dataset,
            input_layer=input_layer,
            return_labels=return_labels,
            model=model,
            model_id=model_id,
            model_keys=Trainer.model_keys,
            root=self.root,
            study_name=self.study_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            act_fn=self.act_fn,
            residual=self.residual
        )

    @staticmethod
    def get_optimal_k_mnist(*args, **kwargs):
        return get_optimal_k_mnist(*args, **kwargs)

    def run_ripser(self, *args, **kwargs):
        return _run_ripser_func(
            get_layers_func=self.get_layers,
            dataset=kwargs.get('dataset'),
            root=self.root,
            study_name=self.study_name,
            true_b=self.true_b,
            dataset_name=self.dataset,
            input_dim=self.input_dim,
            *args,
            **{k: v for k, v in kwargs.items() if k not in ['dataset']}
        )

    def get_betti_mat(self, dir_name='ripser_only_0_k14', eta=2.5, dim=0):
        return _get_betti_mat_func(
            dir_name=dir_name,
            eta=eta,
            dim=dim,
            root=self.root,
            study_name=self.study_name,
            true_b=self.true_b
        )

    def invert_output(self, *args, **kwargs):
        return _invert_output_func(
            *args,
            root=self.root,
            study_name=self.study_name,
            num_models=self.num_models,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            act_fn=self.act_fn,
            residual=self.residual,
            model_keys=Trainer.model_keys,
            **kwargs
        )

    def get_mrd_list(self, *args, **kwargs):
        return _get_mrd_list_func(
            *args,
            root=self.root,
            study_name=self.study_name,
            all_green=Trainer.all_green,
            **kwargs
        )

    # Static methods that delegate to module functions
    @staticmethod
    def mrd(*args, **kwargs):
        return mrd(*args, **kwargs)

    @staticmethod
    def svm_accuracy(*args, **kwargs):
        return svm_accuracy(*args, **kwargs)

    @staticmethod
    def bootstrap_diffs(*args, **kwargs):
        return _bootstrap_diffs_func(*args, Trainer=Trainer, **kwargs)

    @staticmethod
    def save_bootstrap_diffs(*args, **kwargs):
        return _save_bootstrap_diffs_func(*args, Trainer=Trainer, **kwargs)

    @staticmethod
    def all_8_layer_models():
        return all_8_layer_models()

    @staticmethod
    def filter_studies(*args, **kwargs):
        return filter_studies(*args, **kwargs)

    @staticmethod
    def get_bootstrap_data(*args, **kwargs):
        return get_bootstrap_data(*args, **kwargs)

    @staticmethod
    def all_bootstrap_stats(*args, **kwargs):
        return _all_bootstrap_stats_func(*args, Trainer=Trainer, **kwargs)

    @staticmethod
    def get_tss(*args, **kwargs):
        return get_tss(*args, **kwargs)

    # Additional methods preserved from original Trainer class
    
    def run_optuna(
        self,
        num_models,
        T_func,
        start_lr_w_func,
        start_lr_h_func,
        trans_mult_func,
        decay_rate_func,
        early_stopping_within_key=lambda epoch, test_acc: epoch > 100 and test_acc < 0.8,
        early_stopping_whole_trial=lambda trial_num, test_acc: trial_num == 0 and test_acc == 0.61328125,
        prune_after_num_trials=8,
        prune_after_keys_tried=2,
        add_trials: list[dict] = None
    ):
        return _run_optuna_func(
            num_models=num_models,
            T_func=T_func,
            start_lr_w_func=start_lr_w_func,
            start_lr_h_func=start_lr_h_func,
            trans_mult_func=trans_mult_func,
            decay_rate_func=decay_rate_func,
            dataset=self.dataset,
            root=self.root,
            study_name=self.study_name,
            batch_size=self.batch_size,
            num_epochs=self.num_epochs,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=self.hidden_dims,
            act_fn=self.act_fn,
            residual=self.residual,
            model_keys=Trainer.model_keys,
            get_epoch_dataloader_func=self.get_epoch_dataloader,
            early_stopping_within_key=early_stopping_within_key,
            early_stopping_whole_trial=early_stopping_whole_trial,
            prune_after_num_trials=prune_after_num_trials,
            prune_after_keys_tried=prune_after_keys_tried,
            add_trials=add_trials
        )

    def graph_betti_numbers(
        self,
        title=None,
        dir_name='ripser_only_0_k14',
        k=14,
        etas=[2.5],
        maxdim=0,
        figsize=(10, 6),
        save=False,
        filename=None
    ):  
        return _graph_betti_numbers_func(
            root=self.root,
            study_name=self.study_name,
            dataset=self.dataset,
            true_b=self.true_b,
            title=title,
            dir_name=dir_name,
            k=k,
            etas=etas,
            maxdim=maxdim,
            figsize=figsize,
            save=save,
            filename=filename
        )

    @staticmethod
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
        filename=None
    ):
        return _compare_graphs_func(
            title=title,
            studies=studies,
            exclude=exclude,
            dir_names=dir_names,
            eta=eta,
            k=k,
            dim=dim,
            legend=legend,
            colors=colors,
            save_graph=save_graph,
            save_data=save_data,
            filename=filename,
            Trainer=Trainer
        )

    def visualize_layers(self, dataset, model_id, k=6, track_initial=True):
        return _visualize_layers_func(
            get_layers_func=self.get_layers,
            dataset=dataset,
            model_id=model_id,
            k=k,
            track_initial=track_initial
        )

    @staticmethod
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
        print_outliers=False
    ):
        return _plot_mrd_tss_func(
            data=data,
            to_calculate=to_calculate,
            metric=metric,
            mean=mean,
            title=title,
            figsize=figsize,
            s=s,
            lowess_extra_gap=lowess_extra_gap,
            save=save,
            filename=filename,
            add_d=add_d,
            print_all=print_all,
            print_outliers=print_outliers,
            Trainer=Trainer
        )

