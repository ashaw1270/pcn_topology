"""
Public API for the trainer package.
"""

# Main classes and functions
from trainer.core import Trainer
from trainer.model import Model
from trainer.data import load_dataset

# Training functions
from trainer.training.forward import forward, energy
from trainer.training.loops import train, eval, train_on_batch, eval_on_batch
from trainer.training.optimizers import get_opts

# Layer utilities
from trainer.layers import get_optimal_k_mnist

# Topology functions
from trainer.topology.ripser import run_ripser
from trainer.topology.betti import _betti_at_eta, get_betti_mat

# Bootstrap functions
from trainer.bootstrap.stats import (
    bootstrap_diffs,
    save_bootstrap_diffs,
    all_8_layer_models,
    filter_studies,
    get_bootstrap_data,
    all_bootstrap_stats
)

# Inversion/reconstruction functions
from trainer.inversion.reconstruction import invert_output

# Metrics
from trainer.metrics.mrd import mrd, get_mrd_list, svm_accuracy
from trainer.metrics.tss import get_tss

__all__ = [
    'Trainer',
    'Model',
    'load_dataset',
    'forward',
    'energy',
    'train',
    'eval',
    'train_on_batch',
    'eval_on_batch',
    'get_opts',
    'get_optimal_k_mnist',
    'run_ripser',
    '_betti_at_eta',
    'get_betti_mat',
    'bootstrap_diffs',
    'save_bootstrap_diffs',
    'all_8_layer_models',
    'filter_studies',
    'get_bootstrap_data',
    'all_bootstrap_stats',
    'invert_output',
    'mrd',
    'get_mrd_list',
    'svm_accuracy',
    'get_tss',
]
