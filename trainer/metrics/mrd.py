import os
import numpy as np
import dill
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def mrd(recons, dataset, graph=False):
    """
    Compute minimum reconstruction distance (MRD) for a given target class.
    A lower MRD means the reconstructions better fit the target data.
    
    dataset should only contain data from one class, no labels.
    """
    if graph:
        plt.scatter(dataset[:,0], dataset[:,1], s=4, color='green', label='True Dataset')
        plt.scatter(recons[:,0], recons[:,1], s=4, color='red', label='Reconstructions')
        plt.axis('equal')
        plt.legend()
        plt.show()
        
    D = cdist(recons, dataset)
    mrd = np.mean(np.min(D, axis=1))
    return float(mrd)


def get_mrd_list(dir_name='reconstructions_s100', dataset=None, include_ids=False, root=None, study_name=None, all_green=None):
    """
    Get MRD list for all reconstructions in a directory.
    all_green: tuple of (points, labels) for the green class dataset
    """
    if dataset is None:
        if all_green is None:
            raise ValueError("Either dataset or all_green must be provided")
        dataset = all_green[0]
        
    recon_root = f'{root}/{study_name}/{dir_name}'
    mrds = []
    for file in os.listdir(recon_root):
        if 'dill' in file:
            with open(f'{recon_root}/{file}', 'rb') as f:
                recons = dill.load(f)
                mrd_val = mrd(recons, dataset)
                if include_ids:
                    mrds.append((file, mrd_val))
                else:
                    mrds.append(mrd_val)
            
    return mrds


def svm_accuracy(study, dir_name='reconstructions_s100', root=None, svm=None):
    """
    Compute SVM accuracy for reconstructions.
    svm: trained SVM model (must be provided)
    """
    if svm is None:
        raise ValueError("svm model must be provided")
    
    recon_root = f'{root}/{study}/{dir_name}'
    accs = []
    
    for file in os.listdir(recon_root):
        if 'dill' in file:
            with open(f'{recon_root}/{file}', 'rb') as f:
                recons = dill.load(f)
                labels = np.zeros(len(recons))
                predictions = svm.predict(recons)
                accuracy = accuracy_score(labels, predictions)
                accs.append(accuracy)
                
    return accs

