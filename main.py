import sys
import json
import torch
import random
import numpy as np

from utils import *
from data import load_data
from metrics import Metrics
from sklearn.cluster import KMeans
from SpectralNet import SpectralNet
from scipy.spatial.distance import cdist
import sklearn.metrics as metrics
from LE import *
from sklearn.manifold import TSNE

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

print(torch.cuda.is_available())
class InvalidMatrixException(Exception):
    pass
def set_seed(seed: int = 0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(config_path):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open (config_path, 'r') as f:
        config = json.load(f)
    print(config_path)
    dataset = config["dataset"]
    n_clusters = config["n_clusters"]
    should_check_generalization = config["should_check_generalization"] 
    
    x_train, x_test, y_train, y_test = load_data(dataset) 

    if not should_check_generalization:
        if y_train is None:
            x_train = torch.cat([x_train, x_test]) 
            
        else:
            x_train = torch.cat([x_train, x_test])
            y_train = torch.cat([y_train, y_test])

    try:
        spectralnet = SpectralNet(n_clusters=n_clusters, config=config)
        spectralnet.fit(x_train, y_train) 

    except torch._C._LinAlgError:
        raise InvalidMatrixException("The output of the network is not a valid matrix to the orthogonalization layer. " 
                                     "Try to decrease the learning rate to fix the problem.") 
 
    cluster_assignments = spectralnet.center_predict(x_train)
    if y_train is not None:    
        y = y_train.detach().cpu().numpy()
        acc_score = Metrics.acc_score(cluster_assignments, y, n_clusters) 
        nmi_score = Metrics.nmi_score(cluster_assignments, y)
        ari_score = Metrics.ari_score(cluster_assignments, y)
        embeddings = spectralnet.embeddings_
        print(f"ACC: {np.round(acc_score, 3)}")
        print(f"NMI: {np.round(nmi_score, 3)}")
        print(f"ARI: {np.round(ari_score, 3)}")
    
    return embeddings, cluster_assignments



if __name__ == "__main__":

    embeddings, assignments = main("./config/CNAE.json")