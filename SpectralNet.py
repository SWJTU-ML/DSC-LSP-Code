import torch
import numpy as np

from AETrainer import *
from SiameseTrainer import *
from sklearn.cluster import KMeans
from metrics import Metrics
from Different_NBR_And_Core import *
from utils import *
from sklearn import manifold
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import scipy.io as io
from sklearn.manifold import TSNE

class SpectralNet:
    def __init__(self, n_clusters: int, config: dict):

        self.n_clusters = n_clusters
        self.config = config
        self.embeddings_ = None 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


    def fit(self, X: torch.Tensor ,y: torch.Tensor = None):
        should_use_ae = self.config["should_use_ae"]
        should_use_siamese = self.config["should_use_siamese"] 
        create_weights_dir()

        if should_use_ae:  
            ae_trainer = AETrainer(self.config, self.device) 
            self.ae_net = ae_trainer.train(X) 
            X = ae_trainer.embed(X) 
        if should_use_siamese:
            siamese_trainer = SiameseTrainer(self.config, self.device)
            self.siamese_net = siamese_trainer.train(X)
        else:
            self.siamese_net = None

        is_sparse = self.config["is_sparse_graph"] 
        if is_sparse: 
            build_ann(X)    
            
        spectral_trainer = Nbr_Center_SpectralTrainer(self.config, self.device, is_sparse=is_sparse)
        self.spec_net = spectral_trainer.train(X, y, self.siamese_net) 
        
        
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
  
        X = X.view(X.size(0), -1)
        X = X.to(self.device)
        should_use_ae = self.config["should_use_ae"]
        if should_use_ae:
            X = self.ae_net.encoder(X) 
        self.embeddings_ = self.spec_net(X, is_orthonorm = False).detach().cpu().numpy() 

        cluster_assignments = self._get_clusters_by_kmeans(self.embeddings_) 
        return cluster_assignments

    def center_predict(self, X: torch.Tensor) -> np.ndarray:
        X = X.view(X.size(0), -1)
        X = X.to(self.device)
        should_use_ae = self.config["should_use_ae"]
        if should_use_ae:
            X = self.ae_net.encoder(X) 
        self.embeddings_ = self.spec_net(X).detach().cpu().numpy()

        cluster_assignments = self._get_clusters_by_kmeans(self.embeddings_)
        return cluster_assignments
    
    def _get_clusters_by_kmeans(self, embeddings: np.ndarray) -> np.ndarray:
        
        kmeans = KMeans(n_clusters=self.n_clusters).fit(embeddings)  
        cluster_assignments = kmeans.predict(embeddings)
        return cluster_assignments