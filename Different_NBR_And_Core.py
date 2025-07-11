
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import *
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, random_split, TensorDataset
from LE import *
class Nbr_Center_SpectralNetModel(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(Nbr_Center_SpectralNetModel, self).__init__()
        self.architecture = architecture
        self.num_of_layers = self.architecture["n_layers"]
        self.layers = nn.ModuleList()
        self.input_dim = input_dim
        
        current_dim = self.input_dim
        for layer, dim in self.architecture.items():
            next_dim = dim
            if layer == "n_layers":
                continue
            if layer == "output_dim":
                layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.Tanh())
                self.layers.append(layer)
            else:
                layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
                self.layers.append(layer)
                current_dim = next_dim
  

    def forward(self, x: torch.Tensor, is_orthonorm: bool = True) -> torch.Tensor:
        
        for layer in self.layers:
            x = layer(x)
        return x



class Nbr_Center_SpectralNetLoss(nn.Module):
    def __init__(self):
        super(Nbr_Center_SpectralNetLoss, self).__init__()
    def get_mask(self, m, n, k, device, is_eye: bool = False):
        if not is_eye:
            mask = torch.zeros([m, n]).to(device=device)
            for i in range(len(mask)):
                for j in range(i * k, (i+1) * k): 
                    mask[i, j] = 1
        else:
            mask = torch.eye(m, n).to(device=device)
        return mask, torch.count_nonzero(mask) 
    
    def forward(self, W_nbr: torch.Tensor, W_core: torch.Tensor, Y:torch.Tensor, Y_nbr: torch.Tensor, Y_core: torch.Tensor, LE: torch.Tensor, device, bk, bck, lamda,is_train: bool = True) -> torch.Tensor:

        Dynbr = torch.cdist(Y, Y_nbr)
        Dwy_nbr = W_nbr * Dynbr.pow(2)
        mask, l1 = self.get_mask(Dwy_nbr.size(0), Dwy_nbr.size(1), bk, device)
        Dwy_nbr = Dwy_nbr * mask
        if is_train:
            Dycore = torch.cdist(Y_nbr, Y_core)
            Dwy_core = W_core * Dycore.pow(2)
            mask, l2 = self.get_mask(Dwy_core.size(0), Dwy_core.size(1), bck, device)
            Dwy_core = Dwy_core * mask
            Dcore = torch.cdist(Y_core, LE) 
            Dcore = Dcore * torch.eye(Dcore.size(0), Dcore.size(1)).to(device=device)

            loss = torch.sum(Dwy_nbr) /l1  + torch.sum(Dwy_core) / l2 + lamda * torch.sum(Dcore) / Dcore.size(0)
        else:
            loss = torch.sum(Dwy_nbr) / l1
        

        return loss


class Nbr_Center_SpectralTrainer:
    def __init__(self, config: dict, device: torch.device, is_sparse = False):
        self.device = device
        self.is_sparse = is_sparse
        self.spectral_config = config["spectral"]

        self.lr = self.spectral_config["lr"]
        self.epochs = self.spectral_config["epochs"]
        self.lr_decay = self.spectral_config["lr_decay"]
        self.patience = self.spectral_config["patience"]
        self.batch_size = self.spectral_config["batch_size"]
        self.architecture = self.spectral_config["architecture"]
        self.center_k = self.spectral_config["center_k"]
        self.batch_k = self.spectral_config["batch_k"] 
        self.output_dim = self.architecture["output_dim"]
        self.LE_k = self.spectral_config["LE_k"]
        self.should_use_ae = config["should_use_ae"]
        self.batch_ck = self.spectral_config["batch_ck"]
        self.weights_path = "./SpectralNetWeights/Reuters/SN.pth"
        self.lamda = self.spectral_config["lamda"]
        self.sigma = self.spectral_config["sigma"]
    
    def train(self, X: torch.Tensor, y: torch.Tensor, siamese_net: nn.Module = None) -> Nbr_Center_SpectralNetModel:

        self.X = X.view(X.size(0), -1)
        self.y = y
        self.counter = 0
        self.siamese_net = siamese_net
        self.criterion = Nbr_Center_SpectralNetLoss()
        self.spectral_net = Nbr_Center_SpectralNetModel(self.architecture, input_dim=self.X.shape[1]).to(self.device)
        self.optimizer = optim.Adam(self.spectral_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode='min', 
                                                              factor=self.lr_decay, 
                                                              patience=self.patience)

        train_loader, valid_loader, core_point, core_LE = self._get_data_loader()

        print("Training Nbr_Center_SpectralNet:")
        all = 0
        for epoch in range(self.epochs):
            train_loss = 0.0
            count_nbr = 0
            count_core = 0
            for inx, X_grad in train_loader:

                X_nbr, X_core, X_LE, count1, count2 = self.get_train_batch(inx, X_grad, core_point, core_LE) 

                X_grad = X_grad.to(device=self.device)
                X_grad = X_grad.view(X_grad.size(0), -1)
                X_nbr = X_nbr.to(device=self.device)
                X_nbr = X_nbr.view(X_nbr.size(0), -1)
                X_core = X_core.to(device=self.device)
                X_core = X_core.view(X_core.size(0), -1)
                X_LE = X_LE.to(device=self.device)
                X_LE = X_LE.view(X_LE.size(0), -1)

                if self.is_sparse:
                    X_grad = make_batch_for_sparse_grapsh(X_grad)
                
                self.spectral_net.train()
                self.optimizer.zero_grad()
                
                if self.is_sparse:
                    X_grad = make_batch_for_sparse_grapsh(X_grad)

                Y = self.spectral_net(X_grad)
                Y_nbr = self.spectral_net(X_nbr)
                Y_core = self.spectral_net(X_core)

                with torch.no_grad():
                    if self.siamese_net is not None:
                        X_grad = self.siamese_net.forward_once(X_grad)
                        X_nbr = self.siamese_net.forward_once(X_nbr)
                        X_core = self.siamese_net.forward_once(X_core)
                W_nbr = self._get_affinity_matrix(X_grad, X_nbr, self.batch_k)
                W_core = self._get_affinity_matrix(X_nbr, X_core, self.batch_ck)

                loss = self.criterion(W_nbr, W_core, Y, Y_nbr, Y_core, X_LE, self.device, self.batch_k, self.batch_ck,self.lamda)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                count_nbr += count1
                count_core += count2

            train_loss /= len(train_loader)
            count_nbr /= len(train_loader)
            count_core /= len(train_loader)
            
            # Validation step
            valid_loss = self.validate(valid_loader)
            self.scheduler.step(valid_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]
            if current_lr <= self.spectral_config["min_lr"]: break
            print("Epoch: {}/{}, Train Loss: {:.7f}, Train batch nbr numbers: {:.7f}, Train batch core numbers: {:.7f}, Valid Loss: {:.7f}, LR: {:.9f}".
            format(epoch + 1, self.epochs, train_loss, count_nbr, count_core, valid_loss, current_lr))

        return self.spectral_net
    
    def validate(self, valid_loader: DataLoader) -> float:

        valid_loss = 0.0
        self.spectral_net.eval()
        with torch.no_grad():
            for batch in valid_loader:
                inx, X = batch
                X_nbr = self.get_valid_batch(inx, X)
                X, X_nbr = X.to(self.device), X_nbr.to(self.device)

                if self.is_sparse:
                    X = make_batch_for_sparse_grapsh(X)
                    
                Y = self.spectral_net(X)
                Y_nbr = self.spectral_net(X_nbr)
                
                with torch.no_grad():
                    if self.siamese_net is not None:
                        X = self.siamese_net.forward_once(X)
                        X_nbr = self.siamese_net.forward_once(X_nbr)
                
                W = self._get_affinity_matrix(X, X_nbr, self.batch_k)
                null = torch.randint(5, [6, 2])

                loss = self.criterion(W, null, Y, Y_nbr, null, null, self.device, self.batch_k, self.batch_ck, self.lamda,is_train = False)
                valid_loss += loss.item()
        
        self.counter += 1

        valid_loss /= len(valid_loader)
        return valid_loss
            
    
    def _get_affinity_matrix(self, X: torch.Tensor, Y: torch.Tensor, k) -> torch.Tensor:

        Dxy = torch.cdist(X, Y) 
        scale = torch.max(Dxy) * self.sigma 
        W = torch.exp(-torch.pow(Dxy, 2) / (scale ** 2))
        return W


    def _get_data_loader(self) -> tuple:
        self.train_dataset, self.valid_dataset = self.X.split([int(0.9 * len(self.X)), len(self.X) - int(0.9 * len(self.X))],dim=0)

        if self.should_use_ae:
            self.train_dataset = self.train_dataset.cpu()
            self.valid_dataset = self.valid_dataset.cpu()
        _, _, self.train_indices = get_K_NN_Rho(self.train_dataset, self.batch_k)
        _, _, self.valid_indices = get_K_NN_Rho(self.valid_dataset, self.batch_k)

        if self.siamese_net is not None:
            Siamese_data = self.siamese_net.forward_once(self.train_dataset.to(device=self.device))
        else:
            Siamese_data = self.train_dataset
        core_point, core_LE = self.get_core_point_and_embedding(Siamese_data) 

        if self.siamese_net is not None:
            core_point_compute = core_point.to(device=self.device)
            core_point_compute = self.siamese_net.forward_once(core_point_compute)
        else:
            core_point_compute = core_point
        self.nbr_core_inx = self.get_nbr_core_inx(Siamese_data, core_point_compute, self.batch_ck)

        train_ind = torch.arange(len(self.train_dataset))
        train_dataset = TensorDataset(train_ind, self.train_dataset)

        valid_ind = torch.arange(len(self.valid_dataset))
        valid_dataset = TensorDataset(valid_ind, self.valid_dataset)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader, valid_loader, core_point, core_LE

    def get_core_point_and_embedding(self, data):
        if self.siamese_net is not None:
            data = data.cpu().detach().numpy()
        rho, distances, indices = get_K_NN_Rho(data, self.center_k)
        pre_center, pre_arrow, pre_list = Get_Center(data, rho, distances, indices)
        center_index = list(np.unique(pre_center))

        core_point = torch.zeros(len(center_index), self.train_dataset.size(1))
        for index, i in enumerate(center_index):
            core_point[index] = self.train_dataset[i]

        snn = SNN(center_index, pre_center, rho, distances, indices)  
        dist = cal_pairwise_dist(core_point.cpu().numpy())
        
        max_dist = np.max(dist) 
       
        scale = (max_dist*self.sigma)**2  
        core_LE = le(core_point.cpu().numpy(), snn, n_dims = self.output_dim, n_neighbors = self.LE_k, t = scale) 
        return core_point, torch.from_numpy(core_LE)

    def get_nbr_core_inx(self, x, y, k):

        if self.siamese_net is not None:
            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(y)
        distances, indices = nbrs.kneighbors(x)
        return indices

    def get_train_batch(self, inx, grad, core_point, core_LE):
        nbr_i = torch.arange(len(inx)*self.batch_k)
        nbr = torch.zeros(len(inx)*self.batch_k, grad.size(1))
        j = 0
        for i in range(len(inx)):
            for node in self.train_indices[inx[i]][0:self.batch_k]:  
                nbr_i[j] = node
                j = j + 1
        nbr_i = nbr_i.numpy()
        for i in range(len(nbr)):
            nbr[i] = self.train_dataset[nbr_i[i]]

        nbr_core_i = torch.arange(len(nbr_i)*self.batch_ck)
        core = torch.zeros(len(nbr_i)*self.batch_ck, grad.size(1))
        LE = torch.zeros(len(nbr_i)*self.batch_ck, core_LE.size(1))
        j = 0
        for i in range(len(nbr_i)):
            for node in self.nbr_core_inx[nbr_i[i]][0:self.batch_ck]:
                nbr_core_i[j] = node
                j = j + 1
        nbr_core_i = nbr_core_i.numpy()
        for i in range(len(core)):
            core[i] = core_point[nbr_core_i[i]] 
            LE[i] = core_LE[nbr_core_i[i]] 
        return nbr, core, LE, len(np.unique(nbr_i)), len(np.unique(nbr_core_i))
    
    def get_valid_batch(self, inx, x):
        nbr_i = torch.arange(len(inx)*self.batch_k)
        nbr = torch.zeros(len(inx)*self.batch_k, x.size(1))
        j = 0
        for i in range(len(inx)):
            for node in self.valid_indices[inx[i]][0:self.batch_k + 1]:
                nbr_i[j] = node
                j = j + 1
        nbr_i = nbr_i.numpy()
        for i in range(len(nbr)):
            nbr[i] = self.valid_dataset[nbr_i[i]]
        return nbr

class ReduceLROnAvgLossPlateau(_LRScheduler):
    def __init__(self, optimizer, factor=0.1, patience=10, min_lr=0, verbose=False, min_delta=1e-4):

        self.factor = factor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.best = 1e5
        self.avg_losses = []
        self.min_lr = min_lr
        super(ReduceLROnAvgLossPlateau, self).__init__(optimizer)

    def get_lr(self):
        return [base_lr * self.factor ** self.num_bad_epochs
                for base_lr in self.base_lrs]

    def step(self, loss=1.0,  epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        

        current_loss = loss
        if len(self.avg_losses) < self.patience:
            self.avg_losses.append(current_loss)
        else:
            self.avg_losses.pop(0)
            self.avg_losses.append(current_loss)
        avg_loss = sum(self.avg_losses) / len(self.avg_losses)
        if avg_loss < self.best - self.min_delta:
            self.best = avg_loss
            self.wait = 0
        else:
            if self.wait >= self.patience:
                for param_group in self.optimizer.param_groups:
                    old_lr = float(param_group['lr'])
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        param_group['lr'] = new_lr
                        if self.verbose:
                            print(f'Epoch {epoch}: reducing learning rate to {new_lr}.')
                self.wait = 0
            self.wait += 1