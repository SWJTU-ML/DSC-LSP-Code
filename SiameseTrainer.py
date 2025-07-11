import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, random_split


class SiameseNet(nn.Module):
    def __init__(self, architecture: dict, input_dim: int):
        super(SiameseNet, self).__init__()
        self.architecture = architecture
        self.num_of_layers = self.architecture["n_layers"]
        self.layers = nn.ModuleList()

        current_dim = input_dim
        for layer, dim in self.architecture.items():
            if layer == "n_layers":
                continue
            next_dim = dim
            layer = nn.Sequential(nn.Linear(current_dim, next_dim), nn.ReLU())
            self.layers.append(layer)
            current_dim = next_dim

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple: 
        output1 = self.forward_once(x1) 
        output2 = self.forward_once(x2) 
        return output1, output2
  

class SiameseDataset:
    def __init__(self, pairs: list):

        self.pairs = pairs

    def __getitem__(self, index: int):
        x1 = self.pairs[index][0]
        x2 = self.pairs[index][1]
        label = self.pairs[index][2]
        return x1, x2, label

    def __len__(self):
        return len(self.pairs)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):  
        super(ContrastiveLoss, self).__init__()
        self.margin = margin 

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        
        euclidean = nn.functional.pairwise_distance(output1, output2) 
        positive_distance = torch.pow(euclidean, 2) 
        negative_distance = torch.pow(torch.clamp(self.margin - euclidean, min=0.0), 2) 
        loss = torch.mean((label * positive_distance) + ((1 - label) * negative_distance))
        return loss


class SiameseTrainer:
    def __init__(self, config: dict, device: torch.device):
        """
        Args:
            config (dict):          A dictionary containing the configuration
            device (torch.device):  The device to be used for training
        """

        self.device = device
        self.siamese_config = config["siamese"]
        self.lr = self.siamese_config["lr"] 
        self.epochs = self.siamese_config["epochs"]
        self.lr_decay = self.siamese_config["lr_decay"] 
        self.patience = self.siamese_config["patience"] 
        self.batch_size = self.siamese_config["batch_size"]
        self.architecture = self.siamese_config["architecture"]
        self.weights_path = "./weights/siamese_weights_mnist.pth"
    
    def train(self, X: torch.Tensor) -> SiameseNet:

        self.X = X.view(X.size(0), -1)

        self.criterion = ContrastiveLoss() 
        self.siamese_net = SiameseNet(self.architecture, input_dim=self.X.shape[1]).to(self.device)
        self.optimizer = optim.Adam(self.siamese_net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                              mode="min", 
                                                              factor=self.lr_decay,  
                                                              patience=self.patience)  

        if os.path.exists(self.weights_path):
            self.siamese_net.load_state_dict(torch.load(self.weights_path))
            return self.siamese_net
        
        train_loader, valid_loader = self._get_data_loader() 
        print("Training Siamese Network:")
        self.siamese_net.train() 
        for epoch in range(self.epochs): 
            train_loss = 0.0
            for x1, x2, label in train_loader:
                x1 = x1.to(self.device) 
                x1 = x1.view(x1.size(0), -1)
                x2 = x2.to(self.device) 
                x2 = x2.view(x2.size(0), -1)
                label = label.to(self.device)
                self.optimizer.zero_grad()
                output1, output2 = self.siamese_net(x1, x2)
                loss = self.criterion(output1, output2, label) 
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            
            train_loss /= len(train_loader) 
            valid_loss = self.validate(valid_loader) 
            self.scheduler.step(valid_loss) 
            current_lr = self.optimizer.param_groups[0]["lr"]

            if current_lr <= self.siamese_config["min_lr"]:
                break
            print("Epoch: {}/{}, Train Loss: {:.4f}, Valid Loss: {:.4f}, LR: {:.6f}".
            format(epoch + 1, self.epochs, train_loss, valid_loss, current_lr))
        
        torch.save(self.siamese_net.state_dict(), self.weights_path) 
        return self.siamese_net
    
    def validate(self, valid_loader: DataLoader) -> float:

        valid_loss = 0.0
        self.siamese_net.eval()
        with torch.no_grad():
            for x1, x2, label in valid_loader:
                x1 = x1.to(self.device) 
                x1 = x1.view(x1.size(0), -1)
                x2 = x2.to(self.device) 
                x2 = x2.view(x2.size(0), -1)
                label = label.to(self.device)
                output1, output2 = self.siamese_net(x1, x2)
                loss = self.criterion(output1, output2, label)
                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        return valid_loss

    def _get_knn_pairs(self) -> list:

        pairs = []
        X = self.X.detach().cpu().numpy()
        data_indices = np.arange(len(X)) 
        n_neighbors = self.siamese_config["n_neighbors"] 
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree").fit(X)  
        _, neighbors_indices = nbrs.kneighbors(X)

        for i in range(len(X)): 
            non_neighbors_indices = np.delete(data_indices, neighbors_indices[i]) 
            non_neighbors_random_chosen_indices = np.random.choice(non_neighbors_indices, n_neighbors) 

            positive_pairs = [[self.X[i], self.X[n], 1] for n in neighbors_indices[i][1:n_neighbors + 1]]
            negative_pairs = [[self.X[i], self.X[n], 0] for n in non_neighbors_random_chosen_indices] 

            pairs += positive_pairs  
            pairs += negative_pairs

        return pairs

    def _get_approx_nn_pairs(self) -> list:

        pairs = []
        n_samples = self.siamese_config["n_samples"] 
        n_neighbors = self.siamese_config["n_neighbors"] 
        indices = torch.randperm(self.X.shape[0])[:n_samples] 
        x_train = self.X[indices] 
        X_numpy = self.X[indices].detach().cpu().numpy() 
        data_indices = np.arange(len(x_train)) 

        ann = AnnoyIndex(X_numpy.shape[1], 'euclidean') 
        for i, x_ in enumerate(X_numpy):
            ann.add_item(i, x_)
        ann.build(50)

        neighbors_indices = np.empty((len(X_numpy), n_neighbors + 1))
        for i in range(len(X_numpy)):
            nn_i = ann.get_nns_by_item(i, n_neighbors + 1, include_distances=False)
            neighbors_indices[i, :] = np.array(nn_i)
        neighbors_indices = neighbors_indices.astype(int)

        print("Building dataset for the siamese network ...")
        for i in range(len(X_numpy)):
            non_neighbors_indices = np.delete(data_indices, neighbors_indices[i])

            neighbor_idx = np.random.choice(neighbors_indices[i][1:], 1)
            non_nbr_idx = np.random.choice(non_neighbors_indices, 1)

            positive_pairs = [[x_train[i], x_train[neighbor_idx], 1]]
            negative_pairs = [[x_train[i], x_train[non_nbr_idx], 0]]

            pairs += positive_pairs
            pairs += negative_pairs

        return pairs

    def _get_pairs(self) -> list:
        should_use_approx = self.siamese_config["use_approx"] 
        if should_use_approx:
            return self._get_approx_nn_pairs() 
        else:
            return self._get_knn_pairs() 

    def _get_data_loader(self) -> tuple:

        pairs = self._get_pairs() 
        siamese_dataset = SiameseDataset(pairs) 
        siamese_trainset_len = int(len(siamese_dataset) * 0.9) 
        siamese_validset_len = len(siamese_dataset) - siamese_trainset_len
        siamese_trainset, siamese_validset = random_split(siamese_dataset, [siamese_trainset_len, siamese_validset_len])
        siamese_trainloader = DataLoader(siamese_trainset, batch_size=self.siamese_config["batch_size"], shuffle=True) 
        siamese_validloader = DataLoader(siamese_validset, batch_size=self.siamese_config["batch_size"], shuffle=False) 
        return siamese_trainloader, siamese_validloader
        
    def get_data_loader_myself(self) -> tuple:

        X = self.X.detach().cpu().numpy()
        siamese_trainset_len = int(len(X) * 0.9)
        siamese_validset_len = len(X) - siamese_trainset_len
        siamese_trainset, siamese_validset = random_split(X, [siamese_trainset_len, siamese_validset_len])

        train_pairs = self.get_pairs_myself(siamese_trainset)
        siamese_train_dataset = SiameseDataset(train_pairs)
        number = self.siamese_config["n_neighbors"] * self.siamese_config["batch_size"]
        siamese_trainloader = DataLoader(siamese_train_dataset, batch_size=number, shuffle=False)  # 打乱就没意义了

        valid_pairs = self.get_pairs_myself(siamese_validset)
        siamese_valid_dataset = SiameseDataset(valid_pairs)
        siamese_validloader = DataLoader(siamese_valid_dataset, batch_size=number, shuffle=False)
        return siamese_trainloader, siamese_validloader

    def get_pairs_myself(self , X) -> list:
        train_pairs = []

        m = self.siamese_config["batch_size"]
        train_loader = DataLoader(X, batch_size=m, shuffle=True)
        for data in train_loader:
            data_pairs = self.get_knn_data_pairs(data)
            train_pairs += data_pairs
        return train_pairs
    
    def get_knn_data_pairs(self , X) -> list:
        pairs = []
        data_indices = np.arange(len(X))
        n_neighbors = self.siamese_config["n_neighbors"]
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="ball_tree").fit(X) 
        _, neighbors_indices = nbrs.kneighbors(X) 

        for i in range(len(X)):
            if i < len(X)/2:
                positive_pairs = [[self.X[i], self.X[n], 1] for n in neighbors_indices[i][1:n_neighbors + 1]]
                pairs += positive_pairs
            else:
                non_neighbors_indices = np.delete(data_indices, neighbors_indices[i]) 
                non_neighbors_random_chosen_indices = np.random.choice(non_neighbors_indices, n_neighbors) 
                negative_pairs = [[self.X[i], self.X[n], 0] for n in non_neighbors_random_chosen_indices]                        
                pairs += negative_pairs
        return pairs