"""
Author: Egheosa Ogbomo
Date: 19th July 2023

Script to prepare datasets for use with Pytorch's Geometric library by
converting SMILES strings to 

Adapted from:
https://www.blopig.com/blog/2022/02/how-to-turn-a-smiles-string-into-a-molecular-graph-for-
pytorch-geometric/
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from matplotlib.colors import to_rgb
import matplotlib
import seaborn as sns
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
import os
import math
import json

CHECKPOINT_PATH = 'C:/Users/eeo21/Desktop/VS Code Files/GNN_Viscosity_Prediction/SavedModel'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

###################### PREPARE DATASET ##############################

def one_hot_encoding(x, permitted_list):
    """
    Adds input elements X which are not in the permitted list to the last element
    of the permitted list. 

    We define the permitted list of atoms in function below.
    """

    #If statement to check if atom is in permitted list

    if x not in permitted_list:
        x = permitted_list[-1] # Places non-recognised atom at the end of the list of permitted atoms

    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

test_string = 'COCCCOCC=CCC(=O)C=CCCOC(C=Cc1ccccc1)OCOCC=C'

def get_atom_features(atom, 
                      use_chirality = True, 
                      hydrogens_implicit = True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """

    # define list of permitted atoms
    
    permitted_list_of_atoms =  ['C','N','O','S','F','Si','P','Cl','Br','Mg','Na','Ca','Fe','As',
                                'Al','I', 'B','V','K','Tl','Yb','Sb','Sn','Ag','Pd','Co','Se','Ti',
                                'Zn', 'Li','Ge','Cu','Au','Ni','Cd','In','Mn','Zr','Cr','Pt','Hg',
                                'Pb']
    
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    
    # compute atom features

    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()), ["S", "SP", "SP2", "SP3", 
                                                                             "SP3D", "SP3D2", "OTHER"])
    
    is_in_a_ring_enc = [int(atom.IsInRing())]
    
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    
    atomic_mass_scaled = [float((atom.GetMass() - 10.812)/116.092)]
    
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5)/0.6)]
    
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64)/0.76)]
    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled
                                    
    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()), ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW", "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc
    return np.array(atom_feature_vector)

def get_bond_features(bond, 
                      use_stereochemistry = True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)

def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    """
    Inputs:
    
    x_smiles = [smiles_1, smiles_2, ....] ... a list of SMILES strings
    y = [y_1, y_2, ...] ... a list of numerial labels for the SMILES strings (such as associated pKi values)
    
    Outputs:
    
    data_list = [G_1, G_2, ...] ... a list of torch_geometric.data.Data objects which represent labeled molecular graphs that can readily be used for machine learning
    
    """
    
    data_list = []
    
    for (smiles, y_val) in zip(x_smiles, y):
        
        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)
        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2*mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0,1)))
        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
            
        X = torch.tensor(X, dtype = torch.float)
        
        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim = 0)
        
        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))
        
        for (k, (i,j)) in enumerate(zip(rows, cols)):
            
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i),int(j)))
        
        EF = torch.tensor(EF, dtype = torch.float)
        
        # construct label tensor
        y_tensor = torch.tensor(np.array([y_val]), dtype = torch.float)

        # construct Pytorch Geometric data object and append to data list
        data_list.append(Data(x = X, edge_index = E, edge_attr = EF, y = y_tensor))
    return data_list

#### Importing and formatiing the Kajita Dataset

Dataset = pd.read_csv('C:/Users/eeo21/Desktop/VS Code Files/GNN_Viscosity_Prediction/Dataset.csv')

Smiles_list = Dataset['smiles'].tolist()
Y_labels = Dataset['VI'].tolist()

data_list = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(Smiles_list[:300], Y_labels[:300])

################## DEFINE MODEL ARCHITECTURE ######################

## Defining different GNN layers from the Torch Geometric library for use later on 

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(2, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

gnn_model = GCN(hidden_channels=16)
print(gnn_model)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

gnn_model.eval()

# out = gnn_model(data.x, data.edge_index)
# visualize(out, color=data.y)

# canonical training loop for a Pytorch Geometric GNN model gnn_model
# create list of molecular graph objects from list of SMILES x_smiles and list of labels y
# create dataloader for training
dataloader = DataLoader(dataset = data_list, batch_size = 2**7)
# define loss function
loss_function = nn.MSELoss()
# define optimiser
optimiser = torch.optim.Adam(gnn_model.parameters(), lr = 1e-3)
# loop over 10 training epochs
for epoch in range(10):
    # set model to training mode
    gnn_model.train()
    # loop over minibatches for training
    for (k, batch) in enumerate(dataloader):
        # compute current value of loss function via forward pass
        output = gnn_model(batch)
        loss_function_value = loss_function(output[:,0], torch.tensor(batch.y, dtype = torch.float32))
        # set past gradient to zero
        optimiser.zero_grad()
        # compute current gradient via backward pass
        loss_function_value.backward()
        # update model weights using gradient and optimisation method
        optimiser.step()

import torch_geometric
tu_dataset = torch_geometric.datasets.TUDataset(root=CHECKPOINT_PATH, name="MUTAG")

print(tu_dataset.num_classes)

################## DEFINE TRAINING LOOP ###############################

# def train_graph_model(model_name, **model_kwargs):
#     pl.seed_everything(42)

#     # Create a PyTorch Lightning trainer with the generation callback
#     root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
#     os.makedirs(root_dir, exist_ok=True)
#     trainer = pl.Trainer(default_root_dir=root_dir,
#                          callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
#                          accelerator="gpu" if str(device).startswith("cuda") else "cpu",
#                          devices=1,
#                          max_epochs=500,
#                          enable_progress_bar=False)

#     # Check whether pretrained model exists. If yes, load it and skip training
#     pretrained_filename = os.path.join(CHECKPOINT_PATH, f"GraphLevel{model_name}.ckpt")
#     if os.path.isfile(pretrained_filename):
#         print("Found pretrained model, loading...")
#         model = GraphLevelGNN.load_from_checkpoint(pretrained_filename)
#     else:
#         pl.seed_everything(42)
#         model = GraphLevelGNN(c_in=tu_dataset.num_node_features,
#                               c_out=1 if tu_dataset.num_classes==2 else tu_dataset.num_classes,
#                               **model_kwargs)
#         trainer.fit(model, graph_train_loader, graph_val_loader)
#         model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
#     # Test best model on validation and test set
#     train_result = trainer.test(model, graph_train_loader, verbose=False)
#     test_result = trainer.test(model, graph_test_loader, verbose=False)
#     result = {"test": test_result[0]['test_acc'], "train": train_result[0]['test_acc']}
#     return model, result

# model, result = train_graph_model(model_name="Graph_Viscosity_Index_Predictor",
#                                        c_hidden=256,
#                                        layer_name="GraphConv",
#                                        num_layers=3,
#                                        dp_rate_linear=0.5,
#                                        dp_rate=0.0)


