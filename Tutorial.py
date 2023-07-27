
# Standard libraries
import os

# For downloading pre-trained models
import urllib.request
from urllib.error import HTTPError

# PyTorch Lightning
import lightning as L

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# PyTorch geometric
import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn

from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor

# AVAIL_GPUS = min(1, torch.cuda.device_count())
# BATCH_SIZE = 256 if AVAIL_GPUS else 64
# # Path to the folder where the datasets are/should be downloaded
# DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")

# print(DATASET_PATH)
# # Path to the folder where the pretrained models are saved
# CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/GNNs/")

# Setting the seed
L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

##### Constructing Graphs From Dataset

## Github URL where saved models are stored for this tutorial
#base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial7/"
## Files to download
#pretrained_files = ["NodeLevelMLP.ckpt", "NodeLevelGNN.ckpt", "GraphLevelGraphConv.ckpt"]

## Create checkpoint path if it doesn't exist yet
#os.makedirs(CHECKPOINT_PATH, exist_ok=True)

## For each file, check whether it already exists. If not, try downloading it.

#for file_name in pretrained_files:
#     file_path = os.path.join(CHECKPOINT_PATH, file_name)
#     if "/" in file_name:
#         os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
#     if not os.path.isfile(file_path):
#         file_url = base_url + file_name
#         print("Downloading %s..." % file_url)
#         try:
#             urllib.request.urlretrieve(file_url, file_path)
#         except HTTPError as e:
#             print(
#                 "Something went wrong. Please try to download the file from the GDrive folder,"
#                 " or contact the author with the full output including the following error:\n",
#                 e,
#             )

"""
GRAPH CONVOLUTIONAL NETWORK TUTORIAL 
"""

# Set a class up to define the Graph Convolutional Network
# In Pytorch, convention is to use nn.Module as a parent class for all classes used to define new models 

class GCNLayer(nn.Module):
    """
    - Super method used to take __init__ method from torch's Module class 
    - c_in: Dimensionality of input features
    - c_out: Dimensionality of output feature
    """
    def __init__(self, c_in, c_out):
        
        super().__init__()
        
        # Define a 'projection' method in the class which is just a linear transformation
        self.projection = nn.Linear(c_in, c_out)

    # Defining the forward pass
    def forward(self, node_feats, adj_matrix):
        """
        Args:
            - node_feats: Tensor with node features of shape [batch_size, num_nodes, c_in]
            - adj_matrix: Batch of adjacency matrices of the graph. If there is an edge from i to j,
                         adj_matrix[b,i,j]=1 else 0. Supports directed edges by non-symmetric matrices.
                         Assumes to already have added the identity connections (connections to itself).
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        """
        - Here we are doing a sum of all the connection in each dimension of the input matrix
        - Imagine a matrix where each row represents the connections of one node to all the other nodes in the matrix,
        we are literally just seeing how many connections that node has to other nodes in the network, we could also do
        an average, max, convolution operation (?) to get a property that can be learned 
        """
        
        print('Neighbour list before transformation')
        print(adj_matrix)
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        print('Number of neighbours/number of edges linked to each node')
        print(num_neighbours)

        # Apply the linear transformation via the method defined above
        node_feats = self.projection(node_feats)
        print('Node feats after linear transform')
        print(node_feats)

        """
        Perform a batch matrix-matrix multiplication. Will need to transpose a matrix to allow multiplication in this case.

        Perform batch matrix multiplication because to ensure compatibility of dimensionality, which may change depending on the node, some nodes 
        may have more connections than others... in this case because we use a whole adjacency matrix as opposed to adjacency lists, we can get away with 
        matrix multiplication by matmul. 
        """

        # This is effectively providing the transformed node features (based on learned embeddings) that a classification/regression can be performed on
        node_feats = torch.matmul(adj_matrix, node_feats)
        print('Node features after MatMul with adjacency matrix')
        print(node_feats)

        #Divide by number of neighbors for each node to get average number of connections 
        node_feats = node_feats / num_neighbours
        
        print(node_feats)
        #Data returned in output

        return node_feats

node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
adj_matrix = Tensor([[[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]]])

# print("Node features:\n", node_feats)
# print("\nAdjacency matrix:\n", adj_matrix)

# ### Instantiate the graph neural network
# layer = GCNLayer(c_in=2, c_out=2)

# # Set up the weights and bias to be used by the linear layer (would be trainable)
# layer.projection.weight.data = Tensor([[1.0, 0.0], [0.0, 1.0]]) 
# layer.projection.bias.data = Tensor([0.0, 0.0])

# """
# Use of Torch.no_grad():
# - To perform inference without Gradient Calculation.
# - To make sure there's no leak test data into the model.
# """
# with torch.no_grad():
#     out_feats = layer(node_feats, adj_matrix)

# print("Adjacency matrix", adj_matrix)
# print("Input features", node_feats)
# print("Output features", out_feats)

"""
GRAPH ATTENTION NETWORK TUTORIAL 
"""

class GATLayer(nn.Module):
    def __init__(self, c_in, c_out, num_heads=1, concat_heads=True, alpha=0.2):
        """
        Args:
            c_in: Dimensionality of input features
            c_out: Dimensionality of output features
            num_heads: Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
                        - The more attention heads, the more distinct relationships can be learned(?), up to a point.
            concat_heads: If True, the output of the different heads is concatenated instead of averaged.
            alpha: Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        if self.concat_heads:
            assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out = c_out // num_heads

        # Sub-modules and parameters needed in the layer
        self.projection = nn.Linear(c_in, c_out * num_heads)
        self.a = nn.Parameter(Tensor(num_heads, 2 * c_out))  # One per head, outputted attention weight
        self.leakyrelu = nn.LeakyReLU(alpha) #Adding a leaky Relu method 

        # Initialization from the original implementation, initialisation of the beginining paramerters of neural network
        # Using certain initialisation heuristics (e.g. small range of values, sampling from uniform distribution)
        # in initialisation have been found to aid generalisation)

        nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, node_feats, adj_matrix, print_attn_probs=False):
        """
        Args:
            node_feats: Input features of the node. Shape: [batch_size, c_in]
            adj_matrix: Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
            print_attn_probs: If True, the attention weights are printed during the forward pass
                               (for debugging purposes)
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)

        # Apply linear layer and sort nodes by head
        node_feats = self.projection(node_feats)
        node_feats = node_feats.view(batch_size, num_nodes, self.num_heads, -1)

        # We need to calculate the attention logits for every edge in the adjacency matrix
        # Doing this on all possible combinations of nodes is very expensive
        # => Create a tensor of [W*h_i||W*h_j] with i and j being the indices of all edges
        
        # Returns a 2-D tensor where each row is the index for a nonzero value.
        # If input has 'n' dimensions, returned vector is z x n where z is where z
        # is total number of non-zero elements in input tensor

        edges = adj_matrix.nonzero(as_tuple=False)

        node_feats_flat = node_feats.view(batch_size * num_nodes, self.num_heads, -1)

        edge_indices_row = edges[:, 0] * num_nodes + edges[:, 1]
        edge_indices_col = edges[:, 0] * num_nodes + edges[:, 2]
        a_input = torch.cat(
            [
                torch.index_select(input=node_feats_flat, index=edge_indices_row, dim=0),
                torch.index_select(input=node_feats_flat, index=edge_indices_col, dim=0),
            ],
            dim=-1,
        )  # Index select returns a tensor with node_feats_flat being indexed at the desired positions

        # Calculate attention MLP output (independent for each head)
        attn_logits = torch.einsum("bhc,hc->bh", a_input, self.a)
        attn_logits = self.leakyrelu(attn_logits)

        # Map list of attention values back into a matrix
        attn_matrix = attn_logits.new_zeros(adj_matrix.shape + (self.num_heads,)).fill_(-9e15)
        attn_matrix[adj_matrix[..., None].repeat(1, 1, 1, self.num_heads) == 1] = attn_logits.reshape(-1)

        # Weighted average of attention
        attn_probs = F.softmax(attn_matrix, dim=2)
        if print_attn_probs:
            print("Attention probs\n", attn_probs.permute(0, 3, 1, 2))
        node_feats = torch.einsum("bijh,bjhc->bihc", attn_probs, node_feats)

        # If heads should be concatenated, we can do this by reshaping. Otherwise, take mean
        if self.concat_heads:
            node_feats = node_feats.reshape(batch_size, num_nodes, -1)
        else:
            node_feats = node_feats.mean(dim=2)

        return node_feats

layer = GATLayer(2, 2, num_heads=2)
layer.projection.weight.data = Tensor([[1.0, 0.0], [0.0, 1.0]])
layer.projection.bias.data = Tensor([0.0, 0.0])
layer.a.data = Tensor([[-0.2, 0.3], [0.1, -0.1]])

with torch.no_grad():
    out_feats = layer(node_feats, adj_matrix, print_attn_probs=True)

# print("Adjacency matrix", adj_matrix)
# print("Input features", node_feats)
# print("Output features", out_feats)

print('Done')

