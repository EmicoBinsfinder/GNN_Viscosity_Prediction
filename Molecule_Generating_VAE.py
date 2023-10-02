import ast

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import BondType
from rdkit.Chem.Draw import MolsToGridImage

RDLogger.DisableLog("rdApp.*")

working_dir = 'C:/Users/eeo21/VSCodeProjects/GNN_Viscosity_Prediction'

# We can try using the dataset that Ashlie's group made to see if we can start to generate new molecules
# Previous study used only 505 molecules in total to train (transfer learn?) pre-exisiting 

#DOWNLOAD YOUR DATASET

# csv_path = keras.utils.get_file(
#     f"{working_dir}/250k_rndm_zinc_drugs_clean_3.csv",
#     "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv")

df = pd.read_csv("250k_rndm_zinc_drugs_clean_3.csv")

# Removing new line symbols from the dataset with a lambda function (dataset specific)
df["smiles"] = df["smiles"].apply(lambda s: s.replace("\n", ""))

# Defijning elements of molecules present in the dataset (change depending on training dataset being used)

SMILES_CHARSET = "['B', 'C', 'F', 'H', 'I', 'O', 'P', 'S', 'Br', 'Cl']"

# Defining a dictionary with the different bond types that can be represented by a SMILES string
bond_mapping = {"SINGLE":0, "DOUBLE":1, "TRIPLE":2, "AROMATIC":3}

# Updating the bond mappings to rdkit objects for the graph construction using rdkit's Chem module (maybe as a catcher in case it changes?)
bond_mapping.update({0: BondType.SINGLE, 1: BondType.DOUBLE, 2:BondType.TRIPLE, 3:BondType.AROMATIC})

# Extracting list object from list of elements above
SMILES_CHARSET = ast.literal_eval(SMILES_CHARSET)

# Extracting maximum length molecule (in string characters) from dataset
Max_Molecule_Size = max(df["smiles"].str.len())

# Allowing changes between index and SMILES element
SMILES_to_index = dict((c, i) for i, c in enumerate(SMILES_CHARSET))
index_to_SMILES = dict((i, c) for i, c in enumerate(SMILES_CHARSET))

atom_mapping = dict(SMILES_to_index)
atom_mapping.update(index_to_SMILES)

# HYPERPARAMETERS

BATCH_SIZE = 100
EPOCHS = 10

VAE_Learning_Rate = 5e-4
NUM_ATOMS = 120 # Max number of atoms in a single molecule, determines max length of generated atoms

ATOM_TYPES = len(SMILES_CHARSET) # Different number of atom types
BOND_DIM = 5 #Number of bond types (including non-bonds)
LATENT_SPACE_DIMENSIONS = 435 # Need to find out if more or fewer dimensions make a difference

TRAINING_FRAC = 0.9 #Percentage of dataset to be used for training VAE

# FUNCTION TO CONVERT SMILES STRINGS TO GRAPHS

def SMILES_to_graph(SMILES):
    # Converts input SMILES string to rdkit object
    # Can do the same for InCHI, will this result in the same rdkit object?
    Molecule = Chem.MolFromSmiles(SMILES)

    # Initialise adjacency and feature matrices with zeros
    Adjacency_Matrix = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), "float32")
    Feature_Matrix = np.zeros((NUM_ATOMS, ATOM_TYPES), "float32")

    # Iterate over every atom in molecule to populate Adjacency and Feature Matrices
    for atom in Molecule.GetAtoms():
        i = atom.getIdx() # Gets atom's index in the molecule

        # Maps atom type to atom symbols we defined above
        Element = atom_mapping[atom.GetSymbol()] 
        
        Feature_Matrix[i] = np.eye(ATOM_TYPES)[Element] 
        print(Feature_Matrix[i])  ####Checking to see what line above does
        
        # Iterate over one-hop neighbors of atom to characterise its bonds
        for neighbour in atom.GetNeighbors():
            
            # Gets atom index of neighbours to atom in question
            j = neighbour.GetIdx()
            
            # Returns bond type from the RDKit object
            bond = Molecule.GetBondBetweenAtoms(i, j) 
            
            # Get bond type/name from mapping that we created above
            bond_type_idx = bond_mapping[bond.GetBondType().name] 
            
            # Update the Adjacency Matrix with bond entry(matrix is one-hot encoded)
            Adjacency_Matrix[bond_type_idx, [i, j], [j, i]] = 1  
        
        # Dealing with non-bond instances
        Adjacency_Matrix[-1, np.sum(Adjacency_Matrix, axis=0) == 0] = 1

        # Dealing with non-atom instances
        Feature_Matrix[np.where(np.sum(Feature_Matrix, axis=1) == 0)[0], -1] = 1

        return Adjacency_Matrix, Feature_Matrix

def graph_to_molecule(graph):
    # Unpack the graph into feature and adjacency matrices
    Adjacency_Matrix, Feature_Matrix = graph

    Molecule = Chem.RWMol()

    # Removing 'non-atoms' and atoms with no bonds (remove disconnected fragments)
    # Doing this using np.where which choose from list of variables depending on some conditions

    Keep_Index = np.where(np.argmax(Feature_Matrix, axis = 1) != ATOM_TYPES) & (np.sum(Adjacency_Matrix[:-1], axis = (0, 1)) !=0)[0]

    print(f'Feature_Matrix before non atom removal: \n {Feature_Matrix} \n')

    Feature_Matrix = Feature_Matrix[Keep_Index]
    
    print(f'Feature_Matrix after non atom removal: \n {Feature_Matrix} \n')

    print(f'Adjacency_Matrix before non atom removal: \n {Feature_Matrix} \n')
    
    Adjacency_Matrix = Adjacency_Matrix[:, Keep_Index, :][:, :, Keep_Index]

    print(f'Adjacency_Matrix after non atom removal: \n {Feature_Matrix} \n')
    
    # Adding Atoms to Molecule

    for Atom_Type_Index in np.argmax(Feature_Matrix, axis=1):
        Atom = Chem.Atom(atom_mapping[Atom_Type_Index])
        _ = Molecule.AddAtom(Atom)

        # Add bonds between the atoms in the Molecule (from upper triangle of Matrix (#TODO why tho?)) 
        (Bonds_ij, Atoms_i, Atoms_j) = np.where(np.triu(Adjacency_Matrix) == 1)

        for(bond_ij, atom_i, atom_j) in zip(Bonds_ij, Atoms_i, Atoms_j):

            # Checking for aromatic bonds
            if atom_i == atom_j or bond_ij == BOND_DIM - 1:
                continue
            Bond_Type = bond_mapping[bond_ij]
            Molecule.AddBond(int(atom_i), int(atom_j), Bond_Type)

            # Sanitize the molecule
            # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization
            flag = Chem.SanitizeMol(Molecule, catchErrors=True)

            # Return None if sanitization fails 
            if flag != Chem.SanitizeFlags.SANITIZE_NONE:
                return None

            return Molecule
    
# MAKING THE TRAINING DATASET

train_df = df.sample(frac = TRAINING_FRAC, random_state = 42)
train_df.reset_index(drop = True, inplace = True)

Adjacency_Tensors, Feature_Tensors, QED_Tensors = [], [], []

# Assuming CSV format 
for Index in range(8000):
    
    # Retrieve SMILES string from dataset and converting it to graph format
    Adjacency_Tensor, Feature_Tensor = SMILES_to_graph(train_df.loc[Index]['smiles'])

    # Retrieving target property tensor from dataset
    QED_Tensor = train_df.loc[Index]['qed']
    
    Adjacency_Tensors.append(Adjacency_Tensor)
    Feature_Tensors.append(Feature_Tensor)
    QED_Tensors.append(QED_Tensor)

Adjacency_Tensors = np.array(Adjacency_Tensors)
Feature_Tensors = np.array(Feature_Tensors)
QED_Tensors = np.array(QED_Tensors)

# DEFINING THE MODEL ARCHITECTURE

# Define Graph Convolutional Layer

# Hidden units in Graph Convolutional Layers
GraphConvolutionLayer_HiddenUnits = 128 

# Hidden units dropout rate
DropoutRate = 0.2

class GraphConvolutionLayer(keras.layers.Layer):
    def __init__(
            self,
            units = GraphConvolutionLayer_HiddenUnits,
            activation = 'relu',
            use_bias = False,
            kernel_initializer = "glorot_uniform", # Define initialisation of layer weights
            bias_initializer = 'zeros',
            kernel_regularizer = None, # What are these regularisers?
            bias_regularizer = None,
            **kwargs
    ):
        super.__init__(**kwargs)
        
        self.units = units,
        self.activation = keras.activations.get(activation),
        self.use_bias = use_bias,
        self.kernel_initializer = keras.initializers.get(kernel_initializer),
        self.bias_initializer = keras.initializers.get(bias_initializer),
        self.kernal_regularizer = keras.initializers.get(kernel_regularizer),
        self.bias_regularizer = keras.initializers.get(bias_regularizer)

    def build(self, input_shape):
        bond_dim = input_shape[0][1]
        atom_dim = input_shape[1][2]

        # Define trainabliity of weights in layers
        self.kernel = self.add_weight(
            shape = (bond_dim, atom_dim, self.units),
            initializer = self.kernel_initializer,
            regularizer = self.kernal_regularizer,
            trainiable = True,
            name = 'Weights',
            dtype = tf.float32
        )

        # Define trainability of bias parameters
        if self.use_bias:
            self.bias = self.add_weight(
                shape = (bond_dim, 1, self.units),
                initializer = self.bias_initializer,
                regularizer = self.bias_regularizer,
                trainable = True,
                name = 'Bias',
                dtype = tf.float32
            )
        
        self.built = True

    # Define method for inference
    def call(self, inputs, training=False):
        Adjacency_Matrix, Feature_Matrix = inputs

        # Get information from neighbors wiht a matrix multiplication (a new matrix representing the type of atoms next to a specified atom)
        x = tf.matmul(Adjacency_Matrix, Feature_Matrix[:, None, :, :])

        # Apply a linear transformation according to the layer and associated activation function
        x = tf.matmul(x, self.kernel)

        # Apply bias 
        if self.use_bias:
            x += self.bias

        # Reduce bond types dimensions

        x_reduced = tf.reduce_sum(x, axis=1)

        # Apply the activation function
        return self.activation(x_reduced)
    
# DEFINING THE ENCODER AND DECODER LAYERS

def get_encoder(
        gconv_units, latent_dim, adjacency_shape, feature_shape, dense_units, dropout_rate):
    adjacency = keras.layers.Input(shape= adjacency_shape)
    features = keras.layers.Input(shape= feature_shape)

    features_transformed = features
    for units in gconv_units:
        features_transformed = GraphConvolutionLayer(units)([adjacency, features_transformed])

    # Reduce molecular representation to 1D
    x = keras.layers.GlobalAveragePooling1D()(features_transformed)

    # Forward pass
    for units in dense_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

    z_mean = layers.Dense(latent_dim, dtype= 'flaot32', name= "z_mean")(x)
    log_var = layers.Dense(latent_dim, dtype= 'float32', name= "log_var")(x)

    encoder = keras.Model([adjacency, features], [z_mean, log_var], name= "encoder")

    return encoder

def get_decoder(dense_units, dropout_rate, latent_dim, adjacency_shape, feature_shape):
    latent_inputs = keras.Input(shape= (latent_dim,))

    x = latent_inputs
    for units in dense_units:
        x = keras.layers.Dense(units, activation= "tanh")(x)
        x = keras.layers.Dropout(dropout_rate)(x)

    # Map outputs of previous layer to [continuous adjacency tensors]
    x_adjacency = keras.layers.Dense(tf.math.reduce_prod(adjacency_shape))(x)
    x_adjacency = keras.layers.Reshape(adjacency_shape)(x_adjacency)

    # Make tensors symmetrical in the last two dimensions
    x_adjacency = (x_adjacency + tf.transpose(x_adjacency, (0, 1, 3, 2))) / 2

    #
    x_adjacency = keras.layers.Softmax(axis=2)(x_features)


#Can we constrain VAE to target generating molecules with desirable properties?

