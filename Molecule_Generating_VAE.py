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
NUM_ATOMS = 120 # Max number of atoms in a single molecule

ATOM_TYPES = len(SMILES_CHARSET) # Different number of atom types
BOND_DIM = 5
LATENT_SPACE_DIMENSIONS = 435 # Need to find out if more or fewer dimensions make a difference

# FUNCTION TO CONVERT SMILES STRINGS TO GRAPHS

def SMILES_to_graph(SMILES):
    # Converts input SMILES string to rdkit object
    # Can do the same for InCHI, will this result in the same rdkit object?

    Molecule = Chem.MolFromSmiles(SMILES)

    # Initialise adjacency and feature tensors
    Adjacency_Matrix = np.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), "float32")
    Feature_Matrix = np.zeros((NUM_ATOMS, ATOM_TYPES), "float32")





#Can we constrain VAE to target generating molecules with desirable properties?

