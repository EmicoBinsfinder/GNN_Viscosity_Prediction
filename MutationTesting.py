from rdkit import Chem
from rdkit.Chem import Draw
import rdkit
from random import sample
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from random import choice as rnd
import random
from rdkit.Chem.Draw import rdMolDraw2D
from MoleculeDifferenceViewer import view_difference
from copy import deepcopy
from operator import itemgetter
import subprocess
import ast
import pandas as pd
import re
import requests
from math import log10
import os
from os.path import join
import Genetic_Algorithm_Functions as GAF
import sys
from Genetic_Algorithm_Functions import MolCheckandPlot, plotmol, mol_with_atom_index
from Genetic_Algorithm_Functions import CheckSubstruct
from MoleculeDifferenceViewer import view_difference

### Fragments
fragments = ['CCCC', 'CCCCC', 'C(CC)CCC', 'CCC(CCC)CC', 'CCCC(C)CC', 'CCCCCCCCC', 'CCCCCCCC', 'CCCCCC', 'C(CCCCC)C', 'CCCCCc1ccccc1']
fragments = [Chem.MolFromSmiles(x) for x in fragments]

### ATOM NUMBERS
Atoms = ['C', 'O']
AtomMolObjects = [Chem.MolFromSmiles(x) for x in Atoms]
AtomicNumbers = []

# Getting Atomic Numbers for Addable Atoms
for Object in AtomMolObjects:
     for atom in Object.GetAtoms():
          AtomicNumbers.append(atom.GetAtomicNum())         

### BOND TYPES
BondTypes = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]

###### Implementing Genetic Algorithm Using Functions Above
Mutations = ['AddAtom', 'ReplaceAtom', 'ReplaceBond', 'RemoveAtom', 'AddFragment']

PYTHONPATH = 'C:/Users/eeo21/AppData/Local/Programs/Python/Python310/python.exe'
STARTINGDIR = deepcopy(os.getcwd())
os.chdir(STARTINGDIR)

# Load in test datasets
RingDataset = pd.read_csv(join(STARTINGDIR, 'MoleculeDatabaseTest.csv'))
RingDataset = pd.read_csv(join(STARTINGDIR, 'MoleculeDatabaseTestNoRings.csv'))

TestMolecules = ['CCCCCc1ccccc1', 'CCCC(CCCC)CCCC']

TestMoleculesMols = [Chem.MolFromSmiles(x) for x in TestMolecules]
plotmol(TestMoleculesMols[0])

from Genetic_Algorithm_Functions import GeneratePDB
GeneratePDB(SMILES='CCCCCc1ccccc1', PATH=(join(STARTINGDIR, 'Test.pdb')))





# Mut_Mol, Mut_Mol_Sanitized, Mut_Mol_SMILES, StartingMoleculeUnedited = GAF.Mol_Crossover(TestMoleculesMols[0],
#                                                                                     TestMoleculesMols[1], 
#                                                                                     showdiff=False, 
#                                                                                     Verbose=False)

# print(Mut_Mol_SMILES)

# Mut_Mol1, Mut_Mol_Sanitized, Mut_Mol_SMILES, StartingMoleculeUnedited1 = GAF.ReplaceAtom(Mut_Mol, Atoms, fromAromatic=False, showdiff=False)

# print(Mut_Mol_SMILES)

# if Mut_Mol != None:
#     plotmol(Mut_Mol)
#     plotmol(Mut_Mol1)
#     plotmol(StartingMoleculeUnedited)


# os.chdir('F:/PhD/HIGH_THROUGHPUT_STUDIES/MDsimulationEvaluation/ValidationStudies12ACutoff_200mols/')
# STARTDIR = os.getcwd()

# Names = [x for x in os.listdir(os.getcwd()) if os.path.isdir(x)]
# Temps = [313, 373]

# GKVisc100, EinsteinVisc100, EinsteinUncert100, GKUncert100 = GAF.GetVisc(Names[-1], Temps[1], STARTDIR, unit='atm', plot=False)
# GKVisc40, EinsteinVisc40, EinsteinUncert40, GKUncert40 = GAF.GetVisc(Names[-1], Temps[0], STARTDIR, unit='atm', plot=False)

# Dens40 = float(GAF.GetDens(f'{STARTDIR}/{Names[-1]}/Run_1/eqmDensity_{Names[-1]}_T313KP1atm.out'))
# Dens100 = float(GAF.GetDens(f'{STARTDIR}/{Names[-1]}/Run_1/eqmDensity_{Names[-1]}_T373KP1atm.out'))

# KVI = GAF.GetKVI(DVisc40=GKVisc40, DVisc100=GKVisc100, Dens40=Dens40, Dens100=Dens100, STARTINGDIR=STARTINGDIR)
# DVI = GAF.GetDVI(DVisc40=GKVisc40, DVisc100=GKVisc100)

# print(KVI)
# print(DVI)

# Dataset = pd.read_csv('Dataset.csv')
# SMILESList = Dataset['smiles'].to_list()[:49]

# SMILES = 'COCCCOCC=CCC(=O)C=CCCOC(C=Cc1ccccc1)OCOCC=C'

# Scores = GAF.TanimotoSimilarity(SMILES, SMILESList)
# print(Scores)

