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
from Genetic_Algorithm_Functions import MolCheckandPlot
from Genetic_Algorithm_Functions import CheckSubstruct

### Fragments
fragments = ['CCCC', 'CCCCC', 'C(CC)CCC', 'CCC(CCC)CC', 'CCCC(C)CC', 'CCCCCCCCC', 'CCCCCCCC', 'CCCCCC', 'C(CCCCC)C', 'c1ccccc1']
fragments = [Chem.MolFromSmiles(x) for x in fragments]

def RemoveFragment(StartingMolecule, BondTypes, showdiff=False, Verbose=False):
    """
    Steps to implement replace fragment function
    Take in starting molecule
    Perform random fragmentation of molecule by performing x random cuts
        Will need to keep hold of terminal ends of each molecule
        Will need to check fragment being removed does not completely mess up molecule
    Stitch remaning molecules back together by their terminal ends
    Will allow x attempts for this to happen before giving up on this mutation
    Only implement this mutation when number of atoms exceeds a certain number e.g. 5/6 max mol length
    Max fragment removal length of 2/5 max mol length
    """

    StartingMoleculeUnedited = deepcopy(StartingMolecule)
    
    # Change Mol onject
    StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)

    # Get list of Bond Indexes


    # Select two bond indexes to perform fragmentation at (to create three total fragments)

    # Randomly create tuple of fragment molecule
    FragmentedMol = Chem.FragmentOnSomeBonds()

    # If fragment removal successful, save the three generated fragments to fragement database

    # return Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES, StartingMoleculeUnedited

def ReplaceFragment():
    """
    Steps to implement replace fragment function
    Take in starting molecule
    Perform random fragmentation of molecule by performing x random cuts
        Will need to keep hold of terminal ends of each molecule
        Will need to check fragment being replaced does not completely mess up molecule
        Try X attempt to stitch together replaced fragments and remaining fragments
        Select fragment from list of fragments
    """

def MolCrossover():
     """
     How to know which fragments to use in crossover
     """

#AromaticMolecules = 'c1ccccc1'

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

# GENETIC ALGORITHM HYPERPARAMETERS
Silent = False # Edit outputs to only print if this flag is False
NumElite = 15
IDcounter = 0
FirstGenerationAttempts = 0
GenerationMolecules = []
FirstGenSimList = []
MaxNumHeavyAtoms = 50
MinNumHeavyAtoms = 5
showdiff = False # Whether or not to display illustration of each mutation
GenerationSize = 50
LOPLS = True # Whether or not to use OPLS or LOPLS, False uses OPLS
MaxGenerations = 100
NumGenerations = 1
MaxMutationAttempts = 200
Fails = 0
NumMols = 50
Agent = 'Agent1'

PYTHONPATH = 'C:/Users/eeo21/AppData/Local/Programs/Python/Python310/python.exe'
STARTINGDIR = deepcopy(os.getcwd())
os.chdir(STARTINGDIR)

# Load in test datasets
RingDataset = pd.read_csv(join(STARTINGDIR, 'MoleculeDatabaseTest.csv'))
RingDataset = pd.read_csv(join(STARTINGDIR, 'MoleculeDatabaseTestNoRings.csv'))

def RemoveFragment(StartingMolecule, BondTypes, MaxIters= 20, showdiff=False, Verbose=False):
    """
    StartingMolecule: Mol


    Steps to implement replace fragment function
    Take in starting molecule
    Perform random fragmentation of molecule by performing x random cuts
        Will need to keep hold of terminal ends of each molecule
        Will need to check fragment being removed does not completely mess up molecule
    Stitch remaning molecules back together by their terminal ends
    Will allow x attempts for this to happen before giving up on this mutation
    Only implement this mutation when number of atoms exceeds a certain number e.g. 5/6 max mol length
    Max fragment removal length of 2/5 max mol length
    """

    StartingMoleculeUnedited = deepcopy(StartingMolecule)
    
    # Change Mol onject
    StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)

    # Get list of Bond Indexes


    # Select two bond indexes to perform fragmentation at (to create three total fragments)

    # Randomly create tuple of fragment molecule
    FragmentedMol = Chem.FragmentOnSomeBonds()

    # If fragment removal successful, save the three generated fragments to fragement database

    # return Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES, StartingMoleculeUnedited


