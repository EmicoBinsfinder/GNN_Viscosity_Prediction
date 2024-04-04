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
fragments = ['CCCC', 'CCCCC', 'C(CC)CCC', 'CCC(CCC)CC', 'CCCC(C)CC', 'CCCCCCCCC', 'CCCCCCCC', 'CCCCCC', 'C(CCCCC)C', 'c1ccccc1']
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

TestMolecules = ['CCCCCO', 'CCCCCCCCCCCCCCOCC=O', 'COCCC=CCCCC=COC(C)C', 'C=COCCC(=O)CCCCCCCOCCCC(=C)C=CCCCCOCCCC', 'CC=COCCC=CC=CCCCCCCCCCCCCC(C)CO',
                 'C=C(C)C=Cc1cccc(C)c1', 'CCc1ccccc1C=C(C)OC', 'OCCc1cccc(CCCCCC(O)O)c1', 'Cc1ccccc1OC(C)c1ccc(CO)cc1', 'C=Cc1cccc(CC(O)c2cccc(CCCOC=CCCOCO)c2)c1']

TestMoleculesMols = [Chem.MolFromSmiles(x) for x in TestMolecules]

def Mol_Crossover(StartingMolecule, CrossMolList, showdiff=False, Verbose=False):
    try:
        """
        Take in two molecules, the molecule to be mutated, and a list of molecules to crossover with?

        Randomly fragment each molecule 

        Create bond between randomly selected atoms on the molecule

        Need to make sure that we are not bonding an aromatic to an aromatic

        Write code to save which molecule was crossed over with
        """
        StartingMolecule = rnd(TestMoleculesMols)
        StartingMoleculeUnedited = deepcopy(StartingMolecule)
        CrossMolecule = rnd(CrossMolList)
        StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)
        CrossMolecule = Chem.RWMol(CrossMolecule)

        # Need to check and remove atom indexes where the atom is bonded to an atom that is aromatic
        #StartMol
        StartMolRI = StartingMolecule.GetRingInfo()
        StartMolAromaticBonds = StartMolRI.BondRings()
        StartMolAromaticBondsList = []
        for tup in StartMolAromaticBonds:
            for bond in tup:
                StartMolAromaticBondsList.append(int(bond))

        StartMolBondIdxs = [int(x.GetIdx()) for x in StartingMolecule.GetBonds()]

        StartMolBondIdxsFinal = [x for x in StartMolBondIdxs if x not in StartMolAromaticBondsList]
        StartMolSelectedBond = StartingMolecule.GetBondWithIdx(rnd(StartMolBondIdxsFinal))

        StartingMolecule.RemoveBond(StartMolSelectedBond.GetBeginAtomIdx(), StartMolSelectedBond.GetEndAtomIdx())
        StartMolFrags = Chem.GetMolFrags(StartingMolecule, asMols=True)
        StartingMolecule = max(StartMolFrags, default=StartingMolecule, key=lambda m: m.GetNumAtoms())

        #CrossMol
        CrossMolRI = CrossMolecule.GetRingInfo()
        CrossMolAromaticBonds = CrossMolRI.BondRings()
        CrossMolAromaticBondsList = []
        for tup in CrossMolAromaticBonds:
            for bond in tup:
                CrossMolAromaticBondsList.append(int(bond))

        CrossMolBondIdxs = [int(x.GetIdx()) for x in CrossMolecule.GetBonds()]

        CrossMolBondIdxsFinal = [x for x in CrossMolBondIdxs if x not in CrossMolAromaticBondsList]
        CrossMolSelectedBond = CrossMolecule.GetBondWithIdx(rnd(CrossMolBondIdxsFinal))

        CrossMolecule.RemoveBond(CrossMolSelectedBond.GetBeginAtomIdx(), CrossMolSelectedBond.GetEndAtomIdx())
        CrossMolFrags = Chem.GetMolFrags(CrossMolecule, asMols=True)
        CrossMolecule = max(CrossMolFrags, default=CrossMolecule, key=lambda m: m.GetNumAtoms())

        InsertStyle = rnd(['Within', 'Egde'])

        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited = GAF.AddFragment(StartingMolecule, 
                                                                                            CrossMolecule, 
                                                                                            InsertStyle, 
                                                                                            showdiff, 
                                                                                            Verbose)
    
    except:
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited = None, None, None, None

    return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited


for x in TestMoleculesMols:
    print(x)
    Mut_Mol, Mut_Mol_Sanitized, Mut_Mol_SMILES, StartingMoleculeUnedited = Mol_Crossover(x,
                                                                                         TestMoleculesMols, 
                                                                                         showdiff=False, 
                                                                                         Verbose=False)
    if Mut_Mol != None:
        plotmol(Mut_Mol)
        plotmol(StartingMoleculeUnedited)


