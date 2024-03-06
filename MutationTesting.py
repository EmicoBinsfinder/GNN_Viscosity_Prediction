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
from MoleculeDifferenceViewer import view_difference

### Fragments
fragments = ['CCCC', 'CCCCC', 'C(CC)CCC', 'CCC(CCC)CC', 'CCCC(C)CC', 'CCCCCCCCC', 'CCCCCCCC', 'CCCCCC', 'C(CCCCC)C', 'c1ccccc1']
fragments = [Chem.MolFromSmiles(x) for x in fragments]

### ATOM NUMBERS
Atoms = ['C', 'O']
AtomMolObjects = [Chem.MolFromSmiles(x) for x in Atoms]
AtomicNumbers = []

def plotmol(mol):
    img = Draw.MolsToGridImage([mol], subImgSize=(800, 800))
    img.show()

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

AppendFragments = []
TestMolecules = ['CCCCCO', 'CCCCCCCCCCCCCCOCC=O', 'COCCC=CCCCC=COC(C)C', 'C=COCCC(=O)CCCCCCCOCCCC(=C)C=CCCCCOCCCC', 'CC=COCCC=CC=CCCCCCCCCCCCCC(C)CO',
                 'C=C(C)C=Cc1cccc(C)c1', 'CCc1ccccc1C=C(C)OC', 'OCCc1cccc(CCCCCC(O)O)c1', 'Cc1ccccc1OC(C)c1ccc(CO)cc1', 'C=Cc1cccc(CC(O)c2cccc(CCCOC=CCCOCO)c2)c1']

TestMoleculesMols = [Chem.MolFromSmiles(x) for x in TestMolecules]

def RemoveFragment(StartingMolecule, BondTypes, MaxIters= 20, showdiff=False, Verbose=False):
    """
    StartingMolecule: Mol

    Steps to implement replace fragment function

    Take in starting molecule
    
    Check which molecules are 

    Perform random fragmentation of molecule by performing x random cuts
        - Will need to keep hold of terminal ends of each molecule
            * Do this by checking fragment for atoms with only one bonded atom 
        - Will need to check fragment being removed does not completely mess up molecule
    
    Stitch remaning molecules back together by their terminal ends
    
    Will allow x attempts for this to happen before giving up on this mutation
    
    Only implement this mutation when number of atoms exceeds a certain number e.g. 5/6 max mol length
    
    Max fragment removal length of 2/5 max mol length
    """

Verbose = False

StartingMoleculeUnedited = deepcopy(TestMoleculesMols[3])

# Change Mol onject
StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)

# Get list of Bond Indexes
AtomIdxs = []

# Check if there are aromatic rings (RI = Ring Info object)
RI = StartingMolecule.GetRingInfo()
AromaticAtomsObject = StartingMolecule.GetAromaticAtoms()
AromaticAtoms = []
for x in AromaticAtomsObject:
    AromaticAtoms.append(x.GetIdx())

# Choose whether to remove aromatic ring or not
RemoveRing =  rnd([False, False])

if RemoveRing and len(AromaticAtoms) > 0:
    try:
        ChosenRing = rnd(RI.AtomRings())
        """
        Once ring is chosen:
        - Go through each atom in chosen aromatic ring
        - Check the bonds of each atom to see if aromatic or not
        - If bond is not aromatic check which atom in the bond was not aromatic
            *Save the atom and bond index of the non aromatic atom/bond
        - If only one non-aromatic bond, just sever bond and return molecule
        - If exactly two non-aromatic bonds, sever both then create bond between the terminal atoms
        - If more than two non-aromatic atoms
            *Select two of the non-aromatic atoms and create a bond between them, if valence violated, discard attempt 
        """
        BondIdxs = []
        AtomIdxs = []

        for AtomIndex in ChosenRing:
            Atom = StartingMolecule.GetAtomWithIdx(AtomIndex) #Get indexes of atoms in chosen ring
            for Bond in Atom.GetBonds():
                if Bond.IsInRing() == False:
                    BondAtoms = [Bond.GetBeginAtom(), Bond.GetEndAtom()] #Get atoms associated to non-ring bond
                    BondIdxs.append(BondAtoms)
                    for At in BondAtoms:
                        if At.GetIsAromatic() == False:
                            AtomIdxs.append(At.GetIdx())
        """
        To remove fragment:
        Sever the selected bonds from above
        Create single/double bond between two of the non-aromatic atoms in the AtomIdxs
        """

        for B in BondIdxs:
            StartingMolecule.RemoveBond(B[0].GetIdx(), B[1].GetIdx())

        if len(AtomIdxs) > 1:
            BondingAtoms = [AtomIdxs[0], AtomIdxs[1]]
            StartingMolecule.AddBond(BondingAtoms[0], BondingAtoms[1], rnd(BondTypes))

            #Return Largest fragment as final mutated molecule
            mol_frags = Chem.GetMolFrags(StartingMolecule, asMols=True)
            StartingMolecule = max(mol_frags, default=StartingMolecule, key=lambda m: m.GetNumAtoms())
    except:
        if Verbose:
            print('Add atom mutation failed, returning empty objects')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
        else:
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

else:
    StartingMoleculeAtoms = StartingMolecule.GetAtoms()
    AtomIdxs = [x.GetIdx() for x in StartingMoleculeAtoms if x.GetIdx() not in AromaticAtoms]
    # Need to check and remove atom indexes where the atom is bonded to an atom that is aromatic

    UnwantedAtomIdxs = []
    for AtIdx in AtomIdxs:
        Check_Atom = StartingMolecule.GetAtomWithIdx(AtIdx)
        Neighbors = Check_Atom.GetNeighbors()
        for Neighbor in Neighbors:
            if Neighbor.IsInRing() == True or len(Neighbors) <= 1:
                UnwantedAtomIdxs.append(AtIdx)

    # Save indexes of atoms that are neither aromatic nor bonded to an aromatic atom
    FinalAtomIdxs = [x for x in AtomIdxs if x not in UnwantedAtomIdxs]

    # Select two random atoms for fragmentation
    selected_atoms = random.sample(FinalAtomIdxs, 2)

    # Get bonds of selected atoms
    SeveringBonds = []
    ComboBonds = []
    for atomidx in selected_atoms:
        atom = StartingMolecule.GetAtomWithIdx(atomidx)
        BondIdxs = [x.GetIdx() for x in atom.GetBonds()]
        SeveringBonds.append(random.sample(FinalAtomIdxs, 1))
        # Save index of atom on other side of chosen bond that was severed 

    SeveringBonds = [x[0] for x in SeveringBonds]

    #with StartingMolecule as StartingMolecule:
    for b_idx in SeveringBonds:
        b = StartingMolecule.GetBondWithIdx(b_idx)
        StartingMolecule.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())

    frags = Chem.GetMolFrags(StartingMolecule)

    # Only proceed if the removed fragment is less than a quarter the length of the molecule 
    if len(frags)==3 and len(frags[1])<= len(StartingMoleculeUnedited.GetAtoms())*0.4:
        Mol1 = frags[0]
        Mol2 = frags[-1]

        # Get rid of atoms in mol fragments that are aromatic or bonded to an aromatic 
        #Need to get highest atom index in molecule that isn't in an aromatic ring

        Mol1 = [x for x in Mol1 if x in FinalAtomIdxs]
        Mol2 = [x for x in Mol2 if x in FinalAtomIdxs]

        StartingMolecule = Chem.RWMol(StartingMolecule)
        StartingMolecule.AddBond(Mol1[-1], Mol2[0], rnd(BondTypes))

        mol_frags = Chem.GetMolFrags(StartingMolecule, asMols=True)

        #Return the largest fragment as the final molecule
        plotmol(StartingMolecule)
        StartingMolecule = max(mol_frags, default=StartingMolecule, key=lambda m: m.GetNumAtoms())

        MutMolSMILES = Chem.MolToSmiles(StartingMolecule)

    else:
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

# Add indexes to molecule for visualisation
def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

mol0 = mol_with_atom_index(StartingMoleculeUnedited)
mol1 = mol_with_atom_index(StartingMolecule)

def mol_crossover():
    pass







