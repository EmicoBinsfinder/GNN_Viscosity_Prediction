from rdkit import Chem
from rdkit.Chem import Draw
import rdkit
from random import sample
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from random import choice as rnd
import random
from MoleculeDifferenceViewer import view_difference
from copy import deepcopy
import subprocess
import ast
import pandas as pd
import re
from math import log10
import os
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
from scipy.constants import Boltzmann
from scipy.optimize import curve_fit
import sys, argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import integrate
from tqdm import trange
from os.path import join as join
from os import chdir
from os import getcwd
from os import listdir
import os

def MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff, Verbose=False):
    
    Mut_Mol = StartingMolecule.GetMol()
    MutMolSMILES = Chem.MolToSmiles(Mut_Mol)

    Mut_Mol_Sanitized = Chem.SanitizeMol(Mut_Mol, catchErrors=True) 

    if len(Chem.GetMolFrags(Mut_Mol)) != 1:
        if Verbose:
            print('Fragmented result, trying new mutation')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

        if showdiff:
            view_difference(StartingMoleculeUnedited, Mut_Mol)
    
    elif Mut_Mol_Sanitized != rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
        if Verbose:
            print('Validity Check Failed')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

    else: 
        if Verbose:
            print('Validity Check Passed')
        if showdiff:
            view_difference(StartingMoleculeUnedited, Mut_Mol)
    
    return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES

def AddAtom(StartingMolecule, NewAtoms, BondTypes, showdiff=False, Verbose=False):
    """
    Function that adds atom from a list of selected atoms to a starting molecule.

    Need to ensure that probability of this is zero if length of molecule is too short.

    Takes molecule, adds atom based on defined probabilities of position.

    Arguments:
        - StartingMolecule: SMILES String of Starting Molecule
        - NewAtoms: List of atoms that could be added to starting molecule
        - Show Difference: If True, shows illustration of changes to molecule
    """
    StartingMoleculeUnedited = deepcopy(StartingMolecule)

    try:
        # Change to an editable object
        StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)

        # Add selected atom from list of addable atoms 
        StartingMolecule.AddAtom(Chem.Atom(int(rnd(NewAtoms))))

        # Storing indexes of newly added atom and atoms from intial molecule
        frags = Chem.GetMolFrags(StartingMolecule)

        # Check which object is the newly added atom
        for ind, frag in enumerate(frags):
            if len(frag) == 1:
                #Store index of new atom in object
                NewAtomIdx = frags[ind]
            else:
                StartMolIdxs = frags[ind]
        
        StartingMolecule.AddBond(rnd(StartMolIdxs), NewAtomIdx[0], rnd(BondTypes))

        #Sanitize molecule
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)
    
    except:
        if Verbose:
            print('Add atom mutation failed, returning empty objects')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
        else:
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

    return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited

def ReplaceAtom(StartingMolecule, NewAtoms, fromAromatic=False, showdiff=False, Verbose=False):
    """
    Function to replace atom from a selected list of atoms from a starting molecule.
    
    Args:
    - StartingMolecule: SMILES String of Starting Molecule
    - NewAtoms: List of atoms that could be added to starting molecule
    - FromAromatic: If True, will replace atoms from aromatic rings 
    """
    """
    Steps:
    1. Get the indexes of all the bonds in the molecule 
    2. Check if there are any bonds that are aromatic 
    3. Select index of atom that will be replaced
    4. Get bond and bondtype of atom to be replaced
    5. Create new bond of same type where atom was removed
    """
    StartingMoleculeUnedited = deepcopy(StartingMolecule)

    AtomIdxs = []

    for atom in StartingMoleculeUnedited.GetAtoms():
        if fromAromatic == False:
            # Check if atom is Aromatic
            if atom.GetIsAromatic():
                continue
            else:
                AtomIdxs.append(atom.GetIdx())
        else:
            AtomIdxs.append(atom.GetIdx())
            if Verbose:
                print(f'Number of Bonds atom has = {len(atom.GetBonds())}')

    #Select atom to be deleted from list of atom indexes, check that this list is greater than 0
    if len(AtomIdxs) == 0:
        if Verbose:
            print('Empty Atom Index List')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
    
    else:
        #Select a random atom from the index of potential replacement atoms
        ReplaceAtomIdx = rnd(AtomIdxs)
        #Exclude replaced atom type from list of atoms to do replacing with
        ReplaceAtomType = StartingMoleculeUnedited.GetAtomWithIdx(ReplaceAtomIdx).GetSymbol()
        AtomReplacements = [x for x in NewAtoms if x != ReplaceAtomType]

        StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)

        #Replace atom
        ReplacementAtom = rnd(AtomReplacements)
        StartingMolecule.ReplaceAtom(ReplaceAtomIdx, Chem.Atom(ReplacementAtom), preserveProps=True)

        if Verbose:
            print(f'{StartingMoleculeUnedited.GetAtomWithIdx(ReplaceAtomIdx).GetSymbol()}\
            replaced with {Chem.Atom(ReplacementAtom).GetSymbol()}')
        
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES  = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

    return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited

def ReplaceBond(StartingMolecule, Bonds, showdiff=True, Verbose=False):
    """
    Function to replace bond type with a different bond type from a selected list of bonds within a starting molecule.
    
    Args:
    - StartingMolecule: SMILES String of Starting Molecule
    - Bonds: List of bonds that could be used to replace bond in starting molecule
    """
    """
    Steps:
    1. Get the indexes of all the bonds in the molecule 
    2. Check if there are any bonds that are aromatic 
    3. Select index of bond that will be replaced
    4. Get index and bondtype of bond to be replaced
    """

    StartingMoleculeUnedited = deepcopy(StartingMolecule)

    BondIdxs = []
    for bond in StartingMoleculeUnedited.GetBonds():
            # Check if atom is Aromatic
            if bond.GetIsAromatic():
                continue
            else:
                BondIdxs.append(bond.GetIdx())

    #Select atom to be deleted from list of atom indexes, check that this list is greater than 0

    #Random selection of bond to be replaced
    if len(BondIdxs) > 0:
        ReplaceBondIdx = rnd(BondIdxs)

        #Excluding selected bond's bond order from list of potential new bond orders
        ReplaceBondType = StartingMoleculeUnedited.GetBondWithIdx(ReplaceBondIdx).GetBondType()
        BondReplacements = [x for x in Bonds if x != ReplaceBondType]

        StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)

        #Get atoms that selected bond is bonded to 
        ReplacedBond = StartingMolecule.GetBondWithIdx(ReplaceBondIdx)
        Atom1 = ReplacedBond.GetBeginAtomIdx()
        Atom2 = ReplacedBond.GetEndAtomIdx()

        #Replace bond, randomly selecting new bond order from list of possible bond orders
        StartingMolecule.RemoveBond(Atom1, Atom2)
        StartingMolecule.AddBond(Atom1, Atom2, rnd(BondReplacements))

        Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)
        
    else:
        if Verbose:
            print('Empty Bond Index List')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

    return Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES, StartingMoleculeUnedited

def AddFragment(StartingMolecule, Fragment, InsertStyle, showdiff=True, Verbose=False):
    """
    Function to add a fragment from a selected list of fragments to starting molecule.
    
    Args:
    - StartingMolecule: SMILES String of Starting Molecule
    - Fragments: List of fragments 
    - showdiff
    - InsertStyle: Whether to add fragment to edge or within molecule
    """
    """
    Steps:
    1. Select fragment to be added to starting molecule 
    2. Determine whether fragment is going to be inserted within or appended to end of molecule
    
    3. If inserting fragment within molecule:
        - Combine starting molecule and fragment into single disconnected Mol object 
        - Split atom indexes of starting molecule and fragment and save in different objects
        - Check number of bonds each atom has in starting molecule and fragment, exclude any atoms that don't have 
        exactly two bonds
        - Get the atom neighbors of selected atom 
        - Remove one of selected atom's bonds with its neighbors, select which randomly but store which bond is severed 
        - Calculate terminal atoms of fragment (atoms with only one bond, will be at least two, return total number of
        fragments, store terminal atom indexes in a list)
        - Randomly select two terminal atoms from terminal atoms list of fragment
        - Create a new bond between each of the neighbors of the target atom and each of the terminal atoms on the 
        fragment, specify which type of bond, with highest likelihood that it is a single bond
        """
    
    StartingMoleculeUnedited = StartingMolecule

    try:
        #Always check if fragment is a cyclic (benzene) molecule
        if len(Fragment.GetAromaticAtoms()) == len(Fragment.GetAtoms()):
            if Verbose:
                print('Fragment aromatic, inappropriate function')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

        elif len((StartingMoleculeUnedited.GetAromaticAtoms())) == len(StartingMoleculeUnedited.GetAtoms()):
            if Verbose:
                print('Starting molecule is completely aromatic')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
        else:
            # Add fragment to Mol object of starting molecule
            StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)
            StartingMolecule.InsertMol(Fragment)

            # Get indexes of starting molecule and fragment, store them in separate objects 
            frags = Chem.GetMolFrags(StartingMolecule)
            StartMolIdxs = frags[0]
            FragIdxs = frags[1]

            OneBondAtomsMolecule = []
            TwoBondAtomsMolecule = []
            AromaticAtomsMolecule = []
            OneBondAtomsFragment = []
            TwoBondAtomsFragment = []

            # Getting atoms in starting molecule with different amount of bonds, storing indexes in list
            for index in StartMolIdxs:
                Atom = StartingMolecule.GetAtomWithIdx(int(index))
                if Atom.GetIsAromatic() and len(Atom.GetBonds()) == 2:
                    AromaticAtomsMolecule.append(index)
                elif len(Atom.GetBonds()) == 2:
                    TwoBondAtomsMolecule.append(index)
                elif len(Atom.GetBonds()) == 1:
                    OneBondAtomsMolecule.append(index)
                else:
                    continue

            # Getting atoms in fragment with varying bonds, storing indexes in listv
            for index in FragIdxs:
                Atom = StartingMolecule.GetAtomWithIdx(int(index))
                if len(Atom.GetBonds()) == 1:
                    OneBondAtomsFragment.append(index)
                elif len(Atom.GetBonds()) == 2:
                    TwoBondAtomsFragment.append(index)

            ### INSERTING FRAGMENT AT RANDOM POINT WITHIN STARTING MOLECULE
            if InsertStyle == 'Within':

                if len(OneBondAtomsMolecule) == 0:
                    if Verbose:
                        print('No one-bonded terminal atoms in starting molecule, returning empty object')
                    Mut_Mol = None
                    Mut_Mol_Sanitized = None
                    MutMolSMILES = None
                
                else:
                    # Select random two bonded atom, not including aromatic
                    AtomRmv = rnd(TwoBondAtomsMolecule)

                    # Get atom neighbor indexes, remove bonds between selected atom and neighbors 
                    neighbors = [x.GetIdx() for x in StartingMolecule.GetAtomWithIdx(AtomRmv).GetNeighbors()]

                    # Randomly choose which bond of target atom to sever
                    SeverIdx = rnd([0,1])

                    StartingMolecule.RemoveBond(neighbors[SeverIdx], AtomRmv)

                    #Create a bond between the fragment and the now segmented fragments of the starting molecule
                    #For situation where bond before target atom is severed
                    if SeverIdx == 0:

                        StartingMolecule.AddBond(OneBondAtomsFragment[0], AtomRmv - 1, Chem.BondType.SINGLE)
                        StartingMolecule.AddBond(OneBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

                        Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

                    #For situation where bond after target atom is severed
                    elif SeverIdx != 0:
                        StartingMolecule.AddBond(OneBondAtomsFragment[0], AtomRmv + 1, Chem.BondType.SINGLE) 
                        StartingMolecule.AddBond(OneBondAtomsFragment[1], AtomRmv, Chem.BondType.SINGLE)

                        Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

            ### INSERTING FRAGMENT AT END OF STARTING MOLECULE

            elif InsertStyle == 'Edge':
                # Choose whether fragment will be added to an aromatic or non-aromatic bond
                FragAdd = rnd(['Aromatic', 'Non-Aromatic'])

                if len(OneBondAtomsMolecule) == 0:
                    if Verbose:
                        print('No one-bonded terminal atoms in starting molecule, returning empty object')
                    Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

                elif FragAdd == 'Non-Aromatic' or len(AromaticAtomsMolecule) == 0: 
                    #Randomly choose atom from fragment (including aromatic)
                    FragAtom = rnd(FragIdxs)

                    #Select random terminal from starting molecule
                    AtomRmv = rnd(OneBondAtomsMolecule)

                    #Attach fragment to end of molecule
                    StartingMolecule.AddBond(FragAtom, AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)

                elif len(AromaticAtomsMolecule) > 0:
                    #Randomly select 2 bonded (non-branch base) aromatic atom 
                    ArmtcAtom = rnd(AromaticAtomsMolecule)

                    #Randomly select terminal atom from fragment
                    FragAtom = rnd(OneBondAtomsFragment)

                    #Add Bond
                    StartingMolecule.AddBond(ArmtcAtom, FragAtom, Chem.BondType.SINGLE)

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)
                    
            else:
                if Verbose:
                    print('Edge case, returning empty objects')
                Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
    except Exception as E:
        if Verbose:
            print(E)
            print('Index error, starting molecule probably too short, trying different mutation')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
      
    return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited

def RemoveFragment(InputMolecule, BondTypes, showdiff=False, Verbose=False):
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
    
    Will allow x attempts for this to happen before giving up on this mutation(?)
    
    Only implement this mutation when number of atoms exceeds a certain number e.g. 5/6 max mol length
    
    Max fragment removal length of 2/5 max mol length
    """

    try:
        StartingMoleculeUnedited = deepcopy(InputMolecule)

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
        
        if len(AromaticAtoms) > 0:
            # Choose whether to remove aromatic ring or not
            RemoveRing =  rnd([True, False])
        
        else:
            RemoveRing = False

        if Verbose:
            print(f'Attempting to remove aromatic ring: {RemoveRing}')

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
                    StartingMolecule = Chem.RWMol(StartingMolecule) #Need to convert back to editable mol to use with 'MolCheckandPlot'

                    Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, 
                                                            StartingMolecule, 
                                                            showdiff)
            except Exception as E:
                if Verbose:
                    print(E)
                    print('Remove Aromatic ring failed, returning empty objects')
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
            for atomidx in selected_atoms:
                atom = StartingMolecule.GetAtomWithIdx(atomidx)
                BondIdxs = [x.GetIdx() for x in atom.GetBonds()]
                SeveringBonds.append(random.sample(FinalAtomIdxs, 1))
                # Save index of atom on other side of chosen bond that was severed 

            SeveringBonds = [x[0] for x in SeveringBonds]

            for b_idx in SeveringBonds:
                b = StartingMolecule.GetBondWithIdx(b_idx)
                StartingMolecule.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())

            frags = Chem.GetMolFrags(StartingMolecule)

            # Only proceed if the removed fragment is less than a quarter the length of the molecule 
            if len(frags)==3 and len(frags[1]) <= len(StartingMoleculeUnedited.GetAtoms())*0.4 and len(frags[1]) >= 2:
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
                StartingMolecule = max(mol_frags, default=StartingMolecule, key=lambda m: m.GetNumAtoms())
                StartingMolecule = Chem.RWMol(StartingMolecule) #Need to convert back to editable mol to use with 'MolCheckandPlot'

                Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, 
                                                                            StartingMolecule, 
                                                                            showdiff)

            else:
                if Verbose:
                    print('Remove fragment failed, returning empty objects')
                    Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
                else:
                    Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

    except Exception as E:
        if Verbose:
            print(E)
            print('Remove fragment failed, returning empty objects')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
        else:
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

    return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES

def Mol_Crossover(StartingMolecule, CrossMol, showdiff=False, Verbose=False):
    try:
        """
        Take in two molecules, the molecule to be mutated, and a list of molecules to crossover with?

        Randomly fragment each molecule 

        Create bond between randomly selected atoms on the molecule

        Need to make sure that we are not bonding an aromatic to an aromatic

        Write code to save which molecule was crossed over with
        """

        StartingMoleculeUnedited = deepcopy(StartingMolecule)
        StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)
        CrossMolecule = Chem.RWMol(CrossMol)

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

        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited = AddFragment(StartingMolecule, 
                                                                                            CrossMolecule, 
                                                                                            InsertStyle, 
                                                                                            showdiff, 
                                                                                            Verbose)
    
    except Exception as E:
        print(E)
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited = None, None, None, None

    return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited

def RemoveAtom(StartingMolecule, BondTypes, fromAromatic=False, showdiff=True):
    """
    Function to replace atom from a selected list of atoms from a starting molecule.
    
    Args:
    - StartingMolecule: SMILES String of Starting Molecule
    - FromAromatic: If True, will remove atoms from aromatic rings 

    Steps:
    1. Get the indexes of all the bonds in the molecule 
    2. Check if there are any atoms that are aromatic 
    3. Select index of atom that will be replaced, from list of atoms with one or two neighbors only
    4. Get bond and bondtype of bonds selected atom has with its neighbor(s)
    5. If selected atom has one neighbor, remove atom and return edited molecule
    6. If selected atom has two neighbors:
        a. Get indexes of atoms bonded to selected atom
        b. Randomly select bond type to create between left over atoms
        c. Remove selected atom and create new bond of selected bond type between left over atoms 
    """
    StartingMoleculeUnedited = deepcopy(StartingMolecule)
    print(StartingMolecule, StartingMoleculeUnedited)

    #try:
    # Store indexes of atoms in molecule
    AtomIdxs = []

    # Check if starting molecule is completely aromatic
    for atom in StartingMoleculeUnedited.GetAtoms():
        if len((StartingMoleculeUnedited.GetAromaticAtoms())) == len(StartingMoleculeUnedited.GetAtoms()):
            print('Starting molecule is completely aromatic')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
            
        else:
            AtomIdxs.append(atom.GetIdx())

            # Make editable mol object from starting molecule
            StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)

            # Get number of bonds each atom in the molecule has and storing them in separate objects 
            OneBondAtomsMolecule = []
            TwoBondAtomsMolecule = []
            AromaticAtomsMolecule = []

            # Getting atoms in starting molecule with different amount of bonds, storing indexes in list
            for index in AtomIdxs:
                Atom = StartingMolecule.GetAtomWithIdx(int(index))
                if Atom.GetIsAromatic() and len(Atom.GetBonds()) == 2:
                    AromaticAtomsMolecule.append(index)
                elif len(Atom.GetBonds()) == 2:
                    TwoBondAtomsMolecule.append(index)
                elif len(Atom.GetBonds()) == 1:
                    OneBondAtomsMolecule.append(index)
                else:
                    continue
            
            #Select atom to be deleted from list of atom indexes, check that this list is greater than 0
            if len(AtomIdxs) == 0:
                print('Empty Atom Index List')
                Mut_Mol = None
                Mut_Mol_Sanitized = None
                MutMolSMILES = None
            elif fromAromatic and len(AromaticAtomsMolecule) > 0 and len(OneBondAtomsMolecule) > 0 and len(TwoBondAtomsMolecule) > 0:
                # Add the lists of the atoms with different numbers of bonds into one object 
                OneBondAtomsMolecule.extend(TwoBondAtomsMolecule).extend(AromaticAtomsMolecule)
                Indexes = OneBondAtomsMolecule
                RemoveAtomIdx = rnd(Indexes)
                RemoveAtomNeigbors = StartingMolecule.GetAtomWithIdx(RemoveAtomIdx).GetNeighbors()
            elif len(OneBondAtomsMolecule) > 0 and len(TwoBondAtomsMolecule) > 0:
                #Select a random atom from the index of potential replacement atoms that aren't aromatic
                OneBondAtomsMolecule.extend(TwoBondAtomsMolecule)
                Indexes = OneBondAtomsMolecule
                RemoveAtomIdx = rnd(Indexes)
                RemoveAtomNeigbors = StartingMolecule.GetAtomWithIdx(RemoveAtomIdx).GetNeighbors()

                if len(RemoveAtomNeigbors) == 1:
                    StartingMolecule.RemoveAtom(RemoveAtomIdx)
                elif len(RemoveAtomNeigbors) == 2:
                    StartingMolecule.RemoveAtom(RemoveAtomIdx)
                    StartingMolecule.AddBond(RemoveAtomNeigbors[0].GetIdx(), RemoveAtomNeigbors[1].GetIdx(), rnd(BondTypes))
                else:
                    print('Removed atom has illegal number of neighbors')
                    Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

                # Check number of heavy atoms before and after, should have reduced by one 
                if StartingMoleculeUnedited.GetNumHeavyAtoms() == StartingMolecule.GetNumHeavyAtoms():
                    print('Atom removal failed, returning empty object')
                    Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

                # Check what atom was removed from where
                print(f'{StartingMoleculeUnedited.GetAtomWithIdx(RemoveAtomIdx).GetSymbol()} removed from position {RemoveAtomIdx}')

                Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES  = MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff)
            
            else:
                print('Atom removal failed, returning empty object')
                Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

            # except:
            #     print('Atom removal could not be performed, returning empty objects')
            #     Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

    return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited

#Change so that we can have at most 1 benzene molecule
def InsertAromatic(StartingMolecule, AromaticMolecule, InsertStyle='Within', showdiff=False, Verbose=False):
    """
    Function to insert an aromatic atom into a starting molecule
    4. If inserting aromatic ring:
    - Check if fragment is aromatic
    - Combine starting molecule and fragment into single disconnected Mol object 
    - Split atom indexes of starting molecule and fragment and save in different objects
    - Check number of bonds each atom has in starting molecule and fragment, exclude any atoms that don't have 
    exactly two bonds
    - Randomly select one of 2-bonded atoms and store the index of the Atom, and store bond objects of atom's bonds
    - Get the atom neighbors of selected atom 
    - Remove selected atom 
    - Select two unique atoms from cyclic atom
    - Create a new bond between each of the neighbors of the removed atom and each of the terminal atoms on the 
    fragment 
    """

    print(f'InsertStyle is: {InsertStyle}')

    StartingMoleculeUnedited = deepcopy(StartingMolecule)
    Fragment = AromaticMolecule

    #Always check if fragment or starting molecule is a cyclic (benzene) molecule

    try:
        if len(Fragment.GetAromaticAtoms()) != len(Fragment.GetAtoms()):
            if Verbose:
                print('Fragment is not cyclic')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
        
        elif len((StartingMoleculeUnedited.GetAromaticAtoms())) == len(StartingMoleculeUnedited.GetAtoms()):
            if Verbose:
                print('Starting molecule is completely aromatic')
            Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

        else:
            Chem.SanitizeMol(Fragment)

            # Add fragment to Mol object of starting molecule
            StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)
            StartingMolecule.InsertMol(Fragment)

            # Get indexes of starting molecule and fragment, store them in separate objects 
            frags = Chem.GetMolFrags(StartingMolecule)
            StartMolIdxs = frags[0]
            FragIdxs = frags[1]

            OneBondAtomsMolecule = []
            TwoBondAtomsMolecule = []
            AromaticAtomsMolecule = []

            # Getting atoms in starting molecule with different amount of bonds, storing indexes in lists
            for index in StartMolIdxs:
                Atom = StartingMolecule.GetAtomWithIdx(int(index))
                if Atom.GetIsAromatic():
                    AromaticAtomsMolecule.append(index) 
                if len(Atom.GetBonds()) == 2:
                    TwoBondAtomsMolecule.append(index)
                elif len(Atom.GetBonds()) == 1:
                    OneBondAtomsMolecule.append(index)
                else:
                    continue

            if InsertStyle == 'Within':
                # Randomly choose two unique atoms from aromatic molecule
                ArmtcAtoms = sample(FragIdxs, 2)

                # Select random two bonded atom
                AtomRmv = rnd(TwoBondAtomsMolecule)

                # Get atom neighbor indexes, remove bonds between selected atom and neighbors 
                neighbors = [x.GetIdx() for x in StartingMolecule.GetAtomWithIdx(AtomRmv).GetNeighbors()]

                # Randomly choose which bond of target atom to sever
                SeverIdx = rnd([0,1])

                # Sever the selected bond
                StartingMolecule.RemoveBond(neighbors[SeverIdx], AtomRmv)

                #For situation where bond before target atom is severed
                if SeverIdx == 0:
                    StartingMolecule.AddBond(ArmtcAtoms[0], AtomRmv - 1, Chem.BondType.SINGLE)
                    StartingMolecule.AddBond(ArmtcAtoms[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, 
                                                                                StartingMolecule, 
                                                                                showdiff)

                #For situation where bond after target atom is severed
                else:
                    StartingMolecule.AddBond(ArmtcAtoms[0], AtomRmv + 1, Chem.BondType.SINGLE) 
                    StartingMolecule.AddBond(ArmtcAtoms[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, 
                                                                                StartingMolecule, 
                                                                                showdiff)

            elif InsertStyle == 'Edge':

                if len(OneBondAtomsMolecule) == 0:
                    print('No one-bonded terminal atoms in starting molecule, returning empty object')
                    Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None
                
                else:
                    # Randomly choose two unique atoms from aromatic molecule
                    ArmtcAtoms = rnd(FragIdxs)

                    # Select random one bonded atom
                    AtomRmv = rnd(OneBondAtomsMolecule)

                    StartingMolecule.AddBond(ArmtcAtoms, AtomRmv, Chem.BondType.SINGLE) 

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = MolCheckandPlot(StartingMoleculeUnedited, 
                                                                                StartingMolecule, 
                                                                                showdiff)
                
            else:
                print('Edge case, returning empty objects')
                Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

    except:
        print('Index error, starting molecule probably too short, trying different mutation')
        Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES = None, None, None

    return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited

def Mutate(StartingMolecule, Mutation, AromaticMolecule, AtomicNumbers, BondTypes,
           Atoms, showdiff, Fragments):
    
    print(f'Mutation being performed: {Mutation}')
    if Mutation == 'AddAtom':
        result = AddAtom(StartingMolecule, AtomicNumbers, BondTypes, showdiff=showdiff)

    elif Mutation == 'ReplaceAtom':
        result = ReplaceAtom(StartingMolecule, Atoms, fromAromatic=False, showdiff=showdiff)

    elif Mutation == 'ReplaceBond':
        result = ReplaceBond(StartingMolecule, BondTypes, showdiff=showdiff)
    
    elif Mutation == 'RemoveAtom':
        result = RemoveAtom(StartingMolecule, BondTypes, fromAromatic=False, showdiff=showdiff)

    elif Mutation == 'AddFragment':
        InsertStyle = rnd(['Within', 'Egde'])
        result = AddFragment(StartingMolecule, rnd(Fragments), InsertStyle=InsertStyle, showdiff=showdiff)
    
    elif Mutation == 'RemoveFragment':
        pass

    elif Mutation == 'Mol_Crossover':
        pass

    else:
        InsertStyle = rnd(['Within', 'Egde'])
        result = InsertAromatic(StartingMolecule, AromaticMolecule, showdiff=showdiff, InsertStyle=InsertStyle)
    
    return result

def CheckSubstruct(MutMol):
    ### Check for unwanted substructures

    #Checking for sequential oxgens
    SingleBondOxygens = MutMol.HasSubstructMatch(Chem.MolFromSmarts('OO')) 
    DoubleBondOxygens = MutMol.HasSubstructMatch(Chem.MolFromSmarts('O=O'))
    DoubleCDoubleO = MutMol.HasSubstructMatch(Chem.MolFromSmarts('C=C=O'))    
    DoubleCDoubleC = MutMol.HasSubstructMatch(Chem.MolFromSmarts('C=C=C'))    
    BridgeHead = MutMol.HasSubstructMatch(Chem.MolFromSmarts('c-c'))    

    ### Remove C=C bonds that aren't aromatic

    # Check for sequence of single or double bonded oxygens or Bridgehead carbonds
    if SingleBondOxygens or DoubleBondOxygens or DoubleCDoubleO or DoubleCDoubleC or BridgeHead:
        print('Undesirable substructure found, returning empty object')
        return True
    else:
        return False

def fitfunc(MoleculeSMILES, Generation):
    return random.randint(1, Generation*5000)

# Function to run a command from a python script
def runcmd(cmd, verbose = False, *args, **kwargs):
    #bascially allows python to run a bash command, and the code makes sure 
    #the error of the subproceess is communicated if it fails
    process = subprocess.run(
        cmd,
        text=True,
        shell=True)
    
    return process

def GeneratePDB(SMILES, PATH, CONFORMATTEMPTS=10):
    """
    Function that generates PDB file from RDKit Mol object, for use with Packmol
    Inputs:
        - SMILES: SMILES string to be converted to PDB
        - PATH: Location that the PDB will stored
        - CONFORMATTEMPTS: Max number of tries (x5000) to find converged conformer for molecule
    """
    SMILESMol = Chem.MolFromSmiles(SMILES) # Create mol object
    SMILESMol = Chem.AddHs(SMILESMol) # Need to make Hydrogens explicit

    AllChem.EmbedMolecule(SMILESMol, AllChem.ETKDG()) #Create conformer using ETKDG method

    # Initial parameters for conformer optimisation
    MMFFSMILES = 1 
    ConformAttempts = 1
    MaxConformAttempts = CONFORMATTEMPTS

    # Ensure that script continues to iterate until acceptable conformation found
    while MMFFSMILES !=0 and ConformAttempts <= MaxConformAttempts: # Checking if molecule converged
        MMFFSMILES = Chem.rdForceFieldHelpers.MMFFOptimizeMolecule(SMILESMol, maxIters=5000)
        ConformAttempts += 1
        
    # Record parameterised conformer as pdb to be used with packmol later 
    Chem.MolToPDBFile(SMILESMol, f'{PATH}')

def GetMolCharge(PATH):
    """
    Retreive molecule charge from Moltemplate file
    """

    with open(PATH, 'r') as file:
        data = file.readlines()
        charge = data[-1].split('#')[-1].split('\n')[0] #Horrendous way of getting the charge

    return charge

def CheckMoveFile(Name, STARTINGDIR, FileType, CWD):
    if os.path.exists(f"{os.path.join(CWD, f'{Name}.{FileType}')}"):
        print(f'Specified {FileType} file already exists in this location, overwriting')
        os.remove(f"{os.path.join(CWD, f'{Name}.{FileType}')}")

    os.rename(f"{os.path.join(STARTINGDIR, f'{Name}.{FileType}')}", f"{os.path.join(CWD, f'{Name}.{FileType}')}")

def MakeMoltemplateFile(Name, CWD, NumMols, BoxL):
    if os.path.exists(f"{os.path.join(CWD, f'{Name}_system.lt')}"):
        print('Specified Moltemplate file already exists in this location, overwriting.')
        os.remove(f"{os.path.join(CWD, f'{Name}_system.lt')}")

    with open(os.path.join(CWD, f'{Name}_system.lt'), 'x') as file:
                file.write(f"""
import "{Name}.lt"  # <- defines the molecule type.

# Periodic boundary conditions:
write_once("Data Boundary") {{
   0.0  {BoxL}.00  xlo xhi
   0.0  {BoxL}.00  ylo yhi
   0.0  {BoxL}.00  zlo zhi
}}

ethylenes = new {Name} [{NumMols}]
""")

def GetMolMass(mol):
    formula = CalcMolFormula(mol)

    parts = re.findall("[A-Z][a-z]?|[0-9]+", formula)
    mass = 0

    for index in range(len(parts)):
        if parts[index].isnumeric():
            continue

        atom = Chem.Atom(parts[index])
        multiplier = int(parts[index + 1]) if len(parts) > index + 1 and parts[index + 1].isnumeric() else 1
        mass += atom.GetMass() * multiplier
    return mass

def MakePackmolFile(Name, CWD, NumMols, BoxL):
    if os.path.exists(f"{os.path.join(CWD, f'{Name}.inp')}"):
        print('Packmol file already exists in this location, overwriting')
        os.remove(f"{os.path.join(CWD, f'{Name}.inp')}")

    with open(os.path.join(CWD, f'{Name}.inp'), 'x') as file:
        file.write(f"""
tolerance 2.0
output {Name}_PackmolFile.pdb

filetype pdb

structure {Name}.pdb
number {NumMols} 
inside cube 0. 0. 0. {BoxL}.
end structure""")

def MakeMoltemplateFile(Name, CWD, NumMols, BoxL):
    if os.path.exists(f"{os.path.join(CWD, f'{Name}_system.lt')}"):
        print('Specified Moltemplate file already exists in this location, overwriting.')
        os.remove(f"{os.path.join(CWD, f'{Name}_system.lt')}")

    with open(os.path.join(CWD, f'{Name}_system.lt'), 'x') as file:
                file.write(f"""
import "{Name}.lt"  # <- defines the molecule type.

# Periodic boundary conditions:
write_once("Data Boundary") {{
   0.0  {BoxL}.00  xlo xhi
   0.0  {BoxL}.00  ylo yhi
   0.0  {BoxL}.00  zlo zhi
}}

ethylenes = new {Name} [{NumMols}]
""")
    
def CalcBoxLen(MolMass, TargetDens, NumMols):
    # Very conservative implementation of Packmol volume guesser
    BoxL = (((MolMass * NumMols * 2)/ TargetDens) * 1.5) ** (1./3.)
    BoxLRounded = int(BoxL)
    return BoxLRounded

def MakeLAMMPSFile(Name, CWD, Temp, GKRuntime):
    if os.path.exists(f"{os.path.join(CWD, f'{Name}_system_{Temp}K.lammps')}"):
        print('Specified Moltemplate file already exists in this location, overwriting.')
        os.remove(f"{os.path.join(CWD, f'{Name}_system_{Temp}K.lammps')}")

    if os.path.exists(f"{os.path.join(CWD, f'{Name}_system_373K.lammps')}"):
        print('Specified Moltemplate file already exists in this location, overwriting.')
        os.remove(f"{os.path.join(CWD, f'{Name}_system_373K.lammps')}")

    # Write LAMMPS file for 40C run
    with open(os.path.join(CWD, f'{Name}_system_{Temp}K.lammps'), 'x') as file:
        file.write(f"""

# Setup parameters
variable       		T equal {Temp} # Equilibrium temperature [K]
log             	logEQM_{Name}_T${{T}}KP1atm.out

#include         "{Name}_system.in.init" Need to determine correct Kspace params

# Potential information
units           	real
dimension       	3
boundary        	p p p
atom_style      	full

pair_style      	lj/cut/coul/cut 12.0 12.0 
bond_style      	harmonic
angle_style     	harmonic
dihedral_style 		opls
improper_style     	harmonic
pair_modify 		mix geometric tail yes
special_bonds   	lj/coul 0.0 0.0 0.5

# Read lammps data file consist of molecular topology and forcefield info
read_data       	{Name}_system.data
neighbor        	2.0 bin
neigh_modify 		every 1 delay 0 check yes

include         "{Name}_system.in.charges"
include         "{Name}_system.in.settings"

# Define variables
variable        	eqmT equal $T			 			# Equilibrium temperature [K]
variable        	eqmP equal 1.0						# Equilibrium pressure [atm]
variable    		p equal 100						    # Nrepeat, correlation length
variable    		s equal 10       					# Nevery, sample interval
variable    		d equal $s*$p  						# Nfreq, dump interval
variable 			rho equal density

# Minimize system at target temperature using the default conjugate gradient method
velocity        	all create ${{eqmT}} 482648
fix             	min all nve
thermo          	10
thermo_style 		custom step temp press density pxx pyy pzz pxy pxz pyz pe ke etotal evdwl ecoul epair ebond eangle edihed eimp emol etail enthalpy vol
# dump            	1 all custom 10 min_w_{Name}_T${{T}}KP1atm.lammpstrj id mol type x y z mass q
# dump            	2 all custom 10 min_u_{Name}_T${{T}}KP1atm.lammpstrj id mol type xu yu zu mass q
# dump_modify     	1 sort id
# dump_modify     	2 sort id
minimize        	1.0e-16 1.06e-6 100000 500000
# undump          	1
# undump          	2
write_restart   	Min_{Name}_T${{T}}KP1atm.restart

unfix           	min
reset_timestep  	0
neigh_modify 		every 1 delay 0 check yes

# NVT at high temperature
fix             	nvt1000K all nvt temp 1000.0 1000.0 100.0
thermo			    $d
thermo_style 		custom step temp press density pxx pyy pzz pxy pxz pyz pe ke etotal evdwl ecoul epair ebond eangle edihed eimp emol etail enthalpy vol
fix 				thermo_print all print $d "$(step) $(temp) $(press) $(density) $(pxx) $(pyy) $(pzz) $(pxy) $(pxz) $(pyz) $(pe) $(ke) $(etotal) $(evdwl) $(ecoul) $(epair) $(ebond) $(eangle) $(edihed) $(eimp) $(emol) $(etail) $(enthalpy) $(vol)" &
					append thermoNVT1000K_{Name}_T${{T}}KP1atm.out screen no title "# step temp press density pxx pyy pzz pxy pxz pyz pe ke etotal evdwl ecoul epair ebond eangle edihed eimp emol etail enthalpy vol"
# dump            	1 all custom $d NVT1000K_u_{Name}_T${{T}}KP1atm.lammpstrj id mol type xu yu zu mass q
# dump_modify     	1 sort id
run            		250000
# undump          	1
unfix			    nvt1000K
unfix               thermo_print
write_restart   	NVT_{Name}_T1000KP1atm.restart

# NPT: Isothermal-isobaric ensemble to set the desired pressure; compute average density at that pressure
fix 				NPT all npt temp ${{eqmT}} ${{eqmT}} 100.0 iso ${{eqmP}} ${{eqmP}} 25.0
fix             	dave all ave/time $s $p $d v_rho ave running file eqmDensity_{Name}_T${{T}}KP1atm.out
thermo				$d
thermo_style 		custom step temp press density pxx pyy pzz pxy pxz pyz pe ke etotal evdwl ecoul epair ebond eangle edihed eimp emol etail enthalpy vol
fix 				thermo_print all print $d "$(step) $(temp) $(press) $(density) $(pxx) $(pyy) $(pzz) $(pxy) $(pxz) $(pyz) $(pe) $(ke) $(etotal) $(evdwl) $(ecoul) $(epair) $(ebond) $(eangle) $(edihed) $(eimp) $(emol) $(etail) $(enthalpy) $(vol)" &
					append thermoNPT_{Name}_T${{T}}KP1atm.out screen no title "# step temp press density pxx pyy pzz pxy pxz pyz pe ke etotal evdwl ecoul epair ebond eangle edihed eimp emol etail enthalpy vol"
# dump            	1 all custom $d NPT_u_{Name}_T${{T}}KP1atm.lammpstrj id mol type xu yu zu mass q
# dump_modify     	1 sort id
run					1000000
# undump          	1
unfix				NPT
unfix               thermo_print
write_restart  		NPT_{Name}_T${{T}}KP1atm.restart

# NVT: Canonical ensemble to deform the box to match increase in P in previous step
variable        	averho equal f_dave
variable        	adjustrho equal (${{rho}}/${{averho}})^(1.0/3.0) # Adjustment factor needed to bring rho to averge rho
unfix				dave
fix             	NVT all nvt temp ${{eqmT}} ${{eqmT}} 100.0	
fix             	adjust all deform 1 x scale ${{adjustrho}} y scale ${{adjustrho}} z scale ${{adjustrho}}
thermo         		$d
thermo_style 		custom step temp press density pxx pyy pzz pxy pxz pyz pe ke etotal evdwl ecoul epair ebond eangle edihed eimp emol etail enthalpy vol
fix 				thermo_print all print $d "$(step) $(temp) $(press) $(density) $(pxx) $(pyy) $(pzz) $(pxy) $(pxz) $(pyz) $(pe) $(ke) $(etotal) $(evdwl) $(ecoul) $(epair) $(ebond) $(eangle) $(edihed) $(eimp) $(emol) $(etail) $(enthalpy) $(vol)" &
					append thermoNVT_{Name}_T${{T}}KP1atm.out screen no title "# step temp press density pxx pyy pzz pxy pxz pyz pe ke etotal evdwl ecoul epair ebond eangle edihed eimp emol etail enthalpy vol"
# dump            	1 all custom $d NVT_u_{Name}_T${{T}}KP1atm.lammpstrj id mol type xu yu zu mass q
# dump_modify     	1 sort id
run					500000
# undump          	1
unfix				NVT
unfix           	adjust
unfix               thermo_print
write_restart  		NVT_{Name}_T${{T}}KP1atm.restart

# NVE: Microcanonical ensemble to explore the configuration space at constant T and V; relax
fix	       			NVE all nve
fix 				thermostat all langevin ${{eqmT}} ${{eqmT}} 100.0 39847 
thermo          	$d
thermo_style 		custom step temp press density pxx pyy pzz pxy pxz pyz pe ke etotal evdwl ecoul epair ebond eangle edihed eimp emol etail enthalpy vol

fix 			thermo_print all print $d "$(step) $(temp) $(press) $(density) $(pxx) $(pyy) $(pzz) $(pxy) $(pxz) $(pyz) $(pe) $(ke) $(etotal) $(evdwl) $(ecoul) $(epair) $(ebond) $(eangle) $(edihed) $(eimp) $(emol) $(etail) $(enthalpy) $(vol)" &
			append thermoNVE_{Name}_T${{T}}KP1atm.out screen no title "# step temp press density pxx pyy pzz pxy pxz pyz pe ke etotal evdwl ecoul epair ebond eangle edihed eimp emol etail enthalpy vol"

run             	250000

unfix           	NVE

unfix 			thermostat

unfix               	thermo_print

# Output the state generated that is needed to shear the molecules

write_restart  		state_{Name}_T${{T}}KP1atm.restart
write_data 		equilibrated.data

# Green-Kubo method via fix ave/correlate

log                 logGKvisc_{Name}_T${{T}}KP1atm.out

# Define variables
variable        	eqmT equal $T 				# Equilibrium temperature [K]
variable        	tpdn equal 3*1E6 			# Time for production run [fs]

variable    		dt equal 1.0				# time step [fs]
variable 		    V equal vol

# convert from LAMMPS real units to SI
variable    		kB equal 1.3806504e-23
variable            kCal2J equal 4186.0/6.02214e23
variable    		atm2Pa equal 101325.0		
variable    		A2m equal 1.0e-10 			
variable    		fs2s equal 1.0e-15 			
variable			Pas2cP equal 1.0e+3			
variable    		convert equal ${{atm2Pa}}*${{atm2Pa}}*${{fs2s}}*${{A2m}}*${{A2m}}*${{A2m}}
variable            convertWk equal ${{kCal2J}}*${{kCal2J}}/${{fs2s}}/${{A2m}}

##################################### Viscosity Calculation #####################################################
timestep     		${{dt}}						# define time step [fs]

compute         	TT all temp
compute         	myP all pressure TT

###### Thermal Conductivity Calculations 

compute         myKE all ke/atom
compute         myPE all pe/atom
compute         myStress all stress/atom NULL virial

# compute heat flux vectors
compute         flux all heat/flux myKE myPE myStress
variable        Jx equal c_flux[1]/vol
variable        Jy equal c_flux[2]/vol
variable        Jz equal c_flux[3]/vol

fix             	1 all nve
fix             	2 all langevin ${{eqmT}} ${{eqmT}} 100.0 482648

variable        	myPxx equal c_myP[1]
variable        	myPyy equal c_myP[2]
variable       		myPzz equal c_myP[3]
variable     		myPxy equal c_myP[4]
variable     		myPxz equal c_myP[5]
variable     		myPyz equal c_myP[6]

fix             	3 all ave/time 1 1 1 v_myPxx v_myPyy v_myPzz v_myPxy v_myPxz v_myPyz ave one #file Stress_AVGOne111_{Name}_T${{T}}KP1atm.out
fix             	4 all ave/time $s $p $d v_myPxx v_myPyy v_myPzz v_myPxy v_myPxz v_myPyz ave one file Stress_AVGOnespd_{Name}_T${{T}}KP1atm.out

fix          SS all ave/correlate $s $p $d &
             v_myPxy v_myPxz v_myPyz type auto file S0St.dat ave running

variable     scale equal ${{convert}}/(${{kB}}*$T)*$V*$s*${{dt}}
variable     v11 equal trap(f_SS[3])*${{scale}}
variable     v22 equal trap(f_SS[4])*${{scale}}
variable     v33 equal trap(f_SS[5])*${{scale}}

fix          JJ all ave/correlate $s $p $d &
             c_flux[1] c_flux[2] c_flux[3] type auto &
             file profile.heatflux ave running

variable        scaleWk equal ${{convertWk}}/${{kB}}/$T/$T/$V*$s*${{dt}}
variable        k11 equal trap(f_JJ[3])*${{scaleWk}}
variable        k22 equal trap(f_JJ[4])*${{scaleWk}}
variable        k33 equal trap(f_JJ[5])*${{scaleWk}}

##### Diffusion Coefficient Calculations 

compute         vacf all vacf   #Calculate velocity autocorrelation function
fix             5 all vector 1 c_vacf[4]
variable        vacf equal 0.33*${{dt}}*trap(f_5)

thermo       		$d
thermo_style custom step temp press v_myPxy v_myPxz v_myPyz v_v11 v_v22 v_v33 vol v_Jx v_Jy v_Jz v_k11 v_k22 v_k33 v_vacf

fix thermo_print all print $d "$(temp) $(press) $(v_myPxy) $(v_myPxz) $(v_myPyz) $(v_v11) $(v_v22) $(v_v33) $(vol) $(v_Jx) $(v_Jy) $(v_Jz) $(v_k11) $(v_k22) $(v_k33) $(v_vacf)" &
    append thermoNVE_{Name}_T${{T}}KP1atm.out screen no title "# temp press v_myPxy v_myPxz v_myPyz v_v11 v_v22 v_v33 vol v_Jx v_Jy v_Jz v_k11 v_k22 v_k33 v_vacf"

# Dump all molecule coordinates

# save thermal conductivity to file
variable     kav equal (v_k11+v_k22+v_k33)/3.0
fix          fxave1 all ave/time $d 1 $d v_kav file lamda.txt

# save viscosity to a file
variable     visc equal (v_v11+v_v22+v_v33)/3.0
fix          fxave2 all ave/time $d 1 $d v_visc file visc.txt

# save diffusion coefficient to a file
fix          fxave3 all ave/time $d 1 $d v_vacf file diff_coeff.txt

run          {GKRuntime}

variable     ndens equal count(all)/vol
print        "Average viscosity: ${{visc}} [Pa.s] @ $T K, ${{ndens}} atoms/A^3"

write_restart   	GKvisc_{Name}_T${{T}}KP1atm.restart
write_data          GKvisc_{Name}_T${{T}}KP1atm.data
""")
        
def GenMolChecks(result, GenerationMolecules, MaxNumHeavyAtoms, MinNumHeavyAtoms, MaxAromRings):
    if result[0]!= None:
        #Get number of heavy atoms in mutated molecule
        NumHeavyAtoms = result[0].GetNumHeavyAtoms()
        
        # Limit number of heavy atoms in generated candidates
        if NumHeavyAtoms > MaxNumHeavyAtoms:
            print('Molecule has too many heavy atoms')
            MutMol = None
        
        # Check if molecule is too short
        elif NumHeavyAtoms < MinNumHeavyAtoms:
            print('Molecule too short')
            MutMol = None

        # Check if candidate has already been generated by checking if SMILES string is in master list
        elif result[2] in GenerationMolecules:
            print('Molecule previously generated')
            MutMol = None

        # Check for illegal substructures
        elif CheckSubstruct(result[0]):
            MutMol = None

        # Check for number of Aromatic Rings
        elif Chem.rdMolDescriptors.CalcNumAromaticRings(result[0]) > MaxAromRings:
            print('Too many aromatic rings')
            print(Chem.rdMolDescriptors.CalcNumAromaticRings(result[0]))
            MutMol = None
        
        # Check for bridgehead atoms
        elif Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(result[0]) > 0:
            print('Too many bridged aromatic rings')
            MutMol = None
        
        else:
            MutMol = result[0]

    # Check for null results or fragmented molecules
    elif result[0] == None or len(Chem.GetMolFrags(result[0])) > 1:
        print('Fragmented molecule generated')
        MutMol = None

    else:
        MutMol = result[0]

    return MutMol

def fitfunc(MoleculeSMILES, Generation):
    return random.randint(1, Generation*5000)

def GetDens(DensityFile):
    try:
        with open(f'{DensityFile}', 'r+') as file:
            content = file.readlines()[-1]
            content = content.split(' ')[-1]
            Density = float(content.split('\n')[0])
    except Exception as E:
        print(E)
        print('Value for Density not found')
        Density = 0
    return Density

def GetKVisc(DVisc, Dens):
    try:
        return DVisc / Dens
    except:
        return None

def DataUpdate(MoleculeDatabase, IDCounter, MutMolSMILES, MutMol, MutationList, HeavyAtoms, ID, Charge, MolMass, Predecessor):
    MoleculeDatabase.at[IDCounter, 'SMILES'] = MutMolSMILES
    MoleculeDatabase.at[IDCounter, 'MolObject'] = MutMol
    MoleculeDatabase.at[IDCounter, 'MutationList'] = MutationList
    MoleculeDatabase.at[IDCounter, 'HeavyAtoms'] = HeavyAtoms 
    MoleculeDatabase.at[IDCounter, 'ID'] = ID
    MoleculeDatabase.at[IDCounter, 'Charge'] = Charge
    MoleculeDatabase.at[IDCounter, 'MolMass'] = MolMass
    MoleculeDatabase.at[IDCounter, 'Predecessor'] = Predecessor

    return MoleculeDatabase

def CreateArrayJob(STARTINGDIR, CWD, Generation, SimName, Agent):
    #Create an array job for each separate simulation
    if Generation == 1:
        BotValue = 0
        TopValue = 49
    else:
        GenerationRange = Generation*35 - 20
        ListRange = list(range(GenerationRange, GenerationRange + 35))
        TopValue = ListRange[-1]
        BotValue = ListRange[0]

    if os.path.exists(f"{os.path.join(CWD, f'{Agent}_{SimName}.pbs')}"):
        print(f'Specified file already exists in this location, overwriting')
        os.remove(f"{os.path.join(CWD, f'{Agent}_{SimName}.pbs')}")       

    with open(os.path.join(STARTINGDIR, 'Molecules', f'Generation_{Generation}', f'{Agent}_{SimName}.pbs'), 'w') as file:
        file.write(f"""#!/bin/bash
#PBS -l select=1:ncpus=32:mem=62gb
#PBS -l walltime=07:59:59
#PBS -J {BotValue}-{TopValue}

module load intel-suite/2020.2
module load mpi/intel-2019.6.166

cd /rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/GNN_Viscosity_Prediction/Molecules/Generation_{Generation}/Generation_{Generation}_Molecule_${{PBS_ARRAY_INDEX}}
mpiexec ~/tmp/bin/lmp -in Generation_{Generation}_Molecule_${{PBS_ARRAY_INDEX}}_system_{SimName}
""")
    os.rename(f"{os.path.join(STARTINGDIR, 'Molecules', f'Generation_{Generation}', f'{Agent}_{SimName}.pbs')}", f"{os.path.join(CWD, f'{Agent}_{SimName}.pbs')}")

def CreateArrayJobMultiRun(STARTINGDIR, CWD, Generation, SimName, Agent):
    pass

def GetDVI(DVisc40, DVisc100):
    """
    Let's use the DVI method used by Kajita or
    we could make our own relation and see how that 
    compares to KVI measurements 
    """
    try:
        S = (-log10( (log10(DVisc40) + 1.2) / (log10(DVisc100) + 1.2) )) / (log10(175/235))
        DVI = 220 - (7*(10**S))
        return DVI
    except:
        return None

def GetKVI(DVisc40, DVisc100, Dens40, Dens100, STARTINGDIR):
    # Get Kinematic Viscosities
    KVisc40 = GetKVisc(DVisc40, Dens40)
    KVisc100 = GetKVisc(DVisc100, Dens100)

    RefVals = pd.read_excel(os.path.join(STARTINGDIR, 'VILookupTable.xlsx'), index_col=None)

    if KVisc100 == None:
        VI = None

    elif KVisc100 >= 2:
        # Retrive L and H value
        RefVals['Diffs'] = abs(RefVals['KVI'] - KVisc100)
        RefVals_Sorted = RefVals.sort_values(by='Diffs')
        NearVals = RefVals_Sorted.head(2)

        # Put KVI, L and H values into List to organise values for interpolation
        KVIVals = sorted(NearVals['KVI'].tolist())
        LVals = sorted(NearVals['L'].tolist())
        HVals = sorted(NearVals['H'].tolist())

        # Perform Interpolation,
        InterLVal = LVals[0] + (((KVisc100 - KVIVals[0])*(LVals[1]-LVals[0])) / (KVIVals[1]-KVIVals[0]))
        InterHVal = HVals[0] + (((KVisc100 - KVIVals[0])*(HVals[1]-HVals[0])) / (KVIVals[1]-KVIVals[0]))

        # Calculate KVI
        # If U > H
        if KVisc40 >= InterHVal:
            VI = ((InterLVal - KVisc40)/(InterLVal - InterHVal)) * 100
        # If H > U
        elif InterHVal > KVisc40:
            N = ((log10(InterHVal) - log10(KVisc40))/log10(KVisc100))
            VI = (((10**N)-1)/0.00715) + 100
        else:
            print('VI Undefined for input Kinematic Viscosities')
            VI = None
    else:
        print('VI Undefined for input Kinematic Viscosities')
        VI = None
    
    return VI

def plotmol(mol):
    img = Draw.MolsToGridImage([mol], subImgSize=(800, 800))
    img.show()

# Add indexes to molecule for visualisation
def mol_with_atom_index(mol):

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

class SCScorer():
    
    #Model Parameters
    project_root = os.path.dirname(os.path.dirname(__file__))

    score_scale = 5.0
    min_separation = 0.25

    FP_len = 1024
    FP_rad = 2

    def __init__(self, score_scale=score_scale):
        self.vars = []
        self.score_scale = score_scale
        self._restored = False

    def restore(self, weight_path=os.path.join('C:/Users/eeo21/VSCodeProjects/GNN_Viscosity_Prediction/full_reaxys_model_1024bool'), FP_rad=FP_rad, FP_len=FP_len):
        self.FP_len = FP_len; self.FP_rad = FP_rad
        self._load_vars(weight_path)
        print('Restored variables from {}'.format(weight_path))

        if 'uint8' in weight_path or 'counts' in weight_path:
            def mol_to_fp(self, mol):
                if mol is None:
                    return np.array((self.FP_len,), dtype=np.uint8)
                fp = AllChem.GetMorganFingerprint(mol, self.FP_rad, useChirality=True) # uitnsparsevect
                fp_folded = np.zeros((self.FP_len,), dtype=np.uint8)
                for k, v in six.iteritems(fp.GetNonzeroElements()):
                    fp_folded[k % self.FP_len] += v
                return np.array(fp_folded)
        else:
            def mol_to_fp(self, mol):
                if mol is None:
                    return np.zeros((self.FP_len,), dtype=np.float32)
                return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, self.FP_rad, nBits=self.FP_len,
                    useChirality=True), dtype=bool)
        self.mol_to_fp = mol_to_fp

        self._restored = True
        return self

    def smi_to_fp(self, smi):
        if not smi:
            return np.zeros((self.FP_len,), dtype=np.float32)
        return self.mol_to_fp(self, Chem.MolFromSmiles(smi))

    def apply(self, x):
        if not self._restored:
            raise ValueError('Must restore model weights!')
        # Each pair of vars is a weight and bias term
        for i in range(0, len(self.vars), 2):
            last_layer = (i == len(self.vars)-2)
            W = self.vars[i]
            b = self.vars[i+1]
            x = np.matmul(x, W) + b
            if not last_layer:
                x = x * (x > 0) # ReLU
        x = 1 + (score_scale - 1) * sigmoid(x)
        return x

    def get_score_from_smi(self, smi='', v=False):
        if not smi:
            return ('', 0.)
        fp = np.array((self.smi_to_fp(smi)), dtype=np.float32)
        if sum(fp) == 0:
            if v: print('Could not get fingerprint?')
            cur_score = 0.
        else:
            # Run
            cur_score = self.apply(fp)
            if v: print('Score: {}'.format(cur_score))
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        else:
            smi = ''
        return (smi, cur_score)

    def _load_vars(self, weight_path):

        if weight_path.endswith('pickle'):
            import cPickle as pickle
            with open(weight_path, 'rb') as fid:
                self.vars = pickle.load(fid)
                self.vars = [x.tolist() for x in self.vars]
        elif weight_path.endswith('json.gz'):

            with gzip.GzipFile(weight_path, 'r') as fin:    # 4. gzip

                json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)
                json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
                self.vars = json.loads(json_str)
                self.vars = [np.array(x) for x in self.vars]

def SCScore(SMILESList):
    model = SCScorer()
    model.restore(os.path.join('C:/Users/eeo21/VSCodeProjects/GNN_Viscosity_Prediction/full_reaxys_model_1024bool', 'model.ckpt-10654.as_numpy.json.gz'))
    smis = SMILESList
    SCScoreList = []
    for smi in smis:
        (smi, sco) = model.get_score_from_smi(smi)
        SCScoreList.append(sco)
        print('%.4f <--- %s' % (sco, smi))
    
    return SCScoreList

def Toxicity(SMILESList):
    from gt4sd.properties import PropertyPredictorRegistry
    tox21 = PropertyPredictorRegistry.get_property_predictor('tox21', {'algorithm_version': 'v0'})

    Results = []

    for SMILES in SMILESList:
        Partial_Result = tox21(SMILES)
        Result = sum(Partial_Result)
        Results.append(Result)

    return Results

def TanimotoSimilarity(SMILES, SMILESList):
    # Calculate Tanimoto Similarity between target molecule and every other molecule in that population, crossover between molecules and least similar mols to it?
    # How does this work with the k-way tournament selection?

    ## Calculating Tanimoto Similarity using molecular fingerprints

    SMILESList = [x for x in SMILESList if x != SMILES]
    ms = [Chem.MolFromSmiles(x) for x in SMILESList]

    SMILESms = Chem.MolFromSmiles(SMILES)
    # Generate Morgan fingerprints
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    SMILESfps = fpgen.GetFingerprint(SMILESms)
    fps = [fpgen.GetFingerprint(x) for x in ms]

    #Calculate Tanimoto similarity
    SimScores = []

    for index, Molecule in enumerate(SMILESList):
        Score = DataStructs.TanimotoSimilarity(SMILESfps, fps[index])
        SimScores.append([Molecule, Score])
    
    SimScoresOrdered = sorted(SimScores, key=lambda x: x[1], reverse=True)
    
    return SimScoresOrdered

def einstein(timestep, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz, volume, Temp, Time):

    kBT = Boltzmann * Temp
    Pxxyy = (Pxx - Pyy) / 2
    Pyyzz = (Pyy - Pzz) / 2

    '''
    Calculate the viscosity from the Einstein relation 
    by integrating the components of the pressure tensor
    '''
    timestep = timestep * 10**(-12)

    Pxy_int = integrate.cumtrapz(y=Pxy, dx=timestep, initial=0)
    Pxz_int = integrate.cumtrapz(y=Pxz, dx=timestep, initial=0)
    Pyz_int = integrate.cumtrapz(y=Pyz, dx=timestep, initial=0)

    Pxxyy_int = integrate.cumtrapz(y=Pxxyy, dx=timestep, initial=0)
    Pyyzz_int = integrate.cumtrapz(y=Pyyzz, dx=timestep, initial=0)

    integral = (Pxy_int**2 + Pxz_int**2 + Pyz_int**2 + Pxxyy_int**2 + Pyyzz_int**2) / 5

    viscosity = integral[1:] * (volume * 10**(-30) / (2 * kBT * Time[1:] * 10**(-12)) )

    return viscosity

def acf(data):
    steps = data.shape[0]
    lag = int(steps * 0.75)

    # Nearest size with power of 2 (for efficiency) to zero-pad the input data
    size = 2 ** np.ceil(np.log2(2 * steps - 1)).astype('int')

    # Compute the FFT
    FFT = np.fft.fft(data, size)

    # Get the power spectrum
    PWR = FFT.conjugate() * FFT

    # Calculate the auto-correlation from inverse FFT of the power spectrum
    COR = np.fft.ifft(PWR)[:steps].real

    autocorrelation = COR / np.arange(steps, 0, -1)

    return autocorrelation[:lag]

def green_kubo(timestep, Pxy, Pxz, Pyz, volume, Temp):
    # Calculate the kBT value
    kBT = Boltzmann * Temp

    # Calculate the ACFs
    Pxy_acf = acf(Pxy)
    Pxz_acf = acf(Pxz)
    Pyz_acf = acf(Pyz)

    avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf) / 3

    # Integrate the average ACF to get the viscosity
    timestep = timestep * 10**(-12)
    integral = integrate.cumtrapz(y=avg_acf, dx=timestep, initial=0)
    viscosity = integral * (volume * 10**(-30) / kBT)

    return avg_acf, viscosity

def GetVisc(Name, Temp, STARTDIR, unit='atm', plot=False):
    print(Name)
    chdir(join(STARTDIR, Name))

    Runs = [x for x in listdir(getcwd()) if os.path.isdir(x)]
    print(Runs)

    DataframeEinstein = pd.DataFrame()
    DataframeGK = pd.DataFrame()

    for Run in Runs:
        try:
            print(join(STARTDIR, Name, Run))
            chdir(join(STARTDIR, Name, Run))
            df = pd.read_csv(f'Stress_AVGOnespd_{Name}_T{Temp}KP1atm.out')

            datafile = f'Stress_AVGOnespd_{Name}_T{Temp}KP1atm.out'
            steps = len(df) - 5 # Num steps to read from the pressure tensor file
            timestep = 1 # What timestep are you using in the pressure tensor file

            # Extract viscosity from log file
            with open(f'logGKvisc_{Name}_T{Temp}KP1atm.out', "r") as file:
                content = file.readlines()
                for line in content:
                    linecontent = line.split(' ')
                    linecontent = [x for x in linecontent if x != '']
                    if len(linecontent) == 18:
                        try:
                            vol = linecontent[9]
                            volume = float(vol)
                        except:
                            pass

            # Conversion ratio from atm/bar to Pa
            if unit == 'Pa':
                conv_ratio = 1
            elif unit == 'atm':
                conv_ratio = 101325
            elif unit == 'bar':
                conv_ratio = 100000

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initiate the pressure tensor component lists
            Pxx, Pyy, Pzz, Pxy, Pxz, Pyz = [], [], [], [], [], []

            # Read the pressure tensor elements from data file
            with open(datafile, "r") as file:
                next(file)
                next(file)

                for _ in trange(steps, ncols=100, desc='Progress'):
                    line = file.readline()
                    step = list(map(float, line.split()))
                    Pxx.append(step[1]*conv_ratio)
                    Pyy.append(step[2]*conv_ratio)
                    Pzz.append(step[3]*conv_ratio)
                    Pxy.append(step[4]*conv_ratio)
                    Pxz.append(step[5]*conv_ratio)
                    Pyz.append(step[6]*conv_ratio)

            # Convert lists to numpy arrays
            Pxx = np.array(Pxx)
            Pyy = np.array(Pyy)
            Pzz = np.array(Pzz)
            Pxy = np.array(Pxy)
            Pxz = np.array(Pxz)
            Pyz = np.array(Pyz)

            # Generate the time array
            end_step = steps * timestep
            Time = np.linspace(0, end_step, num=steps, endpoint=False)

            viscosityEinstein = einstein(timestep, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz, volume, Temp, Time)
            avg_acf, viscosityGK = green_kubo(timestep, Pxy, Pxz, Pyz, volume, Temp)
            
            DataframeEinstein[f'Viscosity_{Run}'] = viscosityEinstein[:]*1000 
            DataframeGK[f'Viscosity_{Run}'] = viscosityGK[:]*1000

        except Exception as E:
            pass

    DataframeGK['Average'] = DataframeGK.mean(axis=1)
    DataframeGK['STD'] = DataframeGK.std(axis=1)
    DataframeEinstein['Average'] = DataframeEinstein.mean(axis=1)
    DataframeEinstein['STD'] = DataframeEinstein.std(axis=1)

    DataframeGKViscList_Average = DataframeGK['Average'].to_list()
    DataframeGKViscList_AverageSTD = DataframeGK['STD'].to_list()
    DataframeGKViscList_Average = [float(x) for x in DataframeGKViscList_Average]
    DataframeGKViscList_AverageSTD = [float(x) for x in DataframeGKViscList_AverageSTD]
    DataframeEinsteinList_Average = DataframeEinstein['Average'].to_list()
    DataframeEinsteinList_AverageSTD = DataframeEinstein['STD'].to_list()
    DataframeEinsteinList_Average = [float(x) for x in DataframeEinsteinList_Average]
    DataframeEinsteinList_AverageSTD = [float(x) for x in DataframeEinsteinList_AverageSTD]

    GKVisc = round((DataframeGKViscList_Average[-1]), 2)
    EinsteinVisc = round((DataframeEinsteinList_Average[-1]), 2)
    EinsteinUncert = DataframeEinsteinList_AverageSTD[-1]
    GKUncert = DataframeGKViscList_AverageSTD[-1]

    return GKVisc, EinsteinVisc, EinsteinUncert, GKUncert

def KTournament():
    pass