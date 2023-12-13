from rdkit import Chem
from rdkit.Chem import Draw
import rdkit
from random import sample
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdDepictor
from random import choice as rnd
import random
from rdkit.Chem.Draw import rdMolDraw2D
from MoleculeDifferenceViewer import view_difference
from copy import deepcopy
from operator import itemgetter
import subprocess
import ast
import requests
import os

def MolCheckandPlot(StartingMoleculeUnedited, StartingMolecule, showdiff, Verbose=False):
    
    Mut_Mol = StartingMolecule.GetMol()
    MutMolSMILES = Chem.MolToSmiles(Mut_Mol)

    Mut_Mol_Sanitized = Chem.SanitizeMol(Mut_Mol, catchErrors=True) 

    if len(Chem.GetMolFrags(Mut_Mol)) != 1:
        if Verbose:
            print('Fragmented result, trying new mutation')
        Mut_Mol = None
        Mut_Mol_Sanitized = None
        MutMolSMILES = None

        if showdiff:
            view_difference(StartingMoleculeUnedited, Mut_Mol)
    
    elif Mut_Mol_Sanitized != rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
        if Verbose:
            print('Validity Check Failed')
        Mut_Mol = None
        Mut_Mol_Sanitized = None
        MutMolSMILES = None

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
    StartingMoleculeUnedited = StartingMolecule

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
    StartingMoleculeUnedited = StartingMolecule

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
        Mut_Mol = None
        Mut_Mol_Sanitized = None
        MutMolSMILES = None
    
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

    StartingMoleculeUnedited = StartingMolecule

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
        Mut_Mol = None
        Mut_Mol_Sanitized = None
        MutMolSMILES = None

    return Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES, StartingMoleculeUnedited

def AddFragment(StartingMolecule, Fragment, InsertStyle = 'Within', showdiff=True, Verbose=False):
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
    Fragment = Fragment

    try:
        #Always check if fragment is a cyclic (benzene) molecule
        if len(Fragment.GetAromaticAtoms()) == len(Fragment.GetAtoms()):
            if Verbose:
                print('Fragment aromatic, inappropriate function')
            Mut_Mol = None
            Mut_Mol_Sanitized = None
            MutMolSMILES = None
        elif len((StartingMoleculeUnedited.GetAromaticAtoms())) == len(StartingMoleculeUnedited.GetAtoms()):
            if Verbose:
                print('Starting molecule is completely aromatic')
            Mut_Mol = None
            Mut_Mol_Sanitized = None
            MutMolSMILES = None
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
                    Mut_Mol = None
                    Mut_Mol_Sanitized = None
                    MutMolSMILES = None

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
                Mut_Mol = None
                Mut_Mol_Sanitized = None
                MutMolSMILES = None
    except:
        if Verbose:
            print('Index error, starting molecule probably too short, trying different mutation')
        Mut_Mol = None
        Mut_Mol_Sanitized = None
        MutMolSMILES = None
      
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

    #try:
    # Store indexes of atoms in molecule
    AtomIdxs = []

    # Check if starting molecule is completely aromatic
    for atom in StartingMoleculeUnedited.GetAtoms():
        if len((StartingMoleculeUnedited.GetAromaticAtoms())) == len(StartingMoleculeUnedited.GetAtoms()):
            print('Starting molecule is completely aromatic')
            Mut_Mol = None
            Mut_Mol_Sanitized = None
            MutMolSMILES = None
            
        #elif 

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
def InsertBenzene(StartingMolecule, AromaticMolecule, InsertStyle='Within', showdiff=False, Verbose=False):

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

    StartingMoleculeUnedited = StartingMolecule
    Fragment = AromaticMolecule

    #Always check if fragment or starting molecule is a cyclic (benzene) molecule

    try:
        if len(Fragment.GetAromaticAtoms()) != len(Fragment.GetAtoms()):
            if Verbose:
                print('Fragment is not cyclic')
            Mut_Mol = None
            Mut_Mol_Sanitized = None
            MutMolSMILES = None
        
        elif len((StartingMoleculeUnedited.GetAromaticAtoms())) == len(StartingMoleculeUnedited.GetAtoms()):
            if Verbose:
                print('Starting molecule is completely aromatic')
            Mut_Mol = None
            Mut_Mol_Sanitized = None
            MutMolSMILES = None

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
                    Mut_Mol = None
                    Mut_Mol_Sanitized = None
                    MutMolSMILES = None
                
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
                Mut_Mol = None
                Mut_Mol_Sanitized = None
                MutMolSMILES = None
    except:
        print('Index error, starting molecule probably too short, trying different mutation')
        Mut_Mol = None
        Mut_Mol_Sanitized = None
        MutMolSMILES = None

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

    else: 
        #Mutation == 'AddFragment'
        InsertStyle = rnd(['Within', 'Egde'])
        result = AddFragment(StartingMolecule, rnd(Fragments), InsertStyle=InsertStyle, showdiff=showdiff)
    
    # else:
    #     InsertStyle = rnd(['Within', 'Egde'])
    #     result = InsertBenzene(StartingMolecule, AromaticMolecule, showdiff=showdiff, InsertStyle=InsertStyle)
    
    return result

def CheckSubstruct(MutMol):
    ### Check for unwanted substructures

    #Checking for sequential oxgens
    SingleBondOxygens = MutMol.HasSubstructMatch(Chem.MolFromSmarts('OO')) 
    DoubleBondOxygens = MutMol.HasSubstructMatch(Chem.MolFromSmarts('O=O'))
    DoubleCDoubleO = MutMol.HasSubstructMatch(Chem.MolFromSmarts('C=C=O'))    
    DoubleCDoubleC = MutMol.HasSubstructMatch(Chem.MolFromSmarts('C=C=C'))    

    # Check for sequence of single or double bonded oxygens
    if SingleBondOxygens or DoubleBondOxygens or DoubleCDoubleO or DoubleCDoubleC:
        print('Sequential Oxygens Found, returning empty object')
        return True
    else:
        return False

def fitfunc(MoleculeSMILES, generation):
    return random.randint(1, generation*5000)

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

def smiles_to_iupac(smiles):
    CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"
    rep = "iupac_name"
    url = CACTUS.format(smiles, rep)
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def CheckMoveFile(Name, STARTINGDIR, FileType, CWD):
    if os.path.exists(f"{os.path.join(CWD, f'{Name}.{FileType}')}"):
        print('Specified file already exists in this location, removing generated file.')
        os.remove(f'{STARTINGDIR}/{Name}.{FileType}')
    else:
        os.rename(f"{os.path.join(STARTINGDIR, f'{Name}.{FileType}')}", f"{os.path.join(CWD, f'{Name}.{FileType}')}")

def MakePackmolFile(Name, CWD):
    if os.path.exists(f"{os.path.join(CWD, f'{Name}.inp')}"):
        print('Specified file already exists in this location, removing generated file.')
    else:
        with open(os.path.join(CWD, f'{Name}.inp'), 'x') as file:
            file.write(f"""
tolerance 2.0
output {Name}_PackmolFile.pdb

filetype pdb

structure {Name}.pdb
number 50 
inside cube 0. 0. 0. 50.
end structure""")

def MakeMoltemplateFile(Name, CWD):
    if os.path.exists(f"{os.path.join(CWD, f'{Name}._system.lt')}"):
        print('Specified file already exists in this location, removing generated file.')
    else:
        with open(os.path.join(CWD, f'{Name}_system.lt'), 'x') as file:
                    file.write(f"""
import "{Name}.lt"  # <- defines the molecule type.

# Periodic boundary conditions:
write_once("Data Boundary") {{
   0.0  50.00  xlo xhi
   0.0  50.00  ylo yhi
   0.0  50.00  zlo zhi
}}

ethylenes = new {Name} [50]

""")