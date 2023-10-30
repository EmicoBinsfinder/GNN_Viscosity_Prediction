"""
Author: Egheosa Ogbomo
Date: 11/10/2023

An implementation of a genetic algoritm that:

- Allows user to generate varying lengths of molecules for optimisation
- Define their own fitness function (an ML algorithm or MD simulation framework)
- Remove molecules with undesired substructures from populations
- Define likelihood of different mutations/crossovers occuring
- Store SMILES strings and evaluations of valid(?) molecules from each generation in csv
- Allow users to define which elements and mutations are allowed from a database (which can be in csv form)
- Remove disconnected fragments
- Check for molecular similarity
- Illustrate mutated molecules between generations
- Plot an illustration of generated molecules as a word embedding
- Find most common substructure present within a generation of champion molecules to advise future generations
- Limit number of atoms generated by algorithm
- Generate options that conform to a specific type of base oils (e.g. PAO/silicone oils)

Possible Mutations
- Add Atom
- Replace Atom
- Change Bond Order
- Add Ring
- Add Fragment 
- Replace Fragment

CONSTRAINTS

- Allow only one mutation per generation

CONSIDERATIONS

- Think about using Python pickle structure to save molecules for speeding up molecule loading

- Best way to initialise population

- How to penalise long surviving molecules to encourage even greater diversity in later generations

- Need to determine how to handle crossovers

- How to know if something is a gas or liquid

- Are we only going to add to end points
    - Maybe a coinflip or some form of distribution to determine which alteration is made

- Decide a protocol for adding branches top 

- How to implement a validity check before evaluating the fitness of molecules in a generation and which validity checks
to implement
    - Can the molecule be parameterised?

- How to ensure that we are actually generating molecules that will be liquid at room temperature (how do we design 
gaseous lubricants)

- Need to decide best fingerprint for calculating molecular similarity

- Need to make sure that there are no duplicates in a given generation

- Need a parameter for elitism
"""

################# IMPORTS ###################

from rdkit import Chem
from rdkit.Chem import Draw
import rdkit
import random
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdDepictor
from random import choice as rnd
from rdkit.Chem.Draw import rdMolDraw2D
from MoleculeDifferenceViewer import view_difference

DrawingOptions.includeAtomNumbers=True
DrawingOptions.bondLineWidth=1.8
DrawingOptions.atomLabelFontSize=14

### Fragments
fragments = ['c1ccccc1', 'CCCC', 'CCCCC', 'CCCCCCC', 'CCCCCCCC', 'CCCCCCCCC', 'CCCCCCCCCC']
fragments = [Chem.MolFromSmiles(x) for x in fragments]

#print(Chem.GetPeriodicTable())

### ATOM NUMBERS
Atoms = ['C', 'O']
AtomMolObjects = [Chem.MolFromSmiles(x) for x in Atoms]
AtomicNumbers = []

# Getting Atomic Numbers for Addable Atoms
for Object in AtomMolObjects:
     for atom in Object.GetAtoms():
          AtomicNumbers.append(atom.GetAtomicNum())         

### BOND TYPES
BondTypes = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]

"""
1. Combine molecules into a singular SMILES object 
2. Create separate objects for each fragment in SMILES string 
3. Check bond types for each bond for each atom in each fragment, maybe create separate list containing atoms with each 
bond type
4. Check whether bonds are aromatic, will need to Kekulize SMILES strings to enable this
5. Check number of rings in the structure
5. Assign probablilities to each potential mutation
6. Select a mutation based on probabilities, taking into account information from starting structure (just start with 
random probabilities)
Steps for each Mutation
"""

def AddAtom(StartingMolecule, NewAtoms, showdiff=False):
    """
    Function that adds atom from a list of selected atoms to a starting molecule.

    Need to ensure that probability of this is zero if length of molecule is too short.

    Takes molecule, adds atom based on defined probabilities of position.

    Arguments:
        - StartingMolecule: SMILES String of Starting Molecule
        - NewAtoms: List of atoms that could be added to starting molecule
        - Show Difference: If True, shows illustration of changes to molecule
    """
    Starting_molecule_Unedited = StartingMolecule

    # Change to an editable object
    Starting_molecule = Chem.RWMol(Starting_molecule_Unedited)

    # Add selected atom from list of addable atoms 
    Starting_molecule.AddAtom(Chem.Atom(int(rnd(NewAtoms))))

    # Storing indexes of newly added atom and atoms from intial molecule
    frags = Chem.GetMolFrags(Starting_molecule)

    # Check which object is the newly added atom
    for ind, frag in enumerate(frags):
        print(ind, frag)
        if len(frag) == 1:
            #Store index of new atom in object
            NewAtomIdx = frags[ind]
        else:
            StartMolIdxs = frags[ind]
    
    Starting_molecule.AddBond(rnd(StartMolIdxs), NewAtomIdx[0], random.choice(BondTypes))

    #Starting_molecule.AddBond(, NewAtomIdx[0], random.choice(BondTypes))

    # SMILES string chemical validity check
    Mut_Mol = Starting_molecule.GetMol()
    Mut_Mol_Sanitized = Chem.SanitizeMol(Mut_Mol, catchErrors=True) 

    if Mut_Mol_Sanitized == rdkit.Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
        print('Validity Check Passed')
    else:
        print('Validity Check Failed')
   
    if showdiff:
        view_difference(Starting_molecule_Unedited, Mut_Mol)

### TESTING AddAtom Function
#AddAtom(rnd(fragments), AtomicNumbers, showdiff=True)

def ReplaceAtom(StartingMolecule, NewAtoms, fromAromatic=False, showdiff=False):
    """
    Function to remove atom from a selected list of atoms from a starting molecule.
    
    Args:
    - StartingMolecule: SMILES String of Starting Molecule
    - NewAtoms: List of atoms that could be added to starting molecule
    - FromAromatic: If True, will remove atoms from aromatic rings 
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
            print(f'Number of Bonds atom has = {len(atom.GetBonds())}')

    #Select atom to be deleted from list of atom indexes, check that this list is greater than 0
    if len(AtomIdxs) > 0:
        ReplaceAtomIdx = rnd(AtomIdxs)
    else:
        print('Empty Atom Index List')

    #Exclude replaced atom type from list of atoms to do replacing with
    ReplaceAtomType = StartingMoleculeUnedited.GetAtomWithIdx(ReplaceAtomIdx).GetAtomicNum()
    AtomReplacements = [x for x in NewAtoms if x != ReplaceAtomType]

    StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)

    #Replace atom
    ReplacementAtom = rnd(AtomReplacements)
    StartingMolecule.ReplaceAtom(ReplaceAtomIdx, Chem.Atom(ReplacementAtom), preserveProps=True)

    print(f'{StartingMoleculeUnedited.GetAtomWithIdx(ReplaceAtomIdx).GetSymbol()}\
          replaced with {Chem.Atom(ReplacementAtom).GetSymbol()}')
    
    Mut_Mol = StartingMolecule.GetMol()

    #Sanitize molecule
    Mut_Mol_Sanitized = Chem.SanitizeMol(Mut_Mol, catchErrors=True)

    if showdiff:
        view_difference(StartingMoleculeUnedited, Mut_Mol)

### TESTING ReplaceAtom Function
#ReplaceAtom(rnd(fragments), AtomicNumbers, fromAromatic=False, showdiff=True)



def ReplaceBond(StartingMolecule, Bonds, showdiff=False):
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


Bonds = BondTypes

StartingMoleculeUnedited = rnd(fragments)

BondIdxs = []
for bond in StartingMoleculeUnedited.GetBonds():
        # Check if atom is Aromatic
        if bond.GetIsAromatic():
            continue
        else:
            BondIdxs.append(bond.GetIdx())

print(f'Number of Bonds molecules has = {len(StartingMoleculeUnedited.GetBonds())}')
#Select atom to be deleted from list of atom indexes, check that this list is greater than 0

print(BondIdxs)

if len(BondIdxs) > 0:
    ReplaceBondIdx = rnd(BondIdxs)
else:
    print('Empty Bond Index List')

ReplaceBondType = StartingMoleculeUnedited.GetBondWithIdx(ReplaceBondIdx).GetBondType()

BondReplacements = [x for x in Bonds if x != ReplaceBondType]
print(BondReplacements)

StartingMolecule = Chem.RWMol(StartingMoleculeUnedited)

print(type(rnd(BondReplacements)))

StartingMolecule.ReplaceBond(ReplaceBondIdx, rnd(BondReplacements), preserveProps=True)

# print(f'{StartingMoleculeUnedited.GetAtomWithIdx(ReplaceAtomIdx).GetSymbol()}\
#         replaced with {Chem.Atom(ReplacementAtom).GetSymbol()}')

Mut_Mol = StartingMolecule.GetMol()

#Sanitize molecule
Mut_Mol_Sanitized = Chem.SanitizeMol(Mut_Mol, catchErrors=True)

view_difference(StartingMoleculeUnedited, Mut_Mol)








