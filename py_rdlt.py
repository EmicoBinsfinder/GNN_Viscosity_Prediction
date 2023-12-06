import rdlt
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
import time
import sys
import Genetic_Algorithm_Functions as GAF

smi = 'CCCC=CCCCCCC=CCC=CCCC=CCCCC=CCCCC=CCC=CCCCC' # SMILES string

SMILESMol = Chem.MolFromSmiles(smi) # Create mol object
SMILESMol = Chem.AddHs(SMILESMol) # Need to make Hydrogens explicit

AllChem.EmbedMolecule(SMILESMol, AllChem.ETKDG()) 

#### CREATING THE MOLTEMPLATE FILE

#Build rdkit molecule from smiles and generate a conformer, using optimised SMILES string from above
Name = 'Test1' # Name of the molecule that will be generated, will need to use this in system.lt file

LOPLS = False # Whether or not to use LOPLS, defaults to OPLS 
OPLSFeatPath = 'C:/Users/eeo21/VSCodeProjects/GNN_Viscosity_Prediction/opls_lt.fdefn'
LOPLSFeatPath = 'C:/Users/eeo21/VSCodeProjects/GNN_Viscosity_Prediction/lopls_lt.fdefn'
Charge = True # Check net charge of molecule based on a charge dictionary (put in paper)

"""
Added the OPLS/LOPLS check to Genetic algorithm pipeline
This basically serves as another check for unfavourable structures 
We should store molecules generated in this way, could be useful for training NNs, for
simulating with other forcefields, or gaining understanding of unwanted structures
"""

# Need to do OPLS first as LOPLS parameter file only has extra LOPLS terms, not standalone OPLS terms
factory = Chem.ChemicalFeatures.BuildFeatureFactory(OPLSFeatPath)

# Assign features from feature factor to molecule
features = factory.GetFeaturesForMol(SMILESMol)

#Use the features to assign an atom type property to atoms in molecule
[SMILESMol.GetAtomWithIdx(f.GetAtomIds()[0]).SetProp('AtomType',f.GetType()) for f in features]

# Adding LOPLS definitions to molecule, can check 
if LOPLS:
    lfactory = Chem.ChemicalFeatures.BuildFeatureFactory(LOPLSFeatPath)
    lfeatures = lfactory.GetFeaturesForMol(SMILESMol)
    [SMILESMol.GetAtomWithIdx(f.GetAtomIds()[0]).SetProp('AtomType',f.GetType()) for f in lfeatures]

#Check for untyped atoms
failure = False
for atom in SMILESMol.GetAtoms():
    try:
        atom.GetProp('AtomType')
    except KeyError:
        print("Atom {} does not have an assigned atom type!".format(atom.GetIdx()))
        print(atom.GetSymbol())
        failure = True
#if any failed to type, quit
if failure:
    sys.exit("""Refusing to write a .lt file without type assignments.
Check the SMARTS pattern that defines the expected atom type.""")

#basic output
rdlt.writeHeader(Name, LOPLS)
rdlt.writeAtoms(SMILESMol)
rdlt.writeBonds(SMILESMol)
rdlt.writeFooter(Name)

#Check if final molecule has a charge, store this information and penalise molecules during selection

if Charge:
    # Read charge dictionaries for testing
    opls_cdict = rdlt.read_cdict('./opls_lt_dict.pkl')
    if LOPLS:
        lopls_cdict = rdlt.read_cdict('./lopls_lt_dict.pkl')
        opls_cdict.update(lopls_cdict)

    rdlt.sum_of_charges(SMILESMol,opls_cdict)