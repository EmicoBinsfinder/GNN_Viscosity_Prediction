"""
5th December 2023
Author: Egheosa Ogbomo

Script to automate:

- Generation of MMFF parameterised PDBs from GA generated SMILES strings via RDKit
- Generate packmol pdb with x number of molecule in XYZ sized box
- Make a new folder for every molecule being simulated (just generally organise files being generated)
- Submit a separate job submission for every molecule in the generation 
- Simulate at both 40C and 100C for VI calculation
- Only simulate molecules that haven't previously been simulated
- Prevent next generation of
- Create master list of:
    - Simulated molecules and their PDBs
    - SMILES strings
    - Calculated values 

- Make array job script to speed up simulations
- Give molecules appropriate names

- Three runs for viscosity simulation(?), take average of three runs, and the variance
between them, to get uncertainty values

- Need to update GA to move on to next molecule to pad out generation if it can't find 
a new mutation that hasn't already been checked

IDEA for checking other generative methods
- Compare to GA considering this is essentially a brute force method
"""

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

#import Genetic_Algorithm

# Function to run a command from a python script
def runcmd(cmd, verbose = False, *args, **kwargs):
    #bascially allows python to run a bash command, and the code makes sure 
    #the error of the subproceess is communicated if it fails
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

#Testing

smi = 'CCCCCOCCCCCCOC=CCC=CCCCOCC(=O)COCCC(C)=C(C)CO' # SMILES string

SMILESMol = Chem.MolFromSmiles(smi) # Create mol object
SMILESMol = Chem.AddHs(SMILESMol) # Need to make Hydrogens explicit

"""
Below we generate initial 3D conformation of molecule by calling embed method with ETKDG conformation
ETKDG algorithm for 3D coord generation takes into account experimentally witnessed preferred 
conformations such as preferred torsion angles and that aromatic rings are flat

Will probs reduce time taken to find a valid conformation

Useful link for write up:
blopig.com/blog/2016/06/advances-in-conformer-generation-etkdg-and-etdg/
"""
AllChem.EmbedMolecule.ETKDG(SMILESMol, AllChem.ETKDG()) 

# Initial parameters for conformer optimisation
MMFFSMILES = 1 
ConformAttempts = 1
MaxConformAttempts = 10

# Ensure that script continues to iterate until acceptable conformation found
while MMFFSMILES !=0 and ConformAttempts <= MaxConformAttempts: # Checking if molecule converged
    MMFFSMILES = Chem.rdForceFieldHelpers.MMFFOptimizeMolecule(SMILESMol, maxIters=1000)
    print(f'Number of Conformation Attempts: {ConformAttempts}')
    ConformAttempts += 1
    
# Record parameterised conformer as pdb to be used with packmol later 
Chem.MolToPDBFile(SMILESMol, 'C:/Users/eeo21/VSCodeProjects/GNN_Viscosity_Prediction/Test1.pdb')

#### CREATING THE MOLTEMPLATE FILE

#Build rdkit molecule from smiles and generate a conformer, using optimised SMILES string from above

Refresh = False # Overwrites files in a directory
Name = 'Test1' # Name of the moltemplate file that will be generated
LOPLS = True # Whether or not to use LOPLS, defaults to OPLS 
OPLSFeatPath = ''
LOPLSFeatPath = ''
Charge = True # Check net charge of molecule based on a charge dictionary (put in paper)


if Refresh and LOPLS:
    rdlt.generateFeatureDefn(args.refresh,'./lopls_lt.fdefn','./lopls_lt_dict.pkl')
elif args.refresh:
    rdlt.generateFeatureDefn(args.refresh,'./opls_lt.fdefn','./opls_lt_dict.pkl')

#Build a feature factory from the defintion file and assign all features
factory = Chem.ChemicalFeatures.BuildFeatureFactory(args.fdef)
features = factory.GetFeaturesForMol(SMILESMol)

#Use the features to assign an atom type property
[SMILESMol.GetAtomWithIdx(f.GetAtomIds()[0]).SetProp('AtomType',f.GetType()) for f in features];

#if lopls defitions are desired, redo the feature process
# overwrite atomtypes
if LOPLS:
    #print('loplsflag is {}'.format(loplsflag) )
    lfactory = Chem.ChemicalFeatures.BuildFeatureFactory(args.lfdef)
    lfeatures = lfactory.GetFeaturesForMol(m)
    #print(len(lfeatures))
    #for f in lfeatures:
    #    print(f.GetId(), f.GetFamily(), f.GetType(), f.GetAtomIds())
    [m.GetAtomWithIdx(f.GetAtomIds()[0]).SetProp('AtomType',f.GetType()) for f in lfeatures];
    #[print(at.GetProp('AtomType')) for at in m.GetAtoms()]

#find untyped atoms
#
failure = False
for at in m.GetAtoms():
    try:
        at.GetProp('AtomType')
    except KeyError:
        print("Atom {0} does not have an assigned atom type!".format(at.GetIdx()))
        failure = True
#if any failed to type, quit
if failure:
    sys.exit("""Refusing to write a .lt file without type assignments.
Check the SMARTS pattern that defines the expected atom type.""")


#basic output
writeHeader(args.name,args.loplsflag)
writeAtoms(m)
writeBonds(m)
writeFooter(args.name)

if args.charge:
    # Read charge dictionaries for testing
    opls_cdict = read_cdict('./opls_lt_dict.pkl')
    if args.loplsflag:
        lopls_cdict = read_cdict('./lopls_lt_dict.pkl')
        opls_cdict.update(lopls_cdict)

    sum_of_charges(m,opls_cdict)
