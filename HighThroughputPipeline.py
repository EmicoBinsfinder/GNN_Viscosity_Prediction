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
import sys
import Genetic_Algorithm_Functions as GAF
import ast
import shlex
import os

#import Genetic_Algorithm

#Testing
smi = 'C=CC=COCCOCCCOCC=CCCCC=CCC=CO' # SMILES string
Name = 'Test1' # Name of the molecule that will be generated, will need to use this in system.lt file
LOPLS = True
PYTHONPATH = 'C:/Users/eeo21/AppData/Local/Programs/Python/Python310/python.exe'

if LOPLS:
    COMMAND = f'c:/Users/eeo21/VSCodeProjects/GNN_Viscosity_Prediction/rdlt.py --smi "{smi}" -n {Name} -l -c'
else:
    COMMAND = f'c:/Users/eeo21/VSCodeProjects/GNN_Viscosity_Prediction/rdlt.py --smi "{smi}" -n {Name} -c'

STARTINGDIR = f'c:/Users/eeo21/VSCodeProjects/GNN_Viscosity_Prediction/'

GAF.runcmd(f'{PYTHONPATH} {COMMAND} > {Name}.lt')

def GetMolCharge(PATH):
    """
    Retreive molecule charge from Moltemplate file
    """

    with open(PATH, 'r') as file:
        data = file.readlines()
        charge = data[-1].split('#')[-1].split('\n')[0] #Horrendous way of getting the charge

    return charge

charge = GetMolCharge(f'{os.getcwd()}/{Name}.lt')
print(float(charge))

GAF.runcmd(f'mkdir Molecules')
os.chdir(os.path.join(STARTINGDIR, 'Molecules'))
GAF.runcmd(f'mkdir {Name}')
os.chdir(os.path.join(os.getcwd(), f'{Name}'))

if os.path.exists(f"{os.path.join(os.getcwd(), f'{Name}.lt')}"):
    print('Specified file already exists in this location!')
else:
    os.rename(f"{os.path.join(STARTINGDIR, f'{Name}.lt')}", f"{os.path.join(os.getcwd(), f'{Name}.lt')}")

os.chdir(STARTINGDIR) # Return to starting directory

