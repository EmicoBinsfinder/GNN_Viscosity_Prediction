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
- Impose maximum number of heavy atoms
- Recreate and visualise certain mutations
- Assign molecule ID to track mutation history through generation

Possible Mutations
- Add Atom
- Replace Atom
- Change Bond Order
- Add Fragment 
- Delete atom
- Replace Fragment

CONSIDERATIONS

- How to initialise and change population of fragments for remove fragments module

- Maybe a way to store the history of mutations made to a molecule, so that we can adjust (reduce) probabilities of certain
mutations occurring and reoccuring based on past events, increasing range of molecules chekced out.

- Can we keep track of modifications, and track how each legal modification to impacts (i.e. increases or decreases)
visosity/conductivity to determine which modifications are best

- Best way to initialise population

- How to penalise long surviving molecules to encourage even greater diversity in later generations

- How to know if something is a gas or liquid

- Decide a protocol for adding branches top 

- How to implement a validity check before evaluating the fitness of molecules in a generation and which validity checks
to implement

- How to replace

- How to ensure that we are actually generating molecules that will be liquid at room temperature (how do we design 
gaseous lubricants)

- Need to decide best fingerprint for calculating molecular similarity

- Make calls to initialise LAMMPS call and get viscosity value from the simulation

- Varying (decreasing) elitism factor to further increase novelty of molecules

Look at COMPASS 3
"""

############### ENVIRONMENT SETUP ############
import subprocess
def runcmd(cmd, verbose = False, *args, **kwargs):
    #bascially allows python to run a bash command, and the code makes sure 
    #the error of the subproceess is communicated if it fails
    process = subprocess.run(
        cmd,
        text=True,
        shell=True)
    
    return process

# runcmd('module load anaconda3/personal')
# runcmd('source activate HTVS')

# runcmd('export PATH="$PATH:/rds/general/user/eeo21/home/moltemplate/moltemplate/moltemplate/scripts"')
# runcmd('export PATH="$PATH:/rds/general/user/eeo21/home/moltemplate/moltemplate/moltemplate/"')

# runcmd('export PATH="$PATH:/rds/general/user/eeo21/home/Packmol/packmol-20.14.2/"')

################# IMPORTS ###################
import Genetic_Algorithm_Functions as GAF
from rdkit import Chem
from rdkit.Chem import Draw
from random import sample
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdDepictor
from random import choice as rnd
import sys
from rdkit.Chem.Draw import rdMolDraw2D
from MoleculeDifferenceViewer import view_difference
from copy import deepcopy
from operator import itemgetter
import os
import pandas as pd
import numpy

DrawingOptions.includeAtomNumbers=True
DrawingOptions.bondLineWidth=1.8
DrawingOptions.atomLabelFontSize=14

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

# GENETIC ALGORITHM HYPERPARAMETERS
Silent = True # Edit outputs to only print if this flag is False
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
MaxGenerations = 4
NumGenerations = 1
MaxMutationAttempts = 200
Fails = 0

#PYTHONPATH = 'C:/Users/eeo21/AppData/Local/Programs/Python/Python310/python.exe'
STARTINGDIR = deepcopy(os.getcwd())
PYTHONPATH = 'python3'
GAF.runcmd(f'mkdir Molecules')
os.chdir(os.path.join(os.getcwd(), 'Molecules'))
GAF.runcmd(f'mkdir Generation_1')
os.chdir(STARTINGDIR)

# Master Dataframe where molecules from all generations will be stored
MoleculeDatabase = pd.DataFrame(columns=['SMILES', 'MolObject', 'MutationList', 'HeavyAtoms', 'ID', 'Charge', 'MolMass', 'Predecessor', 'Score'])

# Initialise population 
while len(MoleculeDatabase) < GenerationSize:
    # Return to starting directory
    os.chdir(STARTINGDIR)

    print('\n###########################################################')
    print(f'Attempt number: {FirstGenerationAttempts}')
    StartingMolecule = rnd(fragments) #Select starting molecule

    Mutation = rnd(Mutations)
    AromaticMolecule = fragments[-1]

    # Perform mutation 
    result = GAF.Mutate(StartingMolecule, Mutation, AromaticMolecule, AtomicNumbers, BondTypes, Atoms, showdiff, fragments)

    # Implement checks based on predetermined criteria (MolLength, Illegal Substructs etc.)
    if GAF.GenMolChecks(result, GenerationMolecules, MaxNumHeavyAtoms, MinNumHeavyAtoms) == None:
        MutMol = None

    else:
        HeavyAtoms = result[0].GetNumHeavyAtoms() # Get number of heavy atoms in molecule
        MutMol = result[0] # Get Mol object of mutated molecule
        MolMass = GAF.GetMolMass(MutMol) # Get estimate of of molecular mass 
        MutMolSMILES = result[2] # SMILES of mutated molecule
        Predecessor = result[3] # Get history of last two mutations performed on candidate
        ID = IDcounter

        # Try Making all the files 
        try:
            Name = f'Generation_1_Molecule_{ID}' # Set name of Molecule as its SMILES string

            # Set feature definition file path to OPLS or LOPLS depending on user choice 
            if LOPLS:
                LTCOMMAND = f"{os.path.join(os.getcwd(), 'rdlt.py')} --smi {MutMolSMILES} -n {Name} -l -c"
            else:
                LTCOMMAND = f"{os.path.join(os.getcwd(), 'rdlt.py')} --smi {MutMolSMILES} -n {Name} -c"
            
            #Attempt to parameterise with OPLS
            GAF.runcmd(f'{PYTHONPATH} {LTCOMMAND} > {STARTINGDIR}/{Name}.lt')

            #Get molecule charge
            charge = GAF.GetMolCharge(f'{os.getcwd()}/{Name}.lt')

            #If successful, generate a PDB of molecule to use with Packmol
            GAF.GeneratePDB(MutMolSMILES, PATH=os.path.join(STARTINGDIR, f'{Name}.pdb'))

            # Go into directory for this generation
            os.chdir(os.path.join(STARTINGDIR, 'Molecules', 'Generation_1'))
            
            # Make a directory for the current molecule if it can be parameterised 
            GAF.runcmd(f'mkdir {Name}')

            # Enter molecule specific directory
            os.chdir(os.path.join(os.getcwd(), f'{Name}'))

            #Check if file has already been made, skip if so, being sure not to make duplicate, otherwise move file to correct directory
            CWD = os.getcwd() #Need to declare otherwise will get CWD from location function is being called from
            GAF.CheckMoveFile(Name, STARTINGDIR, 'lt', CWD)
            GAF.CheckMoveFile(Name, STARTINGDIR, 'pdb', CWD)

            # Make packmol files
            GAF.MakePackmolFile(Name, CWD)

            # Make Moltemplate files
            GAF.MakeMoltemplateFile(Name, CWD)

            # Make LAMMPS files
            GAF.MakeLAMMPSFile(Name, CWD)

            if PYTHONPATH == 'python3':
                GAF.runcmd(f'packmol < {Name}.inp')
                GAF.runcmd(f'moltemplate.sh -pdb {Name}_PackmolFile.pdb {Name}_system.lt')

                # Check that Moltemplate has generated all necessary files 
                assert os.path.exists(os.path.join(CWD, f'{Name}_system.in.settings')), 'Settings file not generated'                 
                assert os.path.exists(os.path.join(CWD, f'{Name}_system.in.charges')), 'Charges file not generated'
                assert os.path.exists(os.path.join(CWD, f'{Name}_system.data')), 'Data file not generated'                    
                assert os.path.exists(os.path.join(CWD, f'{Name}_system.in.init')), 'Init file not generated'                

            # Update Molecule database
            MoleculeDatabase = GAF.DataUpdate(MoleculeDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
                                              MutationList=[None, Mutation], ID=Name, Charge=charge, MolMass=MolMass, Predecessor=Predecessor)
           
            # Generate list of molecules to simulate in this generation
            FirstGenSimList.append(Name)
            print(f'Final Molecule SMILES: {MutMolSMILES}') 
            
            IDcounter +=1

        except Exception as E:
            print(E)
            #if type(E) == AssertionError:
            #  sys.exit()
            continue     
    FirstGenerationAttempts += 1

# Run MD simulations and retreive performance 

Generation = 1
CWD = os.path.join(STARTINGDIR, 'Molecules', f'Generation_{Generation}')
# Create and run array job for 40C viscosity
GAF.CreateArrayJob(STARTINGDIR, CWD, Generation=1, SimName='313K.lammps')

# Create and run array job for 100C viscosity
GAF.CreateArrayJob(STARTINGDIR, CWD, Generation=1, SimName='373K.lammps')

if PYTHONPATH == 'python3':
    GAF.runcmd(f'qsub {os.path.join(CWD, "313K.lammps.pbs")}')

if PYTHONPATH == 'python3':
    GAF.runcmd(f'qsub {os.path.join(CWD, "373K.lammps.pbs")}')

# Here is where we will get the various values generated from the MD simulations
for Molecule in FirstGenSimList:
    # Create a function to wait until all simulations from this generation are finished
    Score = GAF.fitfunc(Molecule, Generation=1)
    IDNumber = int(Molecule.split('_')[-1])
    MoleculeDatabase.loc[IDNumber, 'Score'] = Score

# Create dictionary for comparison
GenerationMolecules = pd.Series(MoleculeDatabase.Score.values, index=MoleculeDatabase.ID).to_dict()

# Sort dictiornary according to target properties
ScoreSortedMolecules = sorted(GenerationMolecules.items(), key=lambda item:item[1], reverse=True)

#Convert tuple elements in sorted list back to lists 
ScoreSortedMolecules = [list(x) for x in ScoreSortedMolecules]

# Constructing entries for use in subsequent generation
for entry in ScoreSortedMolecules:
    Key = int(entry[0].split('_')[-1])
    entry.insert(1, MoleculeDatabase.iloc[Key]['MolObject'])
    entry.insert(2, MoleculeDatabase.iloc[Key]['MutationList'])
    entry.insert(3, MoleculeDatabase.iloc[Key]['HeavyAtoms'])

GenerationMolecules = ScoreSortedMolecules[:NumElite]

# QSUB each of the simulations based on file names in FirstGenSim list in array job
# Add some form of wait condition to allow all simulations to finish before next steps
    # This could just be a read of the qstat to see how far the array job has progressed
# Hope that all 50 work haha
# Retrieve the viscosities and the density from
# Perform calculation of VI 
# Add values for the density, each of the calculated dynamic viscosities and the VI to master dataframe

MoleculeDatabase.to_excel(f'{STARTINGDIR}/MoleculeDatabase.xlsx')

################################## Subsequent generations #################################################
# for generation in range(2, MaxGenerations + 1):
#     GenerationTotalAttempts = 0
#     GenSimList = []

#     os.chdir(STARTINGDIR)
#     # Store x best performing molecules (x=NumElite in list for next generation, without mutating them)
#     GenerationMoleculeList = ScoreSortedMolecules[:NumElite]
#     os.chdir(os.path.join(os.getcwd(), 'Molecules')) 
#     GAF.runcmd(f'mkdir Generation_{generation}')
#     os.chdir(STARTINGDIR)

#     for i, entry in enumerate(ScoreSortedMolecules): #Start by mutating best performing molecules from previous generation and work down
#         MutMol = None
#         attempts = 0

#         # Stop appending mutated molecules once generation reaches desired size
#         if len(GenerationMoleculeList) == GenerationSize:
#             break

#         # Attempt mutation on each molecule, not moving on until a valid mutation has been suggested
#         while MutMol == None:
#             attempts += 1
#             GenerationTotalAttempts += 1

#             # Limit number of attempts at mutation, if max attempts exceeded, break loop to attempt on next molecule
#             if attempts >= MaxMutationAttempts:
#                 Fails += 1 
#                 break

#             # Starting Molecule mol object
#             StartingMolecule = entry[1]
#             # List containing last two successful mutations performed on molecule
#             PreviousMutations = entry[2]
#             # Number of heavy atoms
#             NumHeavyAtoms = entry[3]
#             # Molecule ID
#             Name = f'Generation_{generation}_Molecule_{IDcounter}'

#             if NumHeavyAtoms > 42:
#                 MutationList = ['RemoveAtom', 'ReplaceAtom', 'ReplaceBond']
#             else:
#                 MutationList = Mutations 
            
#             print(f'\n#################################################################\nNumber of attempts: {attempts}')
#             print(f'Total Mutation Attempts: {GenerationTotalAttempts}')
#             print(f'GENERATION: {generation}')

#             # Randomly select a mutation, here is where we can set mutation probabilities
#             Mutation = rnd(MutationList)

#             # Initialise Aromatic Ring
#             AromaticMolecule = fragments[-1]

#             #Perform mutation, return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited
#             result = GAF.Mutate(StartingMolecule, Mutation, AromaticMolecule, AtomicNumbers, BondTypes, Atoms, showdiff, fragments)
            
#             if GAF.GenMolChecks(result, GenerationMolecules, MaxNumHeavyAtoms, MinNumHeavyAtoms) == None:
#                 MutMol = None
        
#             else:
#                 HeavyAtoms = result[0].GetNumHeavyAtoms() # Get number of heavy atoms in molecule
#                 MutMol = result[0] # Get Mol object of mutated molecule
#                 MolMass = GAF.GetMolMass(MutMol) # Get estimate of of molecular mass 
#                 MutMolSMILES = result[2] # SMILES of mutated molecule
#                 Predecessor = result[3] # Get history of last two mutations performed on candidate

#                 print(f'Number of Heavy Atoms after mutation: {NumHeavyAtoms}')                

#                 # Update previous mutations object
#                 PreviousMutations.pop(0)
#                 PreviousMutations.append(Mutation)

#                 print(f'Final SMILES: {result[2]}')

#                 try: # Try to generate all necessary files to simulate molecule
#                     # Set feature definition file path to OPLS or LOPLS depending on user choice 
#                     if LOPLS:
#                         LTCOMMAND = f"{os.path.join(os.getcwd(), 'rdlt.py')} --smi {MutMolSMILES} -n {Name} -l -c"
#                     else:
#                         LTCOMMAND = f"{os.path.join(os.getcwd(), 'rdlt.py')} --smi {MutMolSMILES} -n {Name} -c"
                    
#                     # Return to starting directory
#                     os.chdir(STARTINGDIR) 
                    
#                     #Attempt to parameterise with (L)OPLS
#                     GAF.runcmd(f'{PYTHONPATH} {LTCOMMAND} > {STARTINGDIR}/{Name}.lt')

#                     #Get molecule charge
#                     charge = GAF.GetMolCharge(f'{os.getcwd()}/{Name}.lt')
#                     print(f'OPLS Molecule Charge is: {float(charge)}')

#                     #If successful, generate a PDB of molecule to use with Packmol
#                     GAF.GeneratePDB(MutMolSMILES, PATH=os.path.join(STARTINGDIR, f'{Name}.pdb'))

#                     # Go into directory for this generation
#                     os.chdir(os.path.join(STARTINGDIR, 'Molecules', f'Generation_{generation}'))
                    
#                     # Make a directory for the current molecule if it can be parameterised 
#                     GAF.runcmd(f'mkdir {Name}')

#                     # Enter molecule specific directory
#                     os.chdir(os.path.join(os.getcwd(), f'{Name}'))

#                     #Check if file has already been made, skip if so, being sure not to make duplicate, otherwise move file to correct directory
#                     CWD = os.getcwd()
#                     GAF.CheckMoveFile(Name, STARTINGDIR, 'lt', CWD)
#                     GAF.CheckMoveFile(Name, STARTINGDIR, 'pdb', CWD)

#                     # Make packmol files
#                     GAF.MakePackmolFile(Name, CWD)

#                     # Make Moltemplate files
#                     GAF.MakeMoltemplateFile(Name, CWD)

#                     # Make LAMMPS files
#                     GAF.MakeLAMMPSFile(Name, CWD)

#                     if PYTHONPATH == 'python3':
#                         GAF.runcmd(f'packmol < {Name}.inp') # Run packmol in command line
#                         GAF.runcmd(f'moltemplate.sh -pdb {Name}_PackmolFile.pdb {Name}_system.lt') # Run moltemplate in command line

#                         # Check that Moltemplate has generated all necessary files 
#                         assert os.path.exists(os.path.join(CWD, f'{Name}_system.in.settings')), 'Settings file not generated'                 
#                         assert os.path.exists(os.path.join(CWD, f'{Name}_system.in.charges')), 'Charges file not generated'
#                         assert os.path.exists(os.path.join(CWD, f'{Name}_system.data')), 'Data file not generated'                    
#                         assert os.path.exists(os.path.join(CWD, f'{Name}_system.in.init')), 'Init file not generated'                

#                     # Return to starting directory
#                     os.chdir(STARTINGDIR)

#                     # Add candidate and it's data to master list
#                     MoleculeDatabase = GAF.DataUpdate(MoleculeDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
#                                             MutationList=PreviousMutations, ID=Name, Charge=charge, MolMass=MolMass, Predecessor=Predecessor)

#                     # Generate list of molecules to simulate in this generation
#                     GenSimList.append(Name)
#                     print(f'Final Molecule SMILES: {MutMolSMILES}') 
#                     GenerationMoleculeList.append([result[2], result[0], PreviousMutations, NumHeavyAtoms, ID, MolMass, Predecessor])
#                     IDcounter += 1

#                 except Exception as E:
#                     print(E)
#                     if type(E) == AssertionError:
#                         sys.exit()
#                     continue   

#         MoleculeDatabase.to_excel(f'{STARTINGDIR}/MoleculeDatabase.xlsx')
#         # # Tasks to perform at end of every generation (i.e get best performing molecules from this generation)
#         # # Simulate molecules that haven't been yet been simulated

#     #This is where we should call the simulation script
#     for Molecule in GenSimList:
#         Generation = generation
#         CWD = os.path.join(STARTINGDIR, 'Molecules', f'Generation_{Generation}')
#         # Create array job for 40C viscosity
#         GAF.CreateArrayJob(STARTINGDIR, CWD, generation, SimName='313K.lammps')
#         # Create array jpb for 100C viscosity
#         GAF.CreateArrayJob(STARTINGDIR, CWD, generation, SimName='373K.lammps')
#         Score = GAF.fitfunc(Molecule, Generation=generation)
#         IDNumber = int(Molecule.split('_')[-1])
#         MoleculeDatabase.loc[IDNumber, 'Score'] = Score

#     # Create dictionary for comparison
#     GenerationMolecules = pd.Series(MoleculeDatabase.Score.values, index=MoleculeDatabase.ID).to_dict()

#     # Sort dictiornary according to target properties
#     ScoreSortedMolecules = sorted(GenerationMolecules.items(), key=lambda item:item[1], reverse=True)

#     #Convert tuple elements in sorted list back to lists 
#     ScoreSortedMolecules = [list(x) for x in ScoreSortedMolecules]

#     # Constructing entries for use in subsequent generation
#     for entry in ScoreSortedMolecules:
#         Key = int(entry[0].split('_')[-1])
#         entry.insert(1, MoleculeDatabase.iloc[Key]['MolObject'])
#         entry.insert(2, MoleculeDatabase.iloc[Key]['MutationList'])
#         entry.insert(3, MoleculeDatabase.iloc[Key]['HeavyAtoms'])
    
#     MoleculeDatabase.to_excel(f'{STARTINGDIR}/MoleculeDatabase.xlsx')

# print(len(GenerationMoleculeList))
# print(f'Number of failed mutations: {Fails}')
# print(len(ScoreSortedMolecules[:NumElite]))

# """
# Extras to add:
# - We calculate dynamic viscosity, need to convert to kinematic
# - Based on above, store density of system
# - Generation of the pbs script for each molecule to be accessed by array script
# - Need to run two separate simulations at the two different temperatures
# - Return number density seeing as we calculate it in script (eqmDensityFile)
# - A way to weight different calculated parameters

# Current approacg used to calculate transport properties (viscosity and thermal conductivity)

# - NVT at high temperature
# - NPT at 1 atmospheric temperature
# - NVT again to deform box to account for change in pressure
# - NVE to relax system
# - NVE used in the production run (you can indeed use NVE and langevin thermostat)

# """