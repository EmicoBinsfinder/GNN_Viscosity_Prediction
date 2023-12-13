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
- Add Branch

CONSIDERATIONS

- Should mutations only happen to the best x molecules or any of the molecules

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

- Can the molecule be parameterised?

- How to ensure that we are actually generating molecules that will be liquid at room temperature (how do we design 
gaseous lubricants)

- Need to decide best fingerprint for calculating molecular similarity

- Need a parameter for elitism

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


runcmd('module load anaconda3/personal')
runcmd('source activate HTVS')

runcmd('export PATH="$PATH:/rds/general/user/eeo21/home/moltemplate/moltemplate/moltemplate/scripts"')
runcmd('export PATH="$PATH:/rds/general/user/eeo21/home/moltemplate/moltemplate/moltemplate/"')

runcmd('export PATH="$PATH:/rds/general/user/eeo21/home/Packmol/packmol-20.14.2/"')

################# IMPORTS ###################
import Genetic_Algorithm_Functions as GAF
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
import os

DrawingOptions.includeAtomNumbers=True
DrawingOptions.bondLineWidth=1.8
DrawingOptions.atomLabelFontSize=14

### Fragments
fragments = ['CCCC', 'CCCCC', 'CCCCCC', 'CCC(CCC)CC', 'CCCC(C)CC', 'CCCCCCCCC', 'CCCCCCCC', 'c1ccccc1']
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

Silent = True # Edit outputs to only print if this flag is true 
NumElite = 15
counter = 0
FirstGenerationAttempts = 0
GeneratedMolecules = {}
GenerationMolecules = []
MaxNumHeavyAtoms = 50
showdiff = False # Whether or not to display illustration of each mutation
GenerationSize = 50
LOPLS = True # Whether or not to use OPLS or LOPLS, False uses OPLS

PYTHONPATH = 'C:/Users/eeo21/AppData/Local/Programs/Python/Python310/python.exe'
STARTINGDIR = deepcopy(os.getcwd())
#PYTHONPATH = 'python3'
GAF.runcmd(f'mkdir Molecules')
os.chdir(os.path.join(os.getcwd(), 'Molecules'))
GAF.runcmd(f'mkdir Generation_1')
os.chdir(STARTINGDIR)

# Initialise population 

while len(GeneratedMolecules) < GenerationSize:

    print('\n###########################################################')
    print(f'Attempt number: {FirstGenerationAttempts}')
    StartingMolecule = rnd(fragments) #Select starting molecule
    StartingMoleculeSMILES = Chem.MolToSmiles(StartingMolecule)

    Mutation = rnd(Mutations)
    AromaticMolecule = fragments[-1]

    # Perform mutation 
    result = GAF.Mutate(StartingMolecule, Mutation, AromaticMolecule, AtomicNumbers, BondTypes, Atoms, showdiff, fragments)

    # Check for null or fragmented results
    if result[2] == None or len(Chem.GetMolFrags(result[0])) > 1:
        continue
    
    # Check if candidate has already been generated by checking if SMILES string is in master list
    elif result[2] in GeneratedMolecules.keys():
        print('Molecule previously generated')
        continue

    # Check if Candidate is too short
    elif result[0].GetNumHeavyAtoms() < 5:
        print('Molecule too short')
        continue

    else:
        # Check if molecule can be parameterised with OPLS
        HeavyAtoms = result[0].GetNumHeavyAtoms() # Get number of heavy atoms in molecule
        MutMol = result[0]
        MutMolSMILES = result[2] # Get Mol object of mutated molecule
        PreviousMolecule = result[3] # Get history of last two mutations performed on candidate
        Score = GAF.fitfunc(StartingMoleculeSMILES, 1) # Apply fitness function to candidate
        ID = counter

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
            print(f'OPLS Molecule Charge is: {float(charge)}')

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

            if PYTHONPATH == 'python3':
                GAF.runcmd(f'packmol < {Name}.inp')
                GAF.runcmd(f'moltemplate.sh -pdb {Name}_PackmolFile.pdb {Name}_system.lt')

            # Make Datafiles with Moltemplate

            # Make LAMMPS datafiles

            # Return to starting directory
            os.chdir(STARTINGDIR) 
                
            # Add molecules to master list
            GeneratedMolecules[f'{MutMolSMILES}'] = [MutMol, [None, Mutation], HeavyAtoms, Score, Name, charge] 

            # Initialise molecules for next generation
            GenerationMolecules.append([MutMolSMILES, MutMol, [None, Mutation], HeavyAtoms, Score, Name, charge])
            print(f'Final Molecule SMILES: {MutMolSMILES}')
            counter +=1

        except Exception as E:
            print(E)
            continue   
 
    FirstGenerationAttempts += 1

################################### Subsequent generations #################################################
MaxGenerations = 4 
NumGenerations = 1
MaxMutationAttempts = 200
Fails = 0

for generation in range(2, MaxGenerations):
    GenerationTotalAttempts = 0
    IDcounter = 1 # Counter for generating molecule IDs

    # Perform elitism selection
    ScoreSortedMolecules = sorted(GenerationMolecules, key=itemgetter(4), reverse=True)

    # Store x best performing molecules (x=NumElite in list for next generation, without mutating them)
    GenerationMolecules = ScoreSortedMolecules[:NumElite]
    os.chdir(os.path.join(os.getcwd(), 'Molecules')) 
    GAF.runcmd(f'mkdir Generation_{generation}')
    os.chdir(STARTINGDIR)

    for i, entry in enumerate(GenerationMolecules): #Start by mutating best performing molecules from previous generation and work down
        MutMol = None
        attempts = 0

        # Stop appending mutated molecules
        if len(GenerationMolecules) == GenerationSize:
            print(len(GenerationMolecules))
            break

        # Attempt mutation on each molecule, not moving on until a valid mutation has been suggested
        while MutMol == None:
            attempts += 1
            GenerationTotalAttempts += 1

            # Limit number of attempts at mutation, if max attempts exceeded, break loop to attempt on next molecule
            if attempts >= MaxMutationAttempts:
                Fails += 1 
                break

            # Starting molecule SMILES string
            StartingMoleculeSMILES = entry[0]
            
            # Starting Molecule mol object
            StartingMolecule = entry[1]
            
            # List containing last two successful mutations performed on molecule
            PreviousMutations = entry[2]
            
            # Number of heavy atoms
            NumHeavyAtoms = entry[3]

            # Molecule ID
            ID = f'Generation_{generation}_Molecule_{IDcounter}'

            if NumHeavyAtoms > 42:
                MutationList = ['RemoveAtom']
            else:
                MutationList = Mutations 
            
            print(f'\n#################################################################\nNumber of attempts: {attempts}')
            print(f'Total Mutation Attempts: {GenerationTotalAttempts}')
            print(f'GENERATION: {generation}')

            # Randomly select a mutation, here is where we can set mutation probabilities
            Mutation = rnd(MutationList)

            # Initialise Aromatic Ring
            AromaticMolecule = fragments[-1]

            #Perform mutation, return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited
            result = GAF.Mutate(StartingMolecule, Mutation, AromaticMolecule, AtomicNumbers, BondTypes, Atoms, showdiff, fragments)
            
            # Check for null results or fragmented molecules
            if result[0] == None or len(Chem.GetMolFrags(result[0])) > 1:
                MutMol = None

            # Check if candidate has already been generated by checking if SMILES string is in master list
            elif result[2] in GeneratedMolecules.keys():
                print('Molecule previously generated')
                MutMol = None

            # Limit number of heavy atoms in generated candidates
            elif NumHeavyAtoms > MaxNumHeavyAtoms:
                print('Molecule has too many heavy atoms')
                MutMol = None

            # Check for molecules that are too short
            elif result[0].GetNumHeavyAtoms() < 3:
                print('Molecule too short')
                MutMol = None

            # Check for illegal substructures
            elif GAF.CheckSubstruct(result[0]):
                MutMol = None
        
            else:
                MutMol = result[0]
                NumHeavyAtoms = result[0].GetNumHeavyAtoms()
                print(f'Number of Heavy Atoms after mutation: {NumHeavyAtoms}')                

                # Update previous mutations object
                PreviousMutations.pop(0)
                PreviousMutations.append(Mutation)

                print(f'Final SMILES: {result[2]}')

                try: # Try to generate all necessary files to simulate molecule
                    Name = f'{ID}' # Set name of Molecule as its SMILES string

                    # Set feature definition file path to OPLS or LOPLS depending on user choice 
                    if LOPLS:
                        LTCOMMAND = f"{os.path.join(os.getcwd(), 'rdlt.py')} --smi {MutMolSMILES} -n {Name} -l -c"
                    else:
                        LTCOMMAND = f"{os.path.join(os.getcwd(), 'rdlt.py')} --smi {MutMolSMILES} -n {Name} -c"
                    
                    #Attempt to parameterise with OPLS
                    GAF.runcmd(f'{PYTHONPATH} {LTCOMMAND} > {STARTINGDIR}/{Name}.lt')

                    #Get molecule charge
                    charge = GAF.GetMolCharge(f'{os.getcwd()}/{Name}.lt')
                    print(f'OPLS Molecule Charge is: {float(charge)}')

                    #If successful, generate a PDB of molecule to use with Packmol
                    GAF.GeneratePDB(MutMolSMILES, PATH=os.path.join(STARTINGDIR, f'{Name}.pdb'))

                    # Go into directory for this generation
                    os.chdir(os.path.join(STARTINGDIR, 'Molecules', f'Generation_{generation}'))
                    
                    # Make a directory for the current molecule if it can be parameterised 
                    GAF.runcmd(f'mkdir {Name}')

                    # Enter molecule specific directory
                    os.chdir(os.path.join(os.getcwd(), f'{Name}'))

                    #Check if file has already been made, skip if so, being sure not to make duplicate, otherwise move file to correct directory
                    CWD = os.getcwd()
                    GAF.CheckMoveFile(Name, STARTINGDIR, 'lt', CWD)
                    GAF.CheckMoveFile(Name, STARTINGDIR, 'pdb', CWD)

                    # Make packmol files
                    GAF.MakePackmolFile(Name, CWD)

                    # Make Moltemplate files
                    GAF.MakeMoltemplateFile(Name, CWD)

                    if PYTHONPATH == 'python3':
                        GAF.runcmd(f'packmol < {Name}.inp')
                        GAF.runcmd(f'moltemplate.sh -pdb {Name}_PackmolFile.pdb {Name}_system.lt')                             

                    # Return to starting directory
                    os.chdir(STARTINGDIR) 
                  
                except Exception as E:
                    print(E)
                    continue   

                # Add candidate and it's data to master list
                GeneratedMolecules[f'{result[2]}'] = [result[0], PreviousMutations, NumHeavyAtoms, 
                                                    Score, ID]

                # Molecules to initiate next generation, add NumElite to insertion index to prevent elite molecules
                # being overwritten
                GenerationMolecules.append([result[2], result[0], PreviousMutations, NumHeavyAtoms, 
                                        Score, ID])
                
                IDcounter += 1

    # # Tasks to perform at end of every generation
    # # Simulate molecules that haven't been yet been simulated

print(f'Number of failed mutations: {Fails}')
