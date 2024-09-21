"""
File to restart genetic algorithm
"""

################# IMPORTS ###################
import Genetic_Algorithm_Functions as GAF
from rdkit import Chem
from rdkit.Chem import Draw
from random import sample
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from random import choice as rnd
from os.path import join
from time import sleep
import sys
from rdkit.Chem.Draw import rdMolDraw2D
from MoleculeDifferenceViewer import view_difference
from copy import deepcopy
import ast
from operator import itemgetter
import os
import glob
import pandas as pd
import random
import math
import numpy as np
import shutil
import traceback
from gt4sd.properties import PropertyPredictorRegistry
import sys

######### Genetic Algorithm Parameters ############

fragments = ['CCCC', 'CCCCC', 'C(CC)CCC', 'CCC(CCC)CC', 'CCCC(C)CC', 'CCCCCCCCC', 'CCCCCCCC', 'CCCCCC', 'C(CCCCC)C']
Mutations = ['AddAtom', 'ReplaceAtom', 'ReplaceBond', 'RemoveAtom', 'AddFragment', 'RemoveFragment']

CopyCommand = 'cp'
Silent = True # Edit outputs to only print if this flag is False
NumElite = 25
IDcounter = 1
FirstGenerationAttempts = 0
MasterMoleculeList = [] #Keeping track of all generated molecules
FirstGenSimList = []
MaxNumHeavyAtoms = 50
MinNumHeavyAtoms = 5
MutationRate = 0.4
showdiff = False # Whether or not to display illustration of each mutation
GenerationSize = 50
LOPLS = False # Whether or not to use OPLS or LOPLS, False uses OPLS
MaxGenerations = 100
MaxMutationAttempts = 200
Fails = 0
NumRuns = 5
NumAtoms = 10000
Agent = 'Agent1'
STARTINGDIR = deepcopy(os.getcwd())
PYTHONPATH = 'python3'
generation = 3

### BOND TYPES
BondTypes = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]

### ATOM NUMBERS
Atoms = ['C', 'O']
AtomMolObjects = [Chem.MolFromSmiles(x) for x in Atoms]
AtomicNumbers = []

# Getting Atomic Numbers for Addable Atoms
for Object in AtomMolObjects:
     for atom in Object.GetAtoms():
          AtomicNumbers.append(atom.GetAtomicNum())   

### Wait until array jobs have finished
MoveOn = False
while MoveOn == False:
    GAF.runcmd(f'qstat > sims.txt')
    sims = []
    try:
        with open(join(STARTINGDIR, 'sims.txt'), 'r') as file:
            next(file) #Avoiding the first two lines
            next(file)
            filelist = file.readlines()
            for sim in filelist:
                if Agent in sim:
                    sims.append(sim)
                    print(sim)
    except:
        pass

    # Check if array jobs have finished
    if len(sims) != 0:
        print('Waiting for 10 mins')
        sleep(600)
    else:
        MoveOn = True

### Reformat directories so that properties can be calculated 
GenDirectory = join(STARTINGDIR, 'Molecules', f'Generation_{generation}')

# List of simulation directories
directories_with_generation = GAF.list_generation_directories(GenDirectory, 'Run') 

Num = 1
for RunDir in directories_with_generation:
    try:
        os.chdir(join(GenDirectory, RunDir))
        PDBNamePath = GAF.find_files_with_extension(os.getcwd(), '.pdb')[0]
        PDBName = GAF.extract_molecule_name(PDBNamePath)
        source_directory = join(GenDirectory, RunDir)
        destination_directory = join(GenDirectory, PDBName, RunDir)
        GAF.move_directory(source_directory, destination_directory)
        Num +=1
    except Exception as E:
        print(E)
        traceback.print_exc()
        pass

GAF.runcmd('rm -r Run*')

# Get PDBs from directories
MOLSMILESList = []

MOLIDList = GAF.list_generation_directories(join(STARTINGDIR, 'Molecules', f'Generation_{generation}'), 'Molecule')
print(MOLIDList)

for MoleculeDir in MOLIDList:
    Path = join(STARTINGDIR, 'Molecules',  f'Generation_{generation}', MoleculeDir, f'{MoleculeDir}.pdb')
    print(Path)
    MolObject = Chem.MolFromPDBFile(Path)
    SMILES = Chem.MolToSmiles(MolObject)
    MOLSMILESList.append(SMILES)

# Master Dataframe where molecules from all generations will be stored
MoleculeDatabase = pd.DataFrame(columns=['SMILES', 'MolObject', 'MutationList', 'HeavyAtoms', 'ID', 'Charge', 'MolMass', 'Predecessor', 'Score', 'Density100C', 'DViscosity40C',
                                        'DViscosity100C', 'KViscosity40C', 'KViscosity100C', 'KVI', 'DVI', 'Toxicity', 'SCScore', 'Density40C', 'SimilarityScore'])

# Generation Dataframe to store molecules from each generation
GenerationDatabase = pd.DataFrame(columns=['SMILES', 'MolObject', 'MutationList', 'HeavyAtoms', 'ID', 'Charge', 'MolMass', 'Predecessor', 'Score', 'Density100C', 'DViscosity40C',
                                        'DViscosity100C', 'KViscosity40C', 'KViscosity100C', 'KVI', 'DVI', 'Toxicity', 'SCScore', 'Density40C', 'SimilarityScore'])

# Here is where we will get the various values generated from the MD simulations

GenSimList = list(zip(MOLIDList, MOLSMILESList))

print(GenSimList)

for Molecule, MOLSMILES in GenSimList:
    try:
        # Create a function to wait until all simulations from this generation are finished
        os.chdir(join(STARTINGDIR, 'Molecules', f'Generation_{generation}', Molecule))
        CWD = os.getcwd()

        print('Getting Similarity Scores')
        ### Similarity Scores
        Scores = GAF.TanimotoSimilarity(MOLSMILES, MOLSMILESList)
        AvScore = 1 - (sum(Scores) / GenerationSize) # The higher the score, the less similar the molecule is to others

        print('Getting SCScore')
        ### SCScore
        SCScore = GAF.SCScore(MOLSMILES)
        SCScoreNorm = SCScore/5

        ### Toxicity
        ToxNorm = GAF.Toxicity(MOLSMILES)

        print('Getting Density')
        DirRuns = GAF.list_generation_directories(CWD, 'Run')
        ExampleRun = DirRuns[0]

        for run in DirRuns:
            try:
                DensityFile40 = f'{CWD}/{ExampleRun}/eqmDensity_{Molecule}_T313KP1atm.out'
                DensityFile100 = f'{CWD}/{ExampleRun}/eqmDensity_{Molecule}_T373KP1atm.out'
            except:
                continue

        ### Viscosity
        DVisc40 = GAF.GetVisc(join(STARTINGDIR, 'Molecules', f'Generation_{generation}'), Molecule, 313)
        DVisc100 = GAF.GetVisc(join(STARTINGDIR, 'Molecules', f'Generation_{generation}'), Molecule, 373)
        Dens40 = GAF.GetDens(DensityFile40)
        Dens100 = GAF.GetDens(DensityFile100)

        ## Viscosity Index
        KVI = GAF.GetKVI(DVisc40, DVisc100, Dens40, Dens100, STARTINGDIR)
        DVI = GAF.GetDVI(DVisc40, DVisc100)

        #Update Molecule Database
        IDNumber = int(Molecule.split('_')[-1])
        MoleculeDatabase.at[IDNumber - 1, 'SMILES'] = MOLSMILES
        MoleculeDatabase.at[IDNumber - 1, 'ID'] = Molecule
        MoleculeDatabase.at[IDNumber - 1, 'Density100C'] = Dens100
        MoleculeDatabase.at[IDNumber - 1, 'Density40C'] = Dens40
        MoleculeDatabase.at[IDNumber - 1, 'DViscosity40C'] = DVisc40
        MoleculeDatabase.at[IDNumber - 1, 'DViscosity100C'] = DVisc100
        MoleculeDatabase.at[IDNumber - 1, 'KViscosity40C'] = GAF.GetKVisc(DVisc=DVisc40, Dens=Dens40)
        MoleculeDatabase.at[IDNumber - 1, 'KViscosity100C'] = GAF.GetKVisc(DVisc=DVisc100, Dens=Dens100)
        MoleculeDatabase.at[IDNumber - 1, 'KVI'] = KVI
        MoleculeDatabase.at[IDNumber - 1, 'DVI'] = DVI
        MoleculeDatabase.at[IDNumber - 1, 'Toxicity'] = ToxNorm
        MoleculeDatabase.at[IDNumber - 1, 'SCScore'] = SCScoreNorm
        MoleculeDatabase.at[IDNumber - 1, 'SimilarityScore'] = SCScoreNorm

        #Update Generation Database
        GenerationDatabase.at[IDNumber - 1, 'SMILES'] = MOLSMILES
        GenerationDatabase.at[IDNumber - 1, 'ID'] = Molecule
        GenerationDatabase.at[IDNumber - 1, 'Density100C'] = Dens100
        GenerationDatabase.at[IDNumber - 1, 'Density40C'] = Dens40
        GenerationDatabase.at[IDNumber - 1, 'DViscosity40C'] = DVisc40
        GenerationDatabase.at[IDNumber - 1, 'DViscosity100C'] = DVisc100
        GenerationDatabase.at[IDNumber - 1, 'KViscosity40C'] = GAF.GetKVisc(DVisc=DVisc40, Dens=Dens40)
        GenerationDatabase.at[IDNumber - 1, 'KViscosity100C'] = GAF.GetKVisc(DVisc=DVisc100, Dens=Dens100)
        GenerationDatabase.at[IDNumber - 1, 'KVI'] = KVI
        GenerationDatabase.at[IDNumber - 1, 'DVI'] = DVI
        GenerationDatabase.at[IDNumber - 1, 'Toxicity'] = ToxNorm
        GenerationDatabase.at[IDNumber - 1, 'SCScore'] = SCScoreNorm
        GenerationDatabase.at[IDNumber - 1, 'SimilarityScore'] = SCScoreNorm

    except Exception as E:
        print(E)
        traceback.print_exc()
        pass

#### Generate Score
ViscScores = MoleculeDatabase['DViscosity40C'].tolist()
SCScores = MoleculeDatabase['SCScore'].tolist()
DVIScores = MoleculeDatabase['DVI'].tolist()
KVIScores = MoleculeDatabase['KVI'].tolist()
ToxicityScores = MoleculeDatabase['Toxicity'].tolist()
SimilarityScores = MoleculeDatabase['SimilarityScore'].tolist()
MoleculeNames = MoleculeDatabase['ID'].tolist()

ViscosityScore  = list(zip(MoleculeNames, ViscScores)) 
MolecularComplexityScore  = list(zip(MoleculeNames, SCScores)) 
DVIScore  = list(zip(MoleculeNames, DVIScores)) 
ToxicityScore  = list(zip(MoleculeNames, ToxicityScores)) 

ViscosityScore = [(x[0], 0) if math.isnan(x[1]) else x for x in ViscosityScore]
DVIScore = [(x[0], 0) if math.isnan(x[1]) else x for x in DVIScore]

# Apply the normalization function
Viscosity_normalized_molecule_scores = [(1-x[1]) for x in GAF.min_max_normalize(ViscosityScore)]
DVI_normalized_molecule_scores = [x[1] for x in GAF.min_max_normalize(DVIScore)]

MoleculeDatabase['ViscNormalisedScore'] = Viscosity_normalized_molecule_scores
MoleculeDatabase['DVINormalisedScore'] = DVI_normalized_molecule_scores
MoleculeDatabase['TotalScore'] = MoleculeDatabase['Toxicity'] + MoleculeDatabase['SCScore'] + MoleculeDatabase['DVINormalisedScore'] + MoleculeDatabase['ViscNormalisedScore'] 
MoleculeDatabase['NichedScore'] = MoleculeDatabase['TotalScore'] / MoleculeDatabase['SimilarityScore']

print(MoleculeDatabase)

#Make a pandas object with just the scores and the molecule ID
GenerationMolecules = pd.Series(MoleculeDatabase.NichedScore.values, index=MoleculeDatabase.ID).dropna()

print(GenerationMolecules)

GenerationMolecules = GenerationMolecules.to_dict()
print(GenerationMolecules)

# Sort dictiornary according to target score
ScoreSortedMolecules = sorted(GenerationMolecules.items(), key=lambda item:item[1], reverse=True)

#Convert tuple elements in sorted list back to lists 
ScoreSortedMolecules = [list(x) for x in ScoreSortedMolecules]

print(ScoreSortedMolecules)

# Constructing entries for use in subsequent generation
for entry in ScoreSortedMolecules:
    Key = int(entry[0].split('_')[-1]) - 1
    entry.insert(1, MoleculeDatabase.loc[Key]['MolObject'])
    entry.insert(2, MoleculeDatabase.loc[Key]['MutationList'])
    entry.insert(3, MoleculeDatabase.loc[Key]['HeavyAtoms'])
    entry.insert(4, MoleculeDatabase.loc[Key]['SMILES'])
try:
    MoleculeDatabase.drop("Unnamed: 0", axis=1, inplace=True) 
except:
    pass

#Save the update Master database and generation database
MoleculeDatabase.to_csv(f'{STARTINGDIR}/MoleculeDatabase_Generation_{generation}.csv', index=False)
MoleculeDatabase.to_csv(f'{STARTINGDIR}/Generation_{generation}_Database.csv', index=False)
generation_Initial = int(generation)
generation_Initial +=1

print(len(ScoreSortedMolecules))

################################## Subsequent generations #################################################
for generation in range(generation_Initial, MaxGenerations + 1):
    GenerationTotalAttempts = 0
    GenSimList = []
    IDcounter = 1

    # Generation Dataframe to store molecules from each generation
    GenerationDatabase = pd.DataFrame(columns=['SMILES', 'MolObject', 'MutationList', 'HeavyAtoms', 'ID', 'Charge', 'MolMass', 'Predecessor', 'Score', 'Density100C', 'DViscosity40C',
                                            'DViscosity100C', 'KViscosity40C', 'KViscosity100C', 'KVI', 'DVI', 'Toxicity', 'SCScore', 'Density40C', 'SimilarityScore'])
    
    MoleculeDatabase = pd.DataFrame(columns=['SMILES', 'MolObject', 'MutationList', 'HeavyAtoms', 'ID', 'Charge', 'MolMass', 'Predecessor', 'Score', 'Density100C', 'DViscosity40C',
                                        'DViscosity100C', 'KViscosity40C', 'KViscosity100C', 'KVI', 'DVI', 'Toxicity', 'SCScore', 'Density40C', 'SimilarityScore'])

    os.chdir(STARTINGDIR)
    # Store x best performing molecules (x=NumElite in list for next generation, without mutating them)
    GenerationMoleculeList = ScoreSortedMolecules[:NumElite]
    os.chdir(join(os.getcwd(), 'Molecules')) 
    GAF.runcmd(f'mkdir Generation_{generation}')
    os.chdir(STARTINGDIR)

    for x in list(range(0, 100)): #Start by mutating best performing molecules from previous generation and work down
        MutMol = None
        attempts = 0

        # Stop appending mutated molecules once generation reaches desired size
        if len(GenerationMoleculeList) == GenerationSize:
            break

        # Attempt crossover/mutation on each molecule, not moving on until a valid mutation has been suggested
        while MutMol == None:
            attempts += 1
            GenerationTotalAttempts += 1

            # Limit number of attempts at mutation, if max attempts exceeded, break loop to attempt on next molecule
            if attempts >= MaxMutationAttempts:
                Fails += 1 
                break

            # Get two parents using 3-way tournament selection
            Parent1 = GAF.KTournament(ScoreSortedMolecules[:NumElite])[0]
            Parent2 = GAF.KTournament(ScoreSortedMolecules[:NumElite])[0]

            # Attempt crossover
            try:
                result = GAF.Mol_Crossover(Chem.MolFromSmiles(Parent1), Chem.MolFromSmiles(Parent2))
                StartingMolecule = result[0]
            except Exception as E:
                continue

            # Number of heavy atoms
            try:
                print(result)
                NumHeavyAtoms = result[0].GetNumHeavyAtoms()
            except:
                continue 
            
            # Molecule ID
            Name = f'Generation_{generation}_Molecule_{IDcounter}'

            if NumHeavyAtoms > MaxNumHeavyAtoms * 0.8:
                MutationList = ['RemoveAtom', 'ReplaceAtom', 'ReplaceBond', 'RemoveFragment']
            else:
                MutationList = Mutations 
            
            print(f'\n#################################################################\nNumber of attempts: {attempts}')
            print(f'Total Crossover and/or Mutation Attempts: {GenerationTotalAttempts}')
            print(f'GENERATION: {generation}')

            #Decide whether to mutate molecule based on mutation rate
            if random.random() <= MutationRate:
                Mutate = True
                print('Attempting to Mutate')
            else:
                Mutate = False
                Mutation = None

            if Mutate:
                # Randomly select a mutation, here is where we can set mutation probabilities
                Mutation = rnd(MutationList)

                # Initialise Aromatic Ring
                AromaticMolecule = fragments[-1]

                try:
                    #Perform mutation, return Mut_Mol, Mut_Mol_Sanitized, MutMolSMILES, StartingMoleculeUnedited
                    result = GAF.Mutate(StartingMolecule, Mutation, AromaticMolecule, AtomicNumbers, BondTypes, Atoms, showdiff, fragments)
                except:
                    continue
            
            if GAF.GenMolChecks(result, MasterMoleculeList, MaxNumHeavyAtoms, MinNumHeavyAtoms, MaxAromRings=2) == None:
                MutMol = None
        
            else:
                HeavyAtoms = result[0].GetNumHeavyAtoms() # Get number of heavy atoms in molecule
                MutMol = result[0] # Get Mol object of mutated molecule
                MolMass = GAF.GetMolMass(MutMol) # Get estimate of of molecular mass 
                MutMolSMILES = result[2] # SMILES of mutated molecule

                print(f'Final SMILES: {result[2]}')

                try: # Try to generate all necessary files to simulate molecule
                    # Set feature definition file path to OPLS or LOPLS depending on user choice 
                    Name = f'Generation_{generation}_Molecule_{IDcounter}' # Set name of Molecule as its SMILES string

                    # Set feature definition file path to OPLS or LOPLS depending on user choice 
                    if LOPLS:
                        LTCOMMAND = f"{join(os.getcwd(), 'rdlt.py')} --smi '{MutMolSMILES}' -n {Name} -l -c"
                    else:
                        LTCOMMAND = f"{join(os.getcwd(), 'rdlt.py')} --smi '{MutMolSMILES}' -n {Name} -c"
                    
                    #Attempt to parameterise with OPLS
                    GAF.runcmd(f'{PYTHONPATH} {LTCOMMAND} > {STARTINGDIR}/{Name}.lt')

                    #Get molecule charge
                    charge = GAF.GetMolCharge(f'{os.getcwd()}/{Name}.lt')

                    #If successful, generate a PDB of molecule to use with Packmol
                    try:
                        GAF.GeneratePDB(MutMolSMILES, PATH=join(STARTINGDIR, f'{Name}.pdb'))
                    except:
                        continue

                    # Go into directory for this generation
                    os.chdir(join(STARTINGDIR, 'Molecules', f'Generation_{generation}'))
                    
                    Foldername = f'{Name}'

                    # Make a directory for the current molecule if it can be parameterised 
                    GAF.runcmd(f'mkdir {Foldername}')

                    # Enter molecule specific directory
                    os.chdir(join(os.getcwd(), Foldername))

                    #Check if file has already been made, skip if so, being sure not to make duplicate, otherwise move file to correct directory
                    CWD = os.getcwd() #Need to declare otherwise will get CWD from location function is being called from

                    #Copy molecule pdb to molecule directory
                    PDBFile = join(STARTINGDIR, f'{Name}.pdb')
                    GAF.runcmd(f'{CopyCommand} "{PDBFile}" {join(CWD, f"{Name}.pdb")}')
                    
                    #Copy molecule lt file to molecule directory
                    LTFile = join(STARTINGDIR, f'{Name}.lt')
                    GAF.runcmd(f'{CopyCommand} "{LTFile}" {join(CWD, f"{Name}.lt")}')
                    
                    #Get estimate for Number of molecules 
                    HMutMol = Chem.AddHs(MutMol)
                    NumMols = int(NumAtoms/HMutMol.GetNumAtoms()) # Maybe add field seeing how many mols were added to box

                    # Estimate starting box length
                    BoxL = GAF.CalcBoxLen(MolMass=MolMass, TargetDens=0.8, NumMols=NumMols)

                    # Make packmol files
                    GAF.MakePackmolFile(Name, CWD, NumMols=NumMols, BoxL=BoxL)

                    # Make Moltemplate files
                    GAF.MakeMoltemplateFile(Name, CWD, NumMols=NumMols, BoxL=BoxL)

                    if PYTHONPATH == 'python3':
                        GAF.runcmd(f'packmol < {Name}.inp')
                        GAF.runcmd(f'moltemplate.sh -pdb {Name}_PackmolFile.pdb {Name}_system.lt')

                        # Check that Moltemplate has generated all necessary files 
                        assert os.path.exists(join(CWD, f'{Name}_system.in.settings')), 'Settings file not generated'                 
                        assert os.path.exists(join(CWD, f'{Name}_system.in.charges')), 'Charges file not generated'
                        assert os.path.exists(join(CWD, f'{Name}_system.data')), 'Data file not generated'               

                    # Return to starting directory
                    os.chdir(STARTINGDIR)
                    # Add candidate and it's data to master list
                    MoleculeDatabase = GAF.DataUpdate(MoleculeDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
                                            MutationList=Mutation, ID=Name, Charge=charge, MolMass=MolMass, Predecessor=[Parent1, Parent2])
                    
                    GenerationDatabase = GAF.DataUpdate(GenerationDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
                        MutationList=Mutation, ID=Name, Charge=charge, MolMass=MolMass, Predecessor=[Parent1, Parent2])
                    
                    # Generate list of molecules to simulate in this generation
                    GenSimList.append([Name, MutMolSMILES, BoxL, NumMols])

                    if MutMolSMILES in MasterMoleculeList:
                        continue
                    else:
                        MasterMoleculeList.append(MutMolSMILES) #Keep track of already generated molecules

                    GenerationMoleculeList.append(MutMolSMILES) #Keep track of already generated molecules
                    print(f'Final Molecule SMILES: {MutMolSMILES}') 
                    IDcounter += 1

                except Exception as E:
                    print(E)
                    traceback.print_exc()
                    os.chdir(STARTINGDIR)
                    continue   

    #### Create duplicate trajectories for each molecule

    RunNum = 1

    if len(GenSimList) != 25:
        print(GenSimList)
        print(len(GenSimList))
        break

    for MolParam in GenSimList:
        Name = MolParam[0]
        MutMolSMILES = MolParam[1]
        BoxL = MolParam[2]
        NumMols = MolParam[3]

        for x in list(range(NumRuns)):
            os.chdir(STARTINGDIR)
            try:
                # Go into directory for this generation
                os.chdir(join(STARTINGDIR, 'Molecules', f'Generation_{generation}'))

                Foldername = f'Run_{RunNum}'
                RunNum +=1 

                # Make a directory for the current molecule if it can be parameterised 
                GAF.runcmd(f'mkdir {Foldername}')

                FilesDir = join(STARTINGDIR, 'Molecules', f'Generation_{generation}', f'{Name}')
                CurDir = join(STARTINGDIR, 'Molecules', f'Generation_{generation}', f'{Foldername}')
                SettingsFilePath = join(STARTINGDIR, 'settings.txt')

                #Move files from master Dir
                GAF.runcmd(f'cp -r {FilesDir}/* {CurDir}')
                GAF.runcmd(f'cp  {SettingsFilePath} {CurDir}')

                # Make LAMMPS files
                os.chdir(CurDir)
                GAF.MakeLAMMPSFile(Name, CurDir, Temp=313, GKRuntime=1500000, Run=Foldername)
                GAF.MakeLAMMPSFile(Name, CurDir, Temp=373, GKRuntime=1500000, Run=Foldername)

            except Exception as E:
                print(E)
                traceback.print_exc()
                pass

    #This is where we should call the simulation script
    CWD = join(STARTINGDIR, 'Molecules', f'Generation_{generation}')
    os.chdir(CWD)

    # Create array job for 40C viscosity
    GAF.CreateArrayJob(STARTINGDIR, CWD, NumRuns, Generation=generation, SimName='313K.lammps', GenerationSize=GenerationSize, Agent=Agent, NumElite=NumElite)
    # Create array job for 100C viscosity
    GAF.CreateArrayJob(STARTINGDIR, CWD, NumRuns, Generation=generation, SimName='373K.lammps', GenerationSize=GenerationSize, Agent=Agent, NumElite=NumElite)

    if PYTHONPATH == 'python3':
        GAF.runcmd(f'qsub {join(CWD, f"{Agent}_313K.lammps.pbs")}')
        GAF.runcmd(f'qsub {join(CWD, f"{Agent}_373K.lammps.pbs")}')

    os.chdir(STARTINGDIR)

    ### REMOVE UNNECESSARY FILES
    directory = STARTINGDIR
    pattern = f'Generation_{generation}_Molecule_*'  # Example: 'file_*.txt' to match files like file_1.txt, file_2.txt, etc.

    # Get a list of all files matching the pattern
    files_to_remove = glob.glob(os.path.join(directory, pattern))

    # Remove each file
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            print(f'Removed: {file_path}')
        except Exception as e:
            print(f'Error removing {file_path}: {e}')

    ### Wait until array jobs have finished
    MoveOn = False
    while MoveOn == False:
        GAF.runcmd(f'qstat > sims.txt')
        sims = []
        try:
            with open(join(STARTINGDIR, 'sims.txt'), 'r') as file:
                next(file) #Avoiding the first two lines
                next(file)
                filelist = file.readlines()
                for sim in filelist:
                    if Agent in sim:
                        sims.append(sim)
                        print(sim)
        except:
            pass

        # Check if array jobs have finished
        if len(sims) != 0:
            print('Waiting for 10 mins')
            sleep(600)
        else:
            MoveOn = True

    ### Reformat directories so that properties can be calculated 
    GenDirectory = join(STARTINGDIR, 'Molecules', f'Generation_{generation}')

    # List of simulation directories
    directories_with_generation = GAF.list_generation_directories(GenDirectory, 'Run') 

    Num = 1
    for RunDir in directories_with_generation:
        try:
            os.chdir(join(GenDirectory, RunDir))
            PDBNamePath = GAF.find_files_with_extension(os.getcwd(), '.pdb')[0]
            PDBName = GAF.extract_molecule_name(PDBNamePath)
            source_directory = join(GenDirectory, RunDir)
            destination_directory = join(GenDirectory, PDBName, RunDir)
            GAF.move_directory(source_directory, destination_directory)
            Num +=1
        except Exception as E:
            print(E)
            traceback.print_exc()
            pass

    GAF.runcmd('rm -r Run*')

    # Here is where we will get the various values generated from the MD simulations
    # Get PDBs from directories
    MOLSMILESList = []

    MOLIDList = GAF.list_generation_directories(join(STARTINGDIR, 'Molecules', f'Generation_{generation}'), 'Molecule')

    for MoleculeDir in MOLIDList:
        Path = join(STARTINGDIR, 'Molecules',  f'Generation_{generation}', MoleculeDir, f'{MoleculeDir}.pdb')
        print(Path)
        MolObject = Chem.MolFromPDBFile(Path)
        SMILES = Chem.MolToSmiles(MolObject)
        MOLSMILESList.append(SMILES)

    # Here is where we will get the various values generated from the MD simulations
    GenSimList = list(zip(MOLIDList, MOLSMILESList))
    print(GenSimList)

    for Molecule, MOLSMILES in GenSimList:
        print(Molecule)
        try:
            # Create a function to wait until all simulations from this generation are finished
            os.chdir(join(STARTINGDIR, 'Molecules', f'Generation_{generation}', Molecule))
            CWD = os.getcwd()

            print('Getting Similarity Scores')
            ### Similarity Scores
            Scores = GAF.TanimotoSimilarity(MOLSMILES, MOLSMILESList)
            AvScore = 1 - (sum(Scores) / GenerationSize) # The higher the score, the less similar the molecule is to others

            print('Getting SCScore')
            ### SCScore
            SCScore = GAF.SCScore(MOLSMILES)
            SCScoreNorm = SCScore/5

            ### Toxicity
            print('Getting Toxicity')
            ToxNorm = GAF.Toxicity(MOLSMILES)

            print('Getting Density')
            DirRuns = GAF.list_generation_directories(CWD, 'Run')
            ExampleRun = DirRuns[0]

            for run in DirRuns:
                try:
                    DensityFile40 = f'{CWD}/{ExampleRun}/eqmDensity_{Molecule}_T313KP1atm.out'
                    DensityFile100 = f'{CWD}/{ExampleRun}/eqmDensity_{Molecule}_T373KP1atm.out'
                except:
                    continue

            print('Getting Viscosity')
            ### Viscosity
            DVisc40 = GAF.GetVisc(join(STARTINGDIR, 'Molecules', f'Generation_{generation}'), Molecule, 313)
            DVisc100 = GAF.GetVisc(join(STARTINGDIR, 'Molecules', f'Generation_{generation}'), Molecule, 373)
            Dens40 = GAF.GetDens(DensityFile40)
            Dens100 = GAF.GetDens(DensityFile100)

            print('Getting VI')
            ## Viscosity Index
            KVI = GAF.GetKVI(DVisc40, DVisc100, Dens40, Dens100, STARTINGDIR)
            DVI = GAF.GetDVI(DVisc40, DVisc100)

            #Update Molecule Database
            IDNumber = int(Molecule.split('_')[-1])
            MoleculeDatabase.at[IDNumber - 1, 'Density100C'] = Dens100
            MoleculeDatabase.at[IDNumber - 1, 'Density40C'] = Dens40
            MoleculeDatabase.at[IDNumber - 1, 'DViscosity40C'] = DVisc40
            MoleculeDatabase.at[IDNumber - 1, 'DViscosity100C'] = DVisc100
            MoleculeDatabase.at[IDNumber - 1, 'KViscosity40C'] = GAF.GetKVisc(DVisc=DVisc40, Dens=Dens40)
            MoleculeDatabase.at[IDNumber - 1, 'KViscosity100C'] = GAF.GetKVisc(DVisc=DVisc100, Dens=Dens100)
            MoleculeDatabase.at[IDNumber - 1, 'KVI'] = KVI
            MoleculeDatabase.at[IDNumber - 1, 'DVI'] = DVI
            MoleculeDatabase.at[IDNumber - 1, 'Toxicity'] = ToxNorm
            MoleculeDatabase.at[IDNumber - 1, 'SCScore'] = SCScoreNorm
            MoleculeDatabase.at[IDNumber - 1, 'SimilarityScore'] = SCScoreNorm

            #Update Generation Database
            IDNumber = int(Molecule.split('_')[-1])
            GenerationDatabase.at[IDNumber - 1, 'Density100C'] = Dens100
            GenerationDatabase.at[IDNumber - 1, 'Density40C'] = Dens40
            GenerationDatabase.at[IDNumber - 1, 'DViscosity40C'] = DVisc40
            GenerationDatabase.at[IDNumber - 1, 'DViscosity100C'] = DVisc100
            GenerationDatabase.at[IDNumber - 1, 'KViscosity40C'] = GAF.GetKVisc(DVisc=DVisc40, Dens=Dens40)
            GenerationDatabase.at[IDNumber - 1, 'KViscosity100C'] = GAF.GetKVisc(DVisc=DVisc100, Dens=Dens100)
            GenerationDatabase.at[IDNumber - 1, 'KVI'] = KVI
            GenerationDatabase.at[IDNumber - 1, 'DVI'] = DVI
            GenerationDatabase.at[IDNumber - 1, 'Toxicity'] = ToxNorm
            GenerationDatabase.at[IDNumber - 1, 'SCScore'] = SCScoreNorm
            GenerationDatabase.at[IDNumber - 1, 'SimilarityScore'] = SCScoreNorm

        except Exception as E:
            print(E)
            traceback.print_exc()
            pass

    #### Generate Score
    ViscScores = MoleculeDatabase['DViscosity40C'].tolist()
    SCScores = MoleculeDatabase['SCScore'].tolist()
    DVIScores = MoleculeDatabase['DVI'].tolist()
    KVIScores = MoleculeDatabase['KVI'].tolist()
    ToxicityScores = MoleculeDatabase['Toxicity'].tolist()
    SimilarityScores = MoleculeDatabase['SimilarityScore'].tolist()
    MoleculeNames = MoleculeDatabase['ID'].tolist()

    ViscosityScore  = list(zip(MoleculeNames, ViscScores)) 
    MolecularComplexityScore  = list(zip(MoleculeNames, SCScores)) 
    DVIScore  = list(zip(MoleculeNames, DVIScores)) 
    ToxicityScore  = list(zip(MoleculeNames, ToxicityScores)) 

    ViscosityScore = [(x[0], 0) if math.isnan(x[1]) else x for x in ViscosityScore]
    DVIScore = [(x[0], 0) if math.isnan(x[1]) else x for x in DVIScore]

    # Apply the normalization function
    Viscosity_normalized_molecule_scores = [(1-x[1]) for x in GAF.min_max_normalize(ViscosityScore)]
    DVI_normalized_molecule_scores = [x[1] for x in GAF.min_max_normalize(DVIScore)]

    MoleculeDatabase['ViscNormalisedScore'] = Viscosity_normalized_molecule_scores
    MoleculeDatabase['DVINormalisedScore'] = DVI_normalized_molecule_scores
    MoleculeDatabase['TotalScore'] = MoleculeDatabase['Toxicity'] + MoleculeDatabase['SCScore'] + MoleculeDatabase['DVINormalisedScore'] + MoleculeDatabase['ViscNormalisedScore'] 
    MoleculeDatabase['NichedScore'] = MoleculeDatabase['TotalScore'] / MoleculeDatabase['SimilarityScore']

    #Make a pandas object with just the scores and the molecule ID
    GenerationMolecules = pd.Series(MoleculeDatabase.NichedScore.values, index=MoleculeDatabase.ID)

    GenerationMolecules = GenerationMolecules.to_dict()

    # Sort dictiornary according to target score
    ScoreSortedMolecules = sorted(GenerationMolecules.items(), key=lambda item:item[1], reverse=True)

    #Convert tuple elements in sorted list back to lists 
    ScoreSortedMolecules = [list(x) for x in ScoreSortedMolecules]

    # Constructing entries for use in subsequent generation
    for entry in ScoreSortedMolecules:
        Key = int(entry[0].split('_')[-1]) - 1
        entry.insert(1, MoleculeDatabase.loc[Key]['MolObject'])
        entry.insert(2, MoleculeDatabase.loc[Key]['MutationList'])
        entry.insert(3, MoleculeDatabase.loc[Key]['HeavyAtoms'])
        entry.insert(4, MoleculeDatabase.loc[Key]['SMILES'])

    try:
        MoleculeDatabase.drop("Unnamed: 0", axis=1, inplace=True) 
    except:
        pass

    #Save the update Master database and generation database
    MoleculeDatabase.to_csv(f'{STARTINGDIR}/MoleculeDatabase_Generation_{generation}.csv', index=False)
    MoleculeDatabase.to_csv(f'{STARTINGDIR}/Generation_{generation}_Database.csv', index=False)