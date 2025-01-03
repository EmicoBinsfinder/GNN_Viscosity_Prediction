"""
Author: Egheosa Ogbomo
Date: 21/11/2024

An implementation of a genetic algoritm using XGBoost as 
fitness function for several property predictions
"""

############### ENVIRONMENT SETUP ############
import subprocess
import sys

def runcmd(cmd, verbose = False, *args, **kwargs):
    #bascially allows python to run a bash command, and the code makes sure 
    #the error of the subproceess is communicated if it fails
    process = subprocess.run(
        cmd,
        text=True,
        shell=True)
    
    return process

sys.path.append('/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/GeneticAlgoMLRun')

################# IMPORTS ###################
import GeneticAlgorithmHelperFunction as GAF
from rdkit import Chem
from rdkit.Chem import Draw
from random import sample
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from random import choice as rnd
from os.path import join
from time import sleep
import sys
from copy import deepcopy
import os
import glob
import pandas as pd
import random
import numpy as np
import shutil
import traceback
from gt4sd.properties import PropertyPredictorRegistry

file_path = '/rds/general/user/eeo21/home/HIGH_THROUGHPUT_STUDIES/MLForTribology/GeneticAlgoMLRun/ModelsandDatasets/Viscosity_40C_Test_Descriptors.csv'
dataset = pd.read_csv(file_path)

# Define initial weightings for each category and remaining scores (user-defined)
user_weights = {
    'ViscScore': 20,  # Example weights; user-defined
    'HCScore': 20,
    'TCScore': 20,
    'Toxicity': 20,
    'DVIScore': 20,
    'SCScore': 20
}

Mols = dataset['SMILES']
fragments = ['CCCC', 'CCCCC', 'C(CC)CCC', 'CCC(CCC)CC', 'CCCC(C)CC', 'CCCCCCCCC', 'CCCCCCCC', 'CCCCCC', 'C(CCCCC)C', 'CC=OOCC(CC=O)O(CO)CO']

Mols = [Chem.MolFromSmiles(x) for x in Mols[:200]]
ReplaceMols = Mols
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

Mutations = ['AddAtom', 'ReplaceAtom', 'ReplaceBond', 'RemoveAtom', 'AddFragment', 'RemoveFragment', 'Napthalenate', 'Esterify', 'Glycolate']

#GENETIC ALGORITHM HYPERPARAMETERS
CopyCommand = 'cp'
Silent = True # Edit outputs to only print if this flag is False
NumElite = 25
IDcounter = 1
FirstGenerationAttempts = 0
MasterMoleculeList = [] #Keeping track of all generated molecules
FirstGenSimList = []
MaxNumHeavyAtoms = 40
MinNumHeavyAtoms = 5
MutationRate = 0.95
showdiff = False # Whether or not to display illustration of each mutation
GenerationSize = 50
LOPLS = False # Whether or not to use OPLS or LOPLS, False uses OPLS
MaxGenerations = 500
MaxMutationAttempts = 2000
Fails = 0
NumRuns = 5
NumAtoms = 10000
Agent = 'Agent1'


STARTINGDIR = deepcopy(os.getcwd())
PYTHONPATH = 'python3'
Generation = 1

Napthalenes = ['C12=CC=CC=C2C=CC=C1', 'C1CCCC2=CC=CC=C12', 'C1CCCC2CCCCC12', 'C1CCC2=CC=CC=C12']

# Master Dataframe where molecules from all generations will be stored
MoleculeDatabase = pd.DataFrame(columns=['SMILES', 'MolObject', 'MutationList', 'HeavyAtoms', 'ID', 'MolMass', 'Predecessor', 'Score', 'Density100C', 'DViscosity40C',
                                        'DViscosity100C', 'DVI', 'Toxicity', 'SCScore', 'Density40C', 'SimilarityScore',
                                        'ThermalConductivity_40C', 'ThermalConductivity_100C', 'HeatCapacity_40C', 'HeatCapacity_100C'])

# Generation Dataframe to store molecules from each generation
GenerationDatabase = pd.DataFrame(columns=['SMILES', 'MolObject', 'MutationList', 'HeavyAtoms', 'ID', 'MolMass', 'Predecessor', 'Score', 'Density100C', 'DViscosity40C',
                                        'DViscosity100C', 'DVI', 'Toxicity', 'SCScore', 'Density40C', 'SimilarityScore',
                                        'ThermalConductivity_40C', 'ThermalConductivity_100C', 'HeatCapacity_40C', 'HeatCapacity_100C'])

# Initialise population 
while len(MoleculeDatabase) < GenerationSize:

    print('\n###########################################################')
    print(f'Attempt number: {FirstGenerationAttempts}')
    StartingMolecule = rnd(Mols) #Select starting molecule

    print(len(MoleculeDatabase))

    Mutation = rnd(Mutations)
    AromaticMolecule = fragments[-1]

    # Perform mutation 
    result = GAF.Mutate(StartingMolecule, Mutation, AromaticMolecule, AtomicNumbers, BondTypes, Atoms, showdiff, Fragments=fragments, Napthalenes=Napthalenes, Mols=ReplaceMols)

    # Implement checks based on predetermined criteria (MolLength, Illegal Substructs etc.)
    if GAF.GenMolChecks(result, MasterMoleculeList, MaxNumHeavyAtoms, MinNumHeavyAtoms, MaxAromRings=2) == None:
        MutMol = None

    elif GAF.count_c_and_o(result[2]) > MaxNumHeavyAtoms:
        print('Mol too heavy')
        MutMol == None
        continue

    else:
        HeavyAtoms = result[0].GetNumHeavyAtoms() # Get number of heavy atoms in molecule
        MutMol = result[0] # Get Mol object of mutated molecule
        MolMass = GAF.GetMolMass(MutMol) # Get estimate of of molecular mass 
        MutMolSMILES = result[2] # SMILES of mutated molecule
        Predecessor = [None] # Get history of last two mutations performed on candidate
        ID = IDcounter
    
        try:
            Name = f'Generation_1_Molecule_{ID}' # Set name of Molecule as its SMILES string

            # Update Molecule database
            MoleculeDatabase = GAF.DataUpdate(MoleculeDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
                                                MutationList=[None, None, Mutation], ID=Name, MolMass=MolMass, Predecessor=Predecessor)

            GenerationDatabase = GAF.DataUpdate(GenerationDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
                                                MutationList=[None, None, Mutation], ID=Name, MolMass=MolMass, Predecessor=Predecessor)
            
            # Generate list of molecules to simulate in this generation
            FirstGenSimList.append([Name, MutMolSMILES])
            MasterMoleculeList.append(MutMolSMILES) #Keep track of already generated molecules
            print(f'Final Molecule SMILES: {MutMolSMILES}')
            IDcounter +=1 

        except Exception as E:
            print(E)
            traceback.print_exc()
            continue
    FirstGenerationAttempts += 1

MoleculeDatabase.to_csv(f'{STARTINGDIR}/MoleculeDatabase.csv')
GenerationDatabase.to_csv(f'{STARTINGDIR}/Generation_{Generation}_Database.csv')

# Here is where we will get the various values generated from the MD simulations
MOLSMILESList = [x[1] for x in FirstGenSimList]

for Molecule, MOLSMILES in FirstGenSimList:
    try:
        ### Similarity Scores
        Scores = GAF.TanimotoSimilarity(MOLSMILES, MOLSMILESList)
        AvScore = 1 - (sum(Scores) / GenerationSize) # The higher the score, the less similar the molecule is to others

        ## SCScore
        SCScore = GAF.SCScore(MOLSMILES)
        SCScoreNorm = SCScore/5

        ## Toxicity
        ToxNorm = GAF.Toxicity(MOLSMILES)

        ### Viscosity
        DVisc40 = GAF.Visc40ML(MOLSMILES)
        DVisc100 = GAF.Visc100ML(MOLSMILES)

        ### Density
        Dens40 = GAF.Dens40ML(MOLSMILES)
        Dens100 = GAF.Dens100ML(MOLSMILES)

        ### Heat Capacity
        HC40 = GAF.HeatCapacity40ML(MOLSMILES)
        HC100 = GAF.HeatCapacity100ML(MOLSMILES)

        ### Thermal Conductivity
        TC40 = GAF.ThermalConductivity40ML(MOLSMILES)
        TC100 = GAF.ThermalConductivity100ML(MOLSMILES)

        ## Viscosity Index
        DVI = GAF.GetDVI(DVisc40, DVisc100)

        #Update Molecule Database
        IDNumber = int(Molecule.split('_')[-1])
        MoleculeDatabase.at[IDNumber, 'Density100C'] = Dens100 #Density 40C
        MoleculeDatabase.at[IDNumber, 'Density40C'] = Dens40 #Density 100C
        MoleculeDatabase.at[IDNumber, 'DViscosity40C'] = DVisc40 #DVisc 40C
        MoleculeDatabase.at[IDNumber, 'DViscosity100C'] = DVisc100 #DVisc 100C
        MoleculeDatabase.at[IDNumber, 'HeatCapacity_40C'] = HC40
        MoleculeDatabase.at[IDNumber, 'HeatCapacity_100C'] = HC100
        MoleculeDatabase.at[IDNumber, 'ThermalConductivity_40C'] = TC40
        MoleculeDatabase.at[IDNumber, 'ThermalConductivity_100C'] = TC100
        MoleculeDatabase.at[IDNumber, 'DVI'] = DVI
        MoleculeDatabase.at[IDNumber, 'Toxicity'] = ToxNorm
        MoleculeDatabase.at[IDNumber, 'SCScore'] = SCScoreNorm
        MoleculeDatabase.at[IDNumber, 'SimilarityScore'] = AvScore

        #Update Generation Database
        GenerationDatabase.at[IDNumber, 'Density100C'] = Dens100 #Density 40C
        GenerationDatabase.at[IDNumber, 'Density40C'] = Dens40 #Density 100C
        GenerationDatabase.at[IDNumber, 'DViscosity40C'] = DVisc40 #DVisc 40C
        GenerationDatabase.at[IDNumber, 'DViscosity100C'] = DVisc100 #DVisc 100C
        GenerationDatabase.at[IDNumber, 'HeatCapacity_40C'] = HC40
        GenerationDatabase.at[IDNumber, 'HeatCapacity_100C'] = HC100
        GenerationDatabase.at[IDNumber, 'ThermalConductivity_40C'] = TC40
        GenerationDatabase.at[IDNumber, 'ThermalConductivity_100C'] = TC100
        GenerationDatabase.at[IDNumber, 'DVI'] = DVI
        GenerationDatabase.at[IDNumber, 'Toxicity'] = ToxNorm
        GenerationDatabase.at[IDNumber, 'SCScore'] = SCScoreNorm
        GenerationDatabase.at[IDNumber, 'SimilarityScore'] = AvScore

    except Exception as E:
        print(E)
        traceback.print_exc()

#### Generate Scores
ViscScores_40C = MoleculeDatabase['DViscosity40C'].tolist()
ViscScores_100C = MoleculeDatabase['DViscosity100C'].tolist()

HCScores_40C = MoleculeDatabase['HeatCapacity_40C'].tolist()
HCScores_100C = MoleculeDatabase['HeatCapacity_100C'].tolist()

TCScores_40C = MoleculeDatabase['DViscosity40C'].tolist()
TCScores_100C = MoleculeDatabase['DViscosity100C'].tolist()

SCScores = MoleculeDatabase['SCScore'].tolist()
DVIScores = MoleculeDatabase['DVI'].tolist()
ToxicityScores = MoleculeDatabase['Toxicity'].tolist()
SimilarityScores = MoleculeDatabase['SimilarityScore'].tolist()
MoleculeNames = MoleculeDatabase['ID'].tolist()

ViscosityScore_40C  = list(zip(MoleculeNames, ViscScores_40C)) 
ViscosityScore_100C  = list(zip(MoleculeNames, ViscScores_100C)) 

TCScore_40C  = list(zip(MoleculeNames, TCScores_40C)) 
TCScore_100C  = list(zip(MoleculeNames, TCScores_100C)) 

HCScore_40C  = list(zip(MoleculeNames, HCScores_40C)) 
HCScore_100C  = list(zip(MoleculeNames, HCScores_100C)) 

MolecularComplexityScore  = list(zip(MoleculeNames, SCScores)) 

DVIScore  = list(zip(MoleculeNames, DVIScores)) 

ToxicityScore  = list(zip(MoleculeNames, ToxicityScores)) 

# Apply the normalization function
Viscosity_normalized_molecule_scores_40C = [(1-x[1]) for x in GAF.min_max_normalize(ViscosityScore_40C)]
Viscosity_normalized_molecule_scores_100C = [(1-x[1]) for x in GAF.min_max_normalize(ViscosityScore_100C)]
HC_normalized_molecule_scores_40C = [(x[1]) for x in GAF.min_max_normalize(HCScore_40C)]
HC_normalized_molecule_scores_100C = [(x[1]) for x in GAF.min_max_normalize(HCScore_100C)]
TC_normalized_molecule_scores_40C = [(x[1]) for x in GAF.min_max_normalize(TCScore_40C)]
TC_normalized_molecule_scores_100C = [(x[1]) for x in GAF.min_max_normalize(TCScore_100C)]
DVI_normalized_molecule_scores = [x[1] for x in GAF.min_max_normalize(DVIScore)]
Tox_normalized_molecule_scores = [(1-x[1]) for x in GAF.min_max_normalize(ToxicityScore)]
SC_normalized_molecule_scores = [x[1] for x in GAF.min_max_normalize(MolecularComplexityScore)]

MoleculeDatabase['ViscNormalisedScore_40C'] = Viscosity_normalized_molecule_scores_40C
MoleculeDatabase['ViscNormalisedScore_100C'] = Viscosity_normalized_molecule_scores_100C
MoleculeDatabase['HCNormalisedScore_40C'] = HC_normalized_molecule_scores_40C
MoleculeDatabase['HCNormalisedScore_100C'] = HC_normalized_molecule_scores_100C
MoleculeDatabase['TCNormalisedScore_40C'] = TC_normalized_molecule_scores_40C
MoleculeDatabase['TCNormalisedScore_100C'] = TC_normalized_molecule_scores_100C

MoleculeDatabase['DVINormalisedScore'] = DVI_normalized_molecule_scores
MoleculeDatabase['ToxNormalisedScore'] = Tox_normalized_molecule_scores
MoleculeDatabase['SCSNormalisedScore'] = SC_normalized_molecule_scores

# Combine scores into categories
MoleculeDatabase['ViscScore'] = (
    (MoleculeDatabase['ViscNormalisedScore_40C'] + MoleculeDatabase['ViscNormalisedScore_100C']) /2
)
MoleculeDatabase['HCScore'] = (
    (MoleculeDatabase['HCNormalisedScore_40C'] + MoleculeDatabase['HCNormalisedScore_100C'])/2
)
MoleculeDatabase['TCScore'] = (
    (MoleculeDatabase['TCNormalisedScore_40C'] + MoleculeDatabase['TCNormalisedScore_100C'])/2
)

# Normalize weights so they sum to 100
total_weight = sum(user_weights.values())
normalized_weights = {key: (value / total_weight) * 100 for key, value in user_weights.items()}

# Compute the total score using normalized weights
MoleculeDatabase['TotalScore'] = (
    MoleculeDatabase['ViscScore'] * normalized_weights['ViscScore'] / 100 +
    MoleculeDatabase['HCScore'] * normalized_weights['HCScore'] / 100 +
    MoleculeDatabase['TCScore'] * normalized_weights['TCScore'] / 100 +
    MoleculeDatabase['DVINormalisedScore'] * normalized_weights['DVIScore'] / 100 +
    MoleculeDatabase['ToxNormalisedScore'] * normalized_weights['Toxicity'] / 100 +
    MoleculeDatabase['SCSNormalisedScore'] * normalized_weights['SCScore'] / 100
)

MoleculeDatabase['NichedScore'] = MoleculeDatabase['TotalScore'] / MoleculeDatabase['SimilarityScore']

#Make a pandas object with just the scores and the molecule ID
GenerationMolecules = pd.Series(MoleculeDatabase.NichedScore.values, index=MoleculeDatabase.ID).dropna().to_dict()

# Sort dictiornary according to target score
ScoreSortedMolecules = sorted(GenerationMolecules.items(), key=lambda item:item[1], reverse=True)

#Convert tuple elements in sorted list back to lists 
ScoreSortedMolecules = [list(x) for x in ScoreSortedMolecules]

# Constructing entries for use in subsequent generation
for entry in ScoreSortedMolecules:
    Key = int(entry[0].split('_')[-1])
    entry.insert(1, MoleculeDatabase.loc[Key]['MolObject'])
    entry.insert(2, MoleculeDatabase.loc[Key]['MutationList'])
    entry.insert(3, MoleculeDatabase.loc[Key]['HeavyAtoms'])
    entry.insert(4, MoleculeDatabase.loc[Key]['SMILES'])

#Save the update Master database and generation database
MoleculeDatabase.to_csv(f'{STARTINGDIR}/MoleculeDatabase.csv')
GenerationDatabase.to_csv(f'{STARTINGDIR}/Generation_{Generation}_Database.csv')

################################## Subsequent generations #################################################
for generation in range(2, MaxGenerations + 1):
    GenerationTotalAttempts = 0
    GenSimList = []

    OrderedMoleculeDatabase = MoleculeDatabase.sort_values(by='NichedScore', ascending=False)

    # Generation Dataframe to store molecules from each generation
    GenerationDatabase = OrderedMoleculeDatabase.head(25)

    GenerationMoleculeList = ScoreSortedMolecules[:NumElite]

    for i, entry in enumerate(ScoreSortedMolecules): #Start by mutating best performing molecules from previous generation and work down
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

                if result[2] == None:
                    continue
                    
            except Exception as E:
                continue

            # List containing last two successful mutations performed on molecule
            Predecessor = [Parent1, Parent2]
            PreviousMutations = entry[2]
            # Number of heavy atoms
            NumHeavyAtoms = int(entry[3])
            # Molecule ID
            Name = f'Generation_{generation}_Molecule_{IDcounter}'

            if GAF.count_c_and_o(result[2]) > 32:
                MutationList = ['RemoveFragment','ReplaceCandidate']
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

            if Mutate:
                Mutation = rnd(MutationList)
                AromaticMolecule = fragments[-1]
                print(Mutation)

                # Perform mutation 
                result = GAF.Mutate(StartingMolecule, Mutation, AromaticMolecule, AtomicNumbers, BondTypes, Atoms, showdiff, Fragments=fragments, Napthalenes=Napthalenes, Mols=ReplaceMols)
            
            else:
                Mutation = None

            # Implement checks based on predetermined criteria (MolLength, Illegal Substructs etc.)
            MutMol = GAF.GenMolChecks(result, MasterMoleculeList, MaxNumHeavyAtoms, MinNumHeavyAtoms, MaxAromRings=3)
            
            if MutMol == None:
                continue

            HeavyAtoms = result[0].GetNumHeavyAtoms() # Get number of heavy atoms in molecule

            if GAF.count_c_and_o(result[2]) > MaxNumHeavyAtoms:
                print('Mol too heavy')
                MutMol == None
                continue

            else:
                MutMol = result[0] # Get Mol object of mutated molecule
                MolMass = GAF.GetMolMass(MutMol) # Get estimate of of molecular mass 
                MutMolSMILES = result[2] # SMILES of mutated molecule
                # Update previous mutations object
                PreviousMutations.pop(0)
                PreviousMutations.append(Mutation)

                print(f'Final SMILES: {result[2]}')
                
                if GAF.count_c_and_o(result[2]) > MaxNumHeavyAtoms:
                    print('Mol too heavy')
                    MutMol = None
                    continue
            
                try:
                    Name = f'Generation_{generation}_Molecule_{IDcounter}' # Set name of Molecule as its SMILES string

                    # Update Molecule database
                    MoleculeDatabase = GAF.DataUpdate(MoleculeDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
                                                        MutationList=PreviousMutations, ID=Name, MolMass=MolMass, Predecessor=Predecessor)

                    GenerationDatabase = GAF.DataUpdate(GenerationDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
                                                        MutationList=PreviousMutations, ID=Name, MolMass=MolMass, Predecessor=Predecessor)
                    
                    # Generate list of molecules to simulate in this generation
                    GenSimList.append([Name, MutMolSMILES])
                    MasterMoleculeList.append(MutMolSMILES) #Keep track of already generated molecules
                    GenerationMoleculeList.append(MutMolSMILES) #Keep track of already generated molecules
                    print(f'Final Molecule SMILES: {MutMolSMILES}')
                    IDcounter +=1 

                except Exception as E:
                    print(E)
                    traceback.print_exc()
                    continue

    # Here is where we will get the various values generated from the MD simulations
    MOLSMILESList = [x[1] for x in GenSimList]

    for Molecule, MOLSMILES in GenSimList:
        try:
            ### Similarity Scores
            Scores = GAF.TanimotoSimilarity(MOLSMILES, MOLSMILESList)
            AvScore = 1 - (sum(Scores) / GenerationSize) # The higher the score, the less similar the molecule is to others

            ## SCScore
            SCScore = GAF.SCScore(MOLSMILES)
            SCScoreNorm = SCScore/5

            ## Toxicity
            ToxNorm = GAF.Toxicity(MOLSMILES)

            ### Viscosity
            DVisc40 = GAF.Visc40ML(MOLSMILES)
            DVisc100 = GAF.Visc100ML(MOLSMILES)

            ### Density
            Dens40 = GAF.Dens40ML(MOLSMILES)
            Dens100 = GAF.Dens100ML(MOLSMILES)

            ### Heat Capacity
            HC40 = GAF.HeatCapacity40ML(MOLSMILES)
            HC100 = GAF.HeatCapacity100ML(MOLSMILES)

            ### Thermal Conductivity
            TC40 = GAF.ThermalConductivity40ML(MOLSMILES)
            TC100 = GAF.ThermalConductivity100ML(MOLSMILES)

            ## Viscosity Index
            DVI = GAF.GetDVI(DVisc40, DVisc100)

            #Update Molecule Database
            IDNumber = int(Molecule.split('_')[-1])
            MoleculeDatabase.at[IDNumber, 'Density100C'] = Dens100 #Density 40C
            MoleculeDatabase.at[IDNumber, 'Density40C'] = Dens40 #Density 100C
            MoleculeDatabase.at[IDNumber, 'DViscosity40C'] = DVisc40 #DVisc 40C
            MoleculeDatabase.at[IDNumber, 'DViscosity100C'] = DVisc100 #DVisc 100C
            MoleculeDatabase.at[IDNumber, 'HeatCapacity_40C'] = HC40
            MoleculeDatabase.at[IDNumber, 'HeatCapacity_100C'] = HC100
            MoleculeDatabase.at[IDNumber, 'ThermalConductivity_40C'] = TC40
            MoleculeDatabase.at[IDNumber, 'ThermalConductivity_100C'] = TC100
            MoleculeDatabase.at[IDNumber, 'DVI'] = DVI
            MoleculeDatabase.at[IDNumber, 'Toxicity'] = ToxNorm
            MoleculeDatabase.at[IDNumber, 'SCScore'] = SCScoreNorm
            MoleculeDatabase.at[IDNumber, 'SimilarityScore'] = AvScore

            GenerationDatabase.at[IDNumber, 'Density100C'] = Dens100 #Density 40C
            GenerationDatabase.at[IDNumber, 'Density40C'] = Dens40 #Density 100C
            GenerationDatabase.at[IDNumber, 'DViscosity40C'] = DVisc40 #DVisc 40C
            GenerationDatabase.at[IDNumber, 'DViscosity100C'] = DVisc100 #DVisc 100C
            GenerationDatabase.at[IDNumber, 'HeatCapacity_40C'] = HC40
            GenerationDatabase.at[IDNumber, 'HeatCapacity_100C'] = HC100
            GenerationDatabase.at[IDNumber, 'ThermalConductivity_40C'] = TC40
            GenerationDatabase.at[IDNumber, 'ThermalConductivity_100C'] = TC100
            GenerationDatabase.at[IDNumber, 'DVI'] = DVI
            GenerationDatabase.at[IDNumber, 'Toxicity'] = ToxNorm
            GenerationDatabase.at[IDNumber, 'SCScore'] = SCScoreNorm
            GenerationDatabase.at[IDNumber, 'SimilarityScore'] = AvScore

        except Exception as E:
            print(E)
            traceback.print_exc()

    #### Generate Scores
    MoleculeNames = MoleculeDatabase['ID'].tolist()
    ViscScores_40C = MoleculeDatabase['DViscosity40C'].tolist()
    ViscScores_100C = MoleculeDatabase['DViscosity100C'].tolist()
    HCScores_40C = MoleculeDatabase['HeatCapacity_40C'].tolist()
    HCScores_100C = MoleculeDatabase['HeatCapacity_100C'].tolist()
    TCScores_40C = MoleculeDatabase['DViscosity40C'].tolist()
    TCScores_100C = MoleculeDatabase['DViscosity100C'].tolist()
    SCScores = MoleculeDatabase['SCScore'].tolist()
    DVIScores = MoleculeDatabase['DVI'].tolist()
    ToxicityScores = MoleculeDatabase['Toxicity'].tolist()
    SimilarityScores = MoleculeDatabase['SimilarityScore'].tolist()

    # Update generation database scores
    GenMoleculeNames = GenerationDatabase['ID'].tolist()
    GenViscScores_40C = GenerationDatabase['DViscosity40C'].tolist()
    GenViscScores_100C = GenerationDatabase['DViscosity100C'].tolist()
    GenHCScores_40C = GenerationDatabase['HeatCapacity_40C'].tolist()
    GenHCScores_100C = GenerationDatabase['HeatCapacity_100C'].tolist()
    GenTCScores_40C = GenerationDatabase['DViscosity40C'].tolist()
    GenTCScores_100C = GenerationDatabase['DViscosity100C'].tolist()
    GenSCScores = GenerationDatabase['SCScore'].tolist()
    GenDVIScores = GenerationDatabase['DVI'].tolist()
    GenToxicityScores = GenerationDatabase['Toxicity'].tolist()
    GenSimilarityScores = GenerationDatabase['SimilarityScore'].tolist()
    GenMoleculeNames = GenerationDatabase['ID'].tolist()

    #### Mol Database Scores - Mol Database
    ViscosityScore_40C  = list(zip(MoleculeNames, ViscScores_40C)) 
    ViscosityScore_100C  = list(zip(MoleculeNames, ViscScores_100C)) 
    TCScore_40C  = list(zip(MoleculeNames, TCScores_40C)) 
    TCScore_100C  = list(zip(MoleculeNames, TCScores_100C)) 
    HCScore_40C  = list(zip(MoleculeNames, HCScores_40C)) 
    HCScore_100C  = list(zip(MoleculeNames, HCScores_100C)) 
    MolecularComplexityScore  = list(zip(MoleculeNames, SCScores)) 
    DVIScore  = list(zip(MoleculeNames, DVIScores)) 
    ToxicityScore  = list(zip(MoleculeNames, ToxicityScores)) 

    # Apply the normalization function
    Viscosity_normalized_molecule_scores_40C = [(1-x[1]) for x in GAF.min_max_normalize(ViscosityScore_40C)]
    Viscosity_normalized_molecule_scores_100C = [(1-x[1]) for x in GAF.min_max_normalize(ViscosityScore_100C)]

    HC_normalized_molecule_scores_40C = [(x[1]) for x in GAF.min_max_normalize(HCScore_40C)]
    HC_normalized_molecule_scores_100C = [(x[1]) for x in GAF.min_max_normalize(HCScore_100C)]

    TC_normalized_molecule_scores_40C = [(x[1]) for x in GAF.min_max_normalize(TCScore_40C)]
    TC_normalized_molecule_scores_100C = [(x[1]) for x in GAF.min_max_normalize(TCScore_100C)]

    DVI_normalized_molecule_scores = [x[1] for x in GAF.min_max_normalize(DVIScore)]
    Tox_normalized_molecule_scores = [(1-x[1]) for x in GAF.min_max_normalize(ToxicityScore)]
    SC_normalized_molecule_scores = [x[1] for x in GAF.min_max_normalize(MolecularComplexityScore)]

    MoleculeDatabase['ViscNormalisedScore_40C'] = Viscosity_normalized_molecule_scores_40C
    MoleculeDatabase['ViscNormalisedScore_100C'] = Viscosity_normalized_molecule_scores_100C
    MoleculeDatabase['HCNormalisedScore_40C'] = HC_normalized_molecule_scores_40C
    MoleculeDatabase['HCNormalisedScore_100C'] = HC_normalized_molecule_scores_100C
    MoleculeDatabase['TCNormalisedScore_40C'] = TC_normalized_molecule_scores_40C
    MoleculeDatabase['TCNormalisedScore_100C'] = TC_normalized_molecule_scores_100C

    MoleculeDatabase['DVINormalisedScore'] = DVI_normalized_molecule_scores
    MoleculeDatabase['ToxNormalisedScore'] = Tox_normalized_molecule_scores
    MoleculeDatabase['SCSNormalisedScore'] = SC_normalized_molecule_scores

    # Combine scores into categories
    MoleculeDatabase['ViscScore'] = (
        (MoleculeDatabase['ViscNormalisedScore_40C'] + MoleculeDatabase['ViscNormalisedScore_100C']) /2
    )
    MoleculeDatabase['HCScore'] = (
        (MoleculeDatabase['HCNormalisedScore_40C'] + MoleculeDatabase['HCNormalisedScore_100C'])/2
    )
    MoleculeDatabase['TCScore'] = (
        (MoleculeDatabase['TCNormalisedScore_40C'] + MoleculeDatabase['TCNormalisedScore_100C'])/2
    )

    # Normalize weights so they sum to 100
    total_weight = sum(user_weights.values())
    normalized_weights = {key: (value / total_weight) * 100 for key, value in user_weights.items()}

    # Compute the total score using normalized weights
    MoleculeDatabase['TotalScore'] = (
        MoleculeDatabase['ViscScore'] * normalized_weights['ViscScore'] / 100 +
        MoleculeDatabase['HCScore'] * normalized_weights['HCScore'] / 100 +
        MoleculeDatabase['TCScore'] * normalized_weights['TCScore'] / 100 +
        MoleculeDatabase['DVINormalisedScore'] * normalized_weights['DVIScore'] / 100 +
        MoleculeDatabase['ToxNormalisedScore'] * normalized_weights['Toxicity'] / 100 +
        MoleculeDatabase['SCSNormalisedScore'] * normalized_weights['SCScore'] / 100
    )

    MoleculeDatabase['NichedScore'] = MoleculeDatabase['TotalScore'] / MoleculeDatabase['SimilarityScore']
    
    ##### Gen database score
    GenViscosityScore_40C  = list(zip(MoleculeNames, GenViscScores_40C)) 
    GenViscosityScore_100C  = list(zip(MoleculeNames, GenViscScores_100C)) 
    GenTCScore_40C  = list(zip(MoleculeNames, GenTCScores_40C)) 
    GenTCScore_100C  = list(zip(MoleculeNames, GenTCScores_100C)) 
    GenHCScore_40C  = list(zip(MoleculeNames, GenHCScores_40C)) 
    GenHCScore_100C  = list(zip(MoleculeNames, GenHCScores_100C)) 
    GenMolecularComplexityScore  = list(zip(MoleculeNames, GenSCScores)) 
    GenDVIScore  = list(zip(MoleculeNames, GenDVIScores)) 
    GenToxicityScore  = list(zip(MoleculeNames, GenToxicityScores)) 

    # Apply the normalization function
    GenViscosity_normalized_molecule_scores_40C = [(1-x[1]) for x in GAF.min_max_normalize(GenViscosityScore_40C)]
    GenViscosity_normalized_molecule_scores_100C = [(1-x[1]) for x in GAF.min_max_normalize(GenViscosityScore_100C)]

    GenHC_normalized_molecule_scores_40C = [(x[1]) for x in GAF.min_max_normalize(GenHCScore_40C)]
    GenHC_normalized_molecule_scores_100C = [(x[1]) for x in GAF.min_max_normalize(GenHCScore_100C)]

    GenTC_normalized_molecule_scores_40C = [(x[1]) for x in GAF.min_max_normalize(GenTCScore_40C)]
    GenTC_normalized_molecule_scores_100C = [(x[1]) for x in GAF.min_max_normalize(GenTCScore_100C)]

    GenDVI_normalized_molecule_scores = [x[1] for x in GAF.min_max_normalize(GenDVIScore)]
    GenTox_normalized_molecule_scores = [(1-x[1]) for x in GAF.min_max_normalize(GenToxicityScore)]
    GenSC_normalized_molecule_scores = [x[1] for x in GAF.min_max_normalize(GenMolecularComplexityScore)]

    GenerationDatabase['ViscNormalisedScore_40C'] = GenViscosity_normalized_molecule_scores_40C
    GenerationDatabase['ViscNormalisedScore_100C'] = GenViscosity_normalized_molecule_scores_100C
    GenerationDatabase['HCNormalisedScore_40C'] = GenHC_normalized_molecule_scores_40C
    GenerationDatabase['HCNormalisedScore_100C'] = GenHC_normalized_molecule_scores_100C
    GenerationDatabase['TCNormalisedScore_40C'] = GenTC_normalized_molecule_scores_40C
    GenerationDatabase['TCNormalisedScore_100C'] = GenTC_normalized_molecule_scores_100C
    GenerationDatabase['DVINormalisedScore'] = GenDVI_normalized_molecule_scores
    GenerationDatabase['ToxNormalisedScore'] = GenTox_normalized_molecule_scores
    GenerationDatabase['SCSNormalisedScore'] = GenSC_normalized_molecule_scores

    # Combine scores into categories
    GenerationDatabase['ViscScore'] = (
        (GenerationDatabase['ViscNormalisedScore_40C'] + GenerationDatabase['ViscNormalisedScore_100C']) /2
    )
    GenerationDatabase['HCScore'] = (
        (GenerationDatabase['HCNormalisedScore_40C'] + GenerationDatabase['HCNormalisedScore_100C'])/2
    )
    GenerationDatabase['TCScore'] = (
        (GenerationDatabase['TCNormalisedScore_40C'] + GenerationDatabase['TCNormalisedScore_100C']) /2
    )

    # Normalize weights so they sum to 100
    total_weight = sum(user_weights.values())
    normalized_weights = {key: (value / total_weight) * 100 for key, value in user_weights.items()}

    # Compute the total score using normalized weights
    GenerationDatabase['TotalScore'] = (
        GenerationDatabase['ViscScore'] * normalized_weights['ViscScore'] / 100 +
        GenerationDatabase['HCScore'] * normalized_weights['HCScore'] / 100 +
        GenerationDatabase['TCScore'] * normalized_weights['TCScore'] / 100 +
        GenerationDatabase['DVINormalisedScore'] * normalized_weights['DVIScore'] / 100 +
        GenerationDatabase['ToxNormalisedScore'] * normalized_weights['Toxicity'] / 100 +
        GenerationDatabase['SCSNormalisedScore'] * normalized_weights['SCScore'] / 100
    )

    GenerationDatabase['NichedScore'] = GenerationDatabase['TotalScore'] / GenerationDatabase['SimilarityScore']

    ### Make a pandas object with just the scores and the molecule ID
    GenerationMolecules = pd.Series(MoleculeDatabase.NichedScore.values, index=MoleculeDatabase.ID).dropna().to_dict()

    # Sort dictiornary according to target score
    ScoreSortedMolecules = sorted(GenerationMolecules.items(), key=lambda item:item[1], reverse=True)

    #Convert tuple elements in sorted list back to lists 
    ScoreSortedMolecules = [list(x) for x in ScoreSortedMolecules]

    # Constructing entries for use in subsequent generation
    for entry in ScoreSortedMolecules:
        Key = int(entry[0].split('_')[-1])
        entry.insert(1, MoleculeDatabase.loc[Key]['MolObject'])
        entry.insert(2, MoleculeDatabase.loc[Key]['MutationList'])
        entry.insert(3, MoleculeDatabase.loc[Key]['HeavyAtoms'])
        entry.insert(4, MoleculeDatabase.loc[Key]['SMILES'])

    #Save the update Master database and generation database
    MoleculeDatabase.to_csv(f'{STARTINGDIR}/MoleculeDatabase.csv')
    GenerationDatabase.to_csv(f'{STARTINGDIR}/Generation_{generation}_Database.csv')
