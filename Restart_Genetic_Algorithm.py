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
from operator import itemgetter
import os
import glob
import pandas as pd
import random
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

############### Restart GA ######################

#Load in progress from master Database
MoleculeDatabase = pd.read_csv('MoleculeDatabase.csv')

# Calculate Generation Number
FullMOLIDList = MoleculeDatabase['ID'].to_list()
generation = FullMOLIDList[-1].split('_')[1]
GenerationDatabase = pd.read_csv(f'{STARTINGDIR}/Generation_{generation}_Database.csv')

MOLIDList = GenerationDatabase['ID'].to_list()

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
MOLSMILESList = GenerationDatabase['SMILES']
GenSimList = list(zip(MOLIDList, MOLSMILESList))

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
        print('Getting Toxicity')
        ToxNorm = GAF.Toxicity(MOLSMILES)

        print('Getting Density')
        DirRuns = GAF.list_generation_directories(CWD, 'Run')
        ExampleRun = DirRuns[0]

        DensityFile40 = f'{CWD}/{ExampleRun}/eqmDensity_{Molecule}_T313KP1atm.out'
        DensityFile100 = f'{CWD}/{ExampleRun}/eqmDensity_{Molecule}_T373KP1atm.out'

        print('Getting Viscosity')
        ### Viscosity
        DVisc40 = GAF.GetVisc(join(STARTINGDIR, 'Molecules', f'Generation_{generation}'), Molecule, 313)
        DVisc100 = GAF.GetVisc(join(STARTINGDIR, 'Molecules', f'Generation_{generation}'), Molecule, 373)
        Dens40 = GAF.GetDens(DensityFile40)
        Dens100 = GAF.GetDens(DensityFile100)
        print(DVisc40, DVisc100)

        print('Getting VI')
        ## Viscosity Index
        KVI = GAF.GetKVI(DVisc40, DVisc100, Dens40, Dens100, STARTINGDIR)
        DVI = GAF.GetDVI(DVisc40, DVisc100)

        #Update Molecule Database
        IDNumber = int(Molecule.split('_')[-1])
        MoleculeDatabase.at[IDNumber, 'Density100C'] = Dens100
        MoleculeDatabase.at[IDNumber, 'Density40C'] = Dens40
        MoleculeDatabase.at[IDNumber, 'DViscosity40C'] = DVisc40
        MoleculeDatabase.at[IDNumber, 'DViscosity100C'] = DVisc100
        MoleculeDatabase.at[IDNumber, 'KViscosity40C'] = GAF.GetKVisc(DVisc=DVisc40, Dens=Dens40)
        MoleculeDatabase.at[IDNumber, 'KViscosity100C'] = GAF.GetKVisc(DVisc=DVisc100, Dens=Dens100)
        MoleculeDatabase.at[IDNumber, 'KVI'] = KVI
        MoleculeDatabase.at[IDNumber, 'DVI'] = DVI
        MoleculeDatabase.at[IDNumber, 'Toxicity'] = ToxNorm
        MoleculeDatabase.at[IDNumber, 'SCScore'] = SCScoreNorm
        MoleculeDatabase.at[IDNumber, 'SimilarityScore'] = SCScoreNorm

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

# Apply the normalization function
Viscosity_normalized_molecule_scores = [(1-x[1]) for x in GAF.min_max_normalize(ViscosityScore)]
DVI_normalized_molecule_scores = [x[1] for x in GAF.min_max_normalize(DVIScore)]

MoleculeDatabase['ViscNormalisedScore'] = Viscosity_normalized_molecule_scores
MoleculeDatabase['DVINormalisedScore'] = DVI_normalized_molecule_scores
MoleculeDatabase['TotalScore'] = MoleculeDatabase['Toxicity'] + MoleculeDatabase['SCScore'] + MoleculeDatabase['DVINormalisedScore'] + MoleculeDatabase['ViscNormalisedScore'] 
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
GenerationDatabase.to_csv(f'{STARTINGDIR}/{generation}Database.csv')
