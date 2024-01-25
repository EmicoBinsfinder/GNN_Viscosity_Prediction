from rdkit import Chem
from rdkit.Chem import Draw
import rdkit
from random import sample
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolDrawing, DrawingOptions
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdDepictor
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from random import choice as rnd
import random
from rdkit.Chem.Draw import rdMolDraw2D
from MoleculeDifferenceViewer import view_difference
from copy import deepcopy
from operator import itemgetter
import subprocess
import ast
import pandas as pd
import re
import requests
from math import log10
import os
from os.path import join
import Genetic_Algorithm_Functions as GAF

### Fragments
fragments = ['CCCC', 'CCCCC', 'C(CC)CCC', 'CCC(CCC)CC', 'CCCC(C)CC', 'CCCCCCCCC', 'CCCCCCCC', 'CCCCCC', 'C(CCCCC)C', 'c1ccccc1']
fragments = [Chem.MolFromSmiles(x) for x in fragments]

# AromaticMolecules = 'c1ccccc1'

def InsertAromatic(StartingMolecule, AromaticMolecule, InsertStyle='Within', showdiff=False, Verbose=False):

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

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = GAF.MolCheckandPlot(StartingMoleculeUnedited, 
                                                                                StartingMolecule, 
                                                                                showdiff)

                #For situation where bond after target atom is severed
                else:
                    StartingMolecule.AddBond(ArmtcAtoms[0], AtomRmv + 1, Chem.BondType.SINGLE) 
                    StartingMolecule.AddBond(ArmtcAtoms[1], AtomRmv, Chem.BondType.SINGLE)

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = GAF.MolCheckandPlot(StartingMoleculeUnedited, 
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

                    Mut_Mol, Mut_Mol_Sanitized,  MutMolSMILES = GAF.MolCheckandPlot(StartingMoleculeUnedited, 
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

Mutations = ['AddAtom', 'ReplaceAtom', 'ReplaceBond', 'RemoveAtom', 'AddFragment', 'InsertAromatic']

# GENETIC ALGORITHM HYPERPARAMETERS
Silent = False # Edit outputs to only print if this flag is False
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
MaxGenerations = 100
NumGenerations = 1
MaxMutationAttempts = 200
Fails = 0
NumMols = 50
Agent = 'Agent1'

PYTHONPATH = 'C:/Users/eeo21/AppData/Local/Programs/Python/Python310/python.exe'
STARTINGDIR = deepcopy(os.getcwd())
#PYTHONPATH = 'python3'
GAF.runcmd(f'mkdir Molecules')
os.chdir(join(os.getcwd(), 'Molecules'))
GAF.runcmd(f'mkdir Generation_1')
os.chdir(STARTINGDIR)

# Master Dataframe where molecules from all generations will be stored
MoleculeDatabase = pd.DataFrame(columns=['SMILES', 'MolObject', 'MutationList', 'HeavyAtoms', 'ID', 'Charge', 'MolMass', 'Predecessor', 'Score', 'Density100C', 'DViscosity40C',
                                        'DViscosity100C', 'KViscosity40C', 'KViscosity100C', 'KVI', 'DVI', 'ThermalConductivity', 'PourPoint', 'DiffusionCoefficient', 'Density40C'])

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
    if GAF.GenMolChecks(result, GenerationMolecules, MaxNumHeavyAtoms, MinNumHeavyAtoms, MaxAromRings=2) == None:
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
                LTCOMMAND = f"{join(os.getcwd(), 'rdlt.py')} --smi {MutMolSMILES} -n {Name} -l -c"
            else:
                LTCOMMAND = f"{join(os.getcwd(), 'rdlt.py')} --smi {MutMolSMILES} -n {Name} -c"
            
            #Attempt to parameterise with OPLS
            GAF.runcmd(f'{PYTHONPATH} {LTCOMMAND} > {STARTINGDIR}/{Name}.lt')

            #Get molecule charge
            charge = GAF.GetMolCharge(f'{os.getcwd()}/{Name}.lt')

            #If successful, generate a PDB of molecule to use with Packmol
            GAF.GeneratePDB(MutMolSMILES, PATH=join(STARTINGDIR, f'{Name}.pdb'))

            # Go into directory for this generation
            os.chdir(join(STARTINGDIR, 'Molecules', 'Generation_1'))
            
            # Make a directory for the current molecule if it can be parameterised 
            GAF.runcmd(f'mkdir {Name}')

            # Enter molecule specific directory
            os.chdir(join(os.getcwd(), f'{Name}'))

            #Check if file has already been made, skip if so, being sure not to make duplicate, otherwise move file to correct directory
            CWD = os.getcwd() #Need to declare otherwise will get CWD from location function is being called from
            GAF.CheckMoveFile(Name, STARTINGDIR, 'lt', CWD)
            GAF.CheckMoveFile(Name, STARTINGDIR, 'pdb', CWD)

            # Estimate starting box length
            BoxL = GAF.CalcBoxLen(MolMass=MolMass, TargetDens=0.8, NumMols=NumMols)

            # Make packmol files
            GAF.MakePackmolFile(Name, CWD, NumMols=NumMols, BoxL=BoxL)

            # Make Moltemplate files
            GAF.MakeMoltemplateFile(Name, CWD, NumMols=NumMols, BoxL=BoxL)

            # Update Molecule database
            MoleculeDatabase = GAF.DataUpdate(MoleculeDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
                                              MutationList=[None, None, Mutation], ID=Name, Charge=charge, MolMass=MolMass, Predecessor=Predecessor)

            # Generate list of molecules to simulate in this generation
            FirstGenSimList.append(Name)
            print(f'Final Molecule SMILES: {MutMolSMILES}') 
            IDcounter +=1
        
        except Exception as E:
            continue     
    FirstGenerationAttempts += 1

# Here is where we will get the various values generated from the MD simulations
for Molecule in FirstGenSimList:
    try:
        # Create a function to wait until all simulations from this generation are finished
        Score = GAF.fitfunc(Molecule, Generation=1)
        os.chdir(join(STARTINGDIR, 'Generation_1', Molecule))
        CWD = os.getcwd()

        #Get Densities
        Dens40 = float(GAF.GetDens(f'{CWD}/eqmDensity_{Molecule}_T313KP1atm.out'))
        Dens100 = float(GAF.GetDens(f'{CWD}/eqmDensity_{Molecule}_T373KP1atm.out'))

        #Get Viscosities
        DVisc40 = float(GAF.GetVisc(f'{CWD}/logGKvisc_{Molecule}_T313KP1atm.out'))
        DVisc100 = float(GAF.GetVisc(f'{CWD}/logGKvisc_{Molecule}_T373KP1atm.out'))

        #Attempt to get KVI and DVI
        KVI = GAF.GetKVI(DVisc40=DVisc40, DVisc100=DVisc100, Dens40=Dens40, Dens100=Dens100, STARTINGDIR=STARTINGDIR)
        DVI = GAF.GetDVI(DVisc40=DVisc40, DVisc100=DVisc100)

        #Update Molecule Database
        IDNumber = int(Molecule.split('_')[-1])
        MoleculeDatabase.at[IDNumber, 'Score'] = Score
        MoleculeDatabase.at[IDNumber, 'Density100C'] = Dens100
        MoleculeDatabase.at[IDNumber, 'Density40C'] = Dens40
        MoleculeDatabase.at[IDNumber, 'DViscosity40C'] = DVisc40
        MoleculeDatabase.at[IDNumber, 'DViscosity100C'] = DVisc100
        MoleculeDatabase.at[IDNumber, 'KViscosity40C'] = GAF.GetKVisc(DVisc=DVisc40, Dens=Dens40)
        MoleculeDatabase.at[IDNumber, 'KViscosity100C'] = GAF.GetKVisc(DVisc=DVisc100, Dens=Dens100)
        MoleculeDatabase.at[IDNumber, 'KVI'] = KVI
        MoleculeDatabase.at[IDNumber, 'DVI'] = DVI
    except:
        pass

#Check if at least 'NumElite' molecules have a usable KVI, if so, will just use those as best performing molecules
if MoleculeDatabase['KVI'].isna().sum() <= 35:
    GenerationMolecules = pd.Series(MoleculeDatabase.KVI.values, index=MoleculeDatabase.ID).dropna().to_dict()
# Else will compare values of molecules with highest KVisc at 100C, as this is main driver of KVI
elif MoleculeDatabase['KViscosity100C'].isna().sum() <= 35:
    GenerationMolecules = pd.Series(MoleculeDatabase.KViscosity100C.values, index=MoleculeDatabase.ID).dropna().to_dict()
# Else assume that Molecules with higher molecular mass will more likely have improved VI
else:
    GenerationMolecules = pd.Series(MoleculeDatabase.MolMass.values, index=MoleculeDatabase.ID).to_dict()

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

# Compare KVIs, or otherwise compare molecule with highest kinematic viscosity

MoleculeDatabase.to_csv(f'{STARTINGDIR}/MoleculeDatabase.csv')

for generation in range(2, MaxGenerations + 1):
    GenerationTotalAttempts = 0
    GenSimList = []

    GenerationDatabase = pd.DataFrame(columns=['SMILES', 'MolObject', 'MutationList', 'HeavyAtoms', 'ID', 'Charge', 'MolMass', 'Predecessor', 'Score', 'Density100C', 'Viscosity40C',
                                        'Viscosity100C', 'VI', 'ThermalConductivity', 'PourPoint', 'DiffusionCoefficient', 'Density40C'])

    os.chdir(STARTINGDIR)
    # Store x best performing molecules (x=NumElite in list for next generation, without mutating them)
    GenerationMoleculeList = ScoreSortedMolecules[:NumElite]
    os.chdir(join(os.getcwd(), 'Molecules')) 
    GAF.runcmd(f'mkdir Generation_{generation}')
    os.chdir(STARTINGDIR)

    for i, entry in enumerate(ScoreSortedMolecules): #Start by mutating best performing molecules from previous generation and work down
        MutMol = None
        attempts = 0

        # Stop appending mutated molecules once generation reaches desired size
        if len(GenerationMoleculeList) == GenerationSize:
            break

        # Attempt mutation on each molecule, not moving on until a valid mutation has been suggested
        while MutMol == None:
            attempts += 1
            GenerationTotalAttempts += 1

            # Limit number of attempts at mutation, if max attempts exceeded, break loop to attempt on next molecule
            if attempts >= MaxMutationAttempts:
                Fails += 1 
                break

            # Starting Molecule mol object
            StartingMolecule = entry[1]
            # List containing last two successful mutations performed on molecule
            PreviousMutations = entry[2]
            # Number of heavy atoms
            NumHeavyAtoms = entry[3]
            # Molecule ID
            Name = f'Generation_{generation}_Molecule_{IDcounter}'

            if NumHeavyAtoms > 42:
                MutationList = ['RemoveAtom', 'ReplaceAtom', 'ReplaceBond']
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
            
            if GAF.GenMolChecks(result, GenerationMolecules, MaxNumHeavyAtoms, MinNumHeavyAtoms, MaxAromRings=2) == None:
                MutMol = None
        
            else:
                HeavyAtoms = result[0].GetNumHeavyAtoms() # Get number of heavy atoms in molecule
                MutMol = result[0] # Get Mol object of mutated molecule
                MolMass = GAF.GetMolMass(MutMol) # Get estimate of of molecular mass 
                MutMolSMILES = result[2] # SMILES of mutated molecule
                Predecessor = result[3] # Get history of last two mutations performed on candidate

                # Update previous mutations object
                PreviousMutations.pop(0)
                PreviousMutations.append(Mutation)

                print(f'Final SMILES: {result[2]}')

                try: # Try to generate all necessary files to simulate molecule
                    # Set feature definition file path to OPLS or LOPLS depending on user choice 
                    if LOPLS:
                        LTCOMMAND = f"{join(os.getcwd(), 'rdlt.py')} --smi {MutMolSMILES} -n {Name} -l -c"
                    else:
                        LTCOMMAND = f"{join(os.getcwd(), 'rdlt.py')} --smi {MutMolSMILES} -n {Name} -c"
                    
                    # Return to starting directory
                    os.chdir(STARTINGDIR) 
                    
                    #Attempt to parameterise with (L)OPLS
                    GAF.runcmd(f'{PYTHONPATH} {LTCOMMAND} > {STARTINGDIR}/{Name}.lt')

                    #Get molecule charge
                    charge = GAF.GetMolCharge(f'{os.getcwd()}/{Name}.lt')

                    #If successful, generate a PDB of molecule to use with Packmol
                    GAF.GeneratePDB(MutMolSMILES, PATH=join(STARTINGDIR, f'{Name}.pdb'))

                    # Go into directory for this generation
                    os.chdir(join(STARTINGDIR, 'Molecules', f'Generation_{generation}'))
                    
                    # Make a directory for the current molecule if it can be parameterised 
                    GAF.runcmd(f'mkdir {Name}')

                    # Enter molecule specific directory
                    os.chdir(join(os.getcwd(), f'{Name}'))

                    #Check if file has already been made, skip if so, being sure not to make duplicate, otherwise move file to correct directory
                    CWD = os.getcwd()
                    GAF.CheckMoveFile(Name, STARTINGDIR, 'lt', CWD)
                    GAF.CheckMoveFile(Name, STARTINGDIR, 'pdb', CWD)

                    # Make packmol files
                    GAF.MakePackmolFile(Name, CWD, NumMols=NumMols, BoxL=BoxL)

                    # Make Moltemplate files
                    GAF.MakeMoltemplateFile(Name, CWD, NumMols=NumMols, BoxL=BoxL)

                    # Make LAMMPS files
                    GAF.MakeLAMMPSFile(Name, CWD, Temp=313, GKRuntime=3000000)
                    GAF.MakeLAMMPSFile(Name, CWD, Temp=373, GKRuntime=3000000)

                    if PYTHONPATH == 'python3':
                        GAF.runcmd(f'packmol < {Name}.inp') # Run packmol in command line
                        GAF.runcmd(f'moltemplate.sh -pdb {Name}_PackmolFile.pdb {Name}_system.lt') # Run moltemplate in command line

                        # Check that Moltemplate has generated all necessary files 
                        assert os.path.exists(join(CWD, f'{Name}_system.in.settings')), 'Settings file not generated'                 
                        assert os.path.exists(join(CWD, f'{Name}_system.in.charges')), 'Charges file not generated'
                        assert os.path.exists(join(CWD, f'{Name}_system.data')), 'Data file not generated'                    
                        assert os.path.exists(join(CWD, f'{Name}_system.in.init')), 'Init file not generated'                

                    # Return to starting directory
                    os.chdir(STARTINGDIR)

                    # Add candidate and it's data to master list
                    MoleculeDatabase = GAF.DataUpdate(MoleculeDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
                                            MutationList=PreviousMutations, ID=Name, Charge=charge, MolMass=MolMass, Predecessor=Predecessor)
                    
                    GenerationDatabase = GAF.DataUpdate(GenerationDatabase, IDCounter=IDcounter, MutMolSMILES=MutMolSMILES, MutMol=MutMol, HeavyAtoms=HeavyAtoms,
                        MutationList=PreviousMutations, ID=Name, Charge=charge, MolMass=MolMass, Predecessor=Predecessor)

                    # Generate list of molecules to simulate in this generation
                    GenSimList.append(Name)
                    print(f'Final Molecule SMILES: {MutMolSMILES}') 
                    GenerationMoleculeList.append([result[2], result[0], PreviousMutations, NumHeavyAtoms, ID, MolMass, Predecessor])
                    IDcounter += 1

                except Exception as E:
                    print(E)
                    continue   

    #This is where we should call the simulation script
    CWD = join(STARTINGDIR, 'Molecules', f'Generation_{generation}')
    # Create array job for 40C viscosity
    GAF.CreateArrayJob(STARTINGDIR, CWD, generation, SimName=f"{Agent}_313K.lammps", Agent=Agent)
    # Create array job for 100C viscosity
    GAF.CreateArrayJob(STARTINGDIR, CWD, generation, SimName=f"{Agent}_373K.lammps", Agent=Agent)

    for Molecule in GenSimList:
        try:
            # Create a function to wait until all simulations from this generation are finished
            Score = GAF.fitfunc(Molecule, Generation=generation)
            os.chdir(join(STARTINGDIR, generation, Molecule))
            CWD = os.getcwd()

            #Get Densities
            Dens40 = float(GAF.GetDens(f'{CWD}/eqmDensity_{Molecule}_T313KP1atm.out'))
            Dens100 = float(GAF.GetDens(f'{CWD}/eqmDensity_{Molecule}_T373KP1atm.out'))

            #Get Viscosities
            DVisc40 = float(GAF.GetVisc(f'{CWD}/logGKvisc_{Molecule}_T313KP1atm.out'))
            DVisc100 = float(GAF.GetVisc(f'{CWD}/logGKvisc_{Molecule}_T373KP1atm.out'))

            #Attempt to get KVI and DVI
            KVI = GAF.GetKVI(DVisc40=DVisc40, DVisc100=DVisc100, Dens40=Dens40, Dens100=Dens100, STARTINGDIR=STARTINGDIR)
            DVI = GAF.GetDVI(DVisc40=DVisc40, DVisc100=DVisc100)

            #Update Molecule Database
            IDNumber = int(Molecule.split('_')[-1])
            MoleculeDatabase.at[IDNumber, 'Score'] = Score
            MoleculeDatabase.at[IDNumber, 'Density100C'] = Dens100
            MoleculeDatabase.at[IDNumber, 'Density40C'] = Dens40
            MoleculeDatabase.at[IDNumber, 'DViscosity40C'] = DVisc40
            MoleculeDatabase.at[IDNumber, 'DViscosity100C'] = DVisc100
            MoleculeDatabase.at[IDNumber, 'KViscosity40C'] = GAF.GetKVisc(DVisc=DVisc40, Dens=Dens40)
            MoleculeDatabase.at[IDNumber, 'KViscosity100C'] = GAF.GetKVisc(DVisc=DVisc100, Dens=Dens100)
            MoleculeDatabase.at[IDNumber, 'KVI'] = KVI
            MoleculeDatabase.at[IDNumber, 'DVI'] = DVI
        except:
            pass

    #Check if at least 'NumElite' molecules have a usable KVI, if so, will just use those as best performing molecules
    if MoleculeDatabase['KVI'].isna().sum() <= 35:
        GenerationMolecules = pd.Series(MoleculeDatabase.KVI.values, index=MoleculeDatabase.ID).dropna().to_dict()
    # Else will compare values of molecules with highest KVisc at 100C, as this is main driver of KVI
    elif MoleculeDatabase['KViscosity100C'].isna().sum() <= 35:
        GenerationMolecules = pd.Series(MoleculeDatabase.KViscosity100C.values, index=MoleculeDatabase.ID).dropna().to_dict()
    # Else assume that Molecules with higher molecular mass will more likely have improved VI
    else:
        GenerationMolecules = pd.Series(MoleculeDatabase.MolMass.values, index=MoleculeDatabase.ID).to_dict()

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
    
    MoleculeDatabase.to_csv(f'{STARTINGDIR}/MoleculeDatabase.csv')
    GenerationDatabase.to_csv(f'{STARTINGDIR}/Generation{generation}_Database.csv')

print(len(GenerationMoleculeList))
print(f'Number of failed mutations: {Fails}')
print(len(ScoreSortedMolecules[:NumElite]))




