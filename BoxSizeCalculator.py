"""
Date: 29th Decemeber 2023

Script to calculate molar mass of molecule from SMILES string
Using molar mass, determine appropriate 

"""

import re
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

def get_mass(formula):

    parts = re.findall("[A-Z][a-z]?|[0-9]+", formula)
    mass = 0

    for index in range(len(parts)):
        if parts[index].isnumeric():
            continue

        atom = Chem.Atom(parts[index])
        multiplier = int(parts[index + 1]) if len(parts) > index + 1 and parts[index + 1].isnumeric() else 1
        mass += atom.GetMass() * multiplier

    return mass

mol = Chem.MolFromSmiles('CCCCCCC(CC)CCCC')
formula = CalcMolFormula(mol)

print(formula)
print(get_mass(formula))