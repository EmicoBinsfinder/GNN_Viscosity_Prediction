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
- Check for similar molecules
- Plot an illustration of 

CONSTRAINTS

- Allow only one mutation per generation
- 


CONSIDERATIONS

- Need to determine how to handle crossovers

- Are we only going to add to end points
    - Maybe a coinflip or some form of distribution to determine where alteration is made

- Decide a protocol for adding branches top 

- How to implement a validity check before evaluating the fitness of molecules in a generation and which validity checks to implement

"""

################# IMPORTS ###################

from rdkit import Chem
from rdkit.Chem import Draw
import rdkit
from rdkit.Chem import AllChem

############## Defining different molecule types ###################
naphthalene = Chem.MolFromSmiles('c12ccccc1cccc2')
benzoxazole = Chem.MolFromSmiles('n1c2ccccc2oc1')
indane = Chem.MolFromSmiles('c1ccc2c(c1)CCC2')
skatole = Chem.MolFromSmiles('CC1=CNC2=CC=CC=C12')
benzene = Chem.MolFromSmiles('c1ccccc1')
quinoline = Chem.MolFromSmiles('n1cccc2ccccc12')

my_molecules = [naphthalene, benzoxazole, indane, skatole, benzene, quinoline]

########### Display Image Code
pil_img = Draw.MolsToGridImage(my_molecules)
#pil_img.show()

############ Code to Highlight certain Bonds

# Could be useful to show generational changes as CReM does
"""
from rdkit.Chem.Draw import rdMolDraw2D
smi = 'c1cc(F)ccc1Cl'
mol = Chem.MolFromSmiles(smi)
patt = Chem.MolFromSmarts('ClccccF')
hit_ats = list(mol.GetSubstructMatch(patt))
hit_bonds = []
for bond in patt.GetBonds():
   aid1 = hit_ats[bond.GetBeginAtomIdx()]
   aid2 = hit_ats[bond.GetEndAtomIdx()]
   hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())
d = rdMolDraw2D.MolDraw2DSVG(500, 500) # or MolDraw2DCairo to get PNGs
rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=hit_ats,
                                   highlightBonds=hit_bonds)
"""

"""
Code for substructure searching 
m = Chem.MolFromSmiles('c1ccccc1O')
patt = Chem.MolFromSmarts('ccO')
m.HasSubstructMatch(patt)
True
m.GetSubstructMatch(patt)
(0, 5, 6)
"""

############ Removing Substructure code

# m = Chem.MolFromSmiles('CC(=O)O')
# patt = Chem.MolFromSmarts('C(=O)O')
# rm = AllChem.DeleteSubstructs(m,patt)
# rmSMILES = Chem.MolToSmiles(rm)

# img_before = Draw.MolToImage(m)
# img_before.show()

# img_after = Draw.MolToImage(rm)
# img_after.show()

################## Replacing Substructures

# Defining the pattern to be replaced 
# To implement as part of GA could just take subsections of SMILES strings per mutation

# repl = Chem.MolFromSmiles('OC')

# patt = Chem.MolFromSmarts('[$(NC(=O))]')
# m = Chem.MolFromSmiles('CC(=O)N')
# img_before = Draw.MolToImage(m)
# img_before.show()

# # Define molecule, the substructure to be replaced, and what is should be replaced with, in that order
# rms = AllChem.ReplaceSubstructs(m,patt,repl)

# Chem.MolToSmiles(rms[0])
# img_after = Draw.MolToImage(rms[0])
# img_after.show()


###### Code to remove side excess molecules from core molecule

m1 = Chem.MolFromSmiles('BrCCc1cncnc1C(=O)O')
img_before = Draw.MolToImage(m1)
img_before.show()

core = Chem.MolFromSmiles('c1cncnc1')
# tmp = Chem.ReplaceSidechains(m1,core)
# img_after = Draw.MolToImage(tmp)
# img_after.show()
#Chem.MolToSmiles(tmp)

####### Code to remove core molecule and be left with fragments

tmp = Chem.ReplaceCore(m1,core)
Chem.MolToSmiles(tmp)
img_after = Draw.MolToImage(tmp)
img_after.show()

# Code to get molecular fragments
"""
Really useful for determining longest continuous fragment and getting rid of nonsense, such as with VAE generated molecules
"""
#rs = Chem.GetMolFrags(tmp,asMols=True)


