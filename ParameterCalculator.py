"""
Author: Egheosa Ogbomo
Date: 25th January 2024

Script to calculate MSD, Thermoconductivity and Viscosity from LAMMPS simulations

Required inputs:

- File name
- Pressure units
- Number of timesteps in pressure file (could read this ourselves)
- First timestep to start processing from
- Target temperature
- Volume of the production run
- Whether to calculate via GK or Einstein method
- Whether to use 3 or 6 tensor with GK method
- Whether to plot autocorrelation functions
- Whether to plot evolution of parameters
- Frequency of parameter evolution calculations

"""

import os
import time
import pylab
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.constants import Boltzmann
from multiprocessing import Process
from tqdm import trange


Pressure = 1.0 #In atm
TargetTemp = 313
PressureUnits = 'Bar'
kBT = Boltzmann * TargetTemp
NumTensors = 3

##### Read/Process File #####
with open('logGKvisc_Generation_2_Molecule_59_T313KP1atm.out', 'r+') as file:
    contents = file.read()

contents1 = contents.split('Mbytes')[-1]
contents2 = contents1.split('Loop')[0]
contents3 = contents2.split('\n')

for line in contents3[1:200]:
    line = line.split(' ')
    line = [x for x in line if x] 
    
    step, v11, v22, v33, k11, k22, k33 = [], [], [], [], [], [], [] 
    
    # Seperate out different components
    if line[0].isnumeric():
        step.append(line[0])
        v11.append(line[7]*1000)
        v22.append(line[8]*1000)
        v33.append(line[9]*1000)
        k11.append(line[13])
        k22.append(line[14])
        k33.append(line[15])

    else:
        pass

