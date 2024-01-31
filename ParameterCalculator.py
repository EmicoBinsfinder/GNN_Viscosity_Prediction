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
from matplotlib import pyplot as plt
import Genetic_Algorithm_Functions as GAF
from copy import deepcopy
from os import chdir
from os.path import join

Pressure = 1.0 #In atm
TargetTemp = 313
PressureUnits = 'Bar'
kBT = Boltzmann * TargetTemp
NumTensors = 3

def get_plot_params(logfile, Molecule, WORKINGDIR, Temp, showplot=False):
    Generation = Molecule.split('_')[1]
    chdir(join(WORKINGDIR, f'Generation_{Generation}', Molecule))

    try:
        ##### Read/Process File #####
        with open(f'{logfile}', 'r+') as file:
            contents = file.read()

        contents1 = contents.split('Mbytes')[-1]
        contents2 = contents1.split('Loop')[0]
        contents3 = contents2.split('\n')

        # Params for 40C run 
        step, v11, v22, v33, k11, k22, k33, visclist, kappalist = [], [], [], [], [], [], [], [], []
        for line in contents3[1:]:
            line = line.split(' ')
            line = [x for x in line if x] 
            
            # Seperate out different components
            if len(line) > 2:
                if line[0].isnumeric():

                    # Record timestep            
                    step.append(line[0])

                    # Append ACF values 
                    v11.append(float(line[6])*1000)
                    v22.append(float(line[7])*1000)
                    v33.append(float(line[8])*1000)
                    k11.append(float(line[13]))
                    k22.append(float(line[14]))
                    k33.append(float(line[15]))
                    
                    # Get latest kappa and viscosity value
                    visc = (float(line[6])*1000 + float(line[7])*1000 + float(line[8])*1000) / 3
                    kappa = (float(line[13]) + float(line[14]) + float(line[15])) / 3

                    visclist.append(visc)
                    kappalist.append(kappa)

        # Transform timesteps
        step = [(float(x) - 2000000)*1e-6 for x in step]

        # Plot Visc evolution
        ViscPlt, Vplot = plt.subplots()
        Vplot.set_title(f'Viscosity - {Temp}')
        Vplot.set_ylabel('Viscosity (Cp)')
        Vplot.set_xlabel('Time (ns)')
        Vplot.plot(step, visclist)
        Vplot.grid(color='grey', linestyle='--', linewidth=0.5)
        Vplot.grid(which="minor", linestyle='--', linewidth=0.2)
        plt.minorticks_on()
        Vplot.set_xlim(-0.05, 3)
        plt.savefig(join(WORKINGDIR, f'Generation_{Generation}', Molecule, f'ViscPlot_{Temp}.png'))
        plt.close()
        if showplot:
            plt.show()
        
        # Plot Kappa evolution
        KappaPlt, Kplot = plt.subplots()
        Kplot.set_title(f'Thermal Conductivity - {Temp}')
        Kplot.set_ylabel('Thermal Conductivity (W/m$^2$)')
        Kplot.set_xlabel('Time (ns)')
        Kplot.plot(step, kappalist)
        Kplot.grid(color='grey', linestyle='--', linewidth=0.5)
        Kplot.grid(which="minor", linestyle='--', linewidth=0.2)
        plt.minorticks_on()
        Kplot.set_xlim(-0.05, 3)
        plt.savefig(join(WORKINGDIR, f'Generation_{Generation}', Molecule, f'KappaPlot_{Temp}.png'))
        plt.close()
        if showplot:
            plt.show()

        # Save final Visc and Kappa                
        visc_final = (v11[-1] + v22[-1] + v33[-1]) / 3
        kappa_final = (k11[-1] + k22[-1] + k33[-1]) / 3
    
    except:
        visc_final, kappa_final, step = None, None, None

    return visc_final, kappa_final, step

# Plot Kappa evolution
MoleculeDatabase = pd.read_csv('MoleculeDatabaseOriginal.csv')

WORKINGDIR = 'C:/Users/eeo21/Desktop/Molecules'
STARTINGDIR = deepcopy(os.getcwd())
print(STARTINGDIR)
Molecules = MoleculeDatabase['ID'].to_list()
MoleculeDatabase = MoleculeDatabase.rename(columns={'ThermalConductivity': 'ThermalConductivity_40C', 'PourPoint': 'ThermalConductivity_100C'})

for Molecule in Molecules:
    Generation = Molecule.split('_')[1]
    IDNumber = int(Molecule.split('_')[3])
    chdir(join(WORKINGDIR, f'Generation_{Generation}', Molecule))

    visc_40, kappa_40, step_40 = get_plot_params(f'logGKvisc_{Molecule}_T313KP1atm.out', Molecule, WORKINGDIR, Temp='313K')
    visc_100, kappa_100, step_40 = get_plot_params(f'logGKvisc_{Molecule}_T373KP1atm.out', Molecule, WORKINGDIR, '373K')

    #Get densities 
    Dens40 = GAF.GetDens(f'eqmDensity_{Molecule}_T313KP1atm.out')
    Dens100 = GAF.GetDens(f'eqmDensity_{Molecule}_T373KP1atm.out')

    KVisc40 = GAF.GetKVisc(visc_40, Dens40)
    KVisc100 = GAF.GetKVisc(visc_100, Dens100)

    KVI = GAF.GetKVI(visc_40, visc_100, Dens40, Dens100, STARTINGDIR)
    DVI = GAF.GetDVI(visc_40, visc_100)

    MoleculeDatabase.at[IDNumber, 'Density100C'] = Dens100
    MoleculeDatabase.at[IDNumber, 'Density40C'] = Dens40
    MoleculeDatabase.at[IDNumber, 'DViscosity40C'] = visc_40
    MoleculeDatabase.at[IDNumber, 'DViscosity100C'] = visc_100
    MoleculeDatabase.at[IDNumber, 'KViscosity40C'] = KVisc40
    MoleculeDatabase.at[IDNumber, 'KViscosity100C'] = KVisc100
    MoleculeDatabase.at[IDNumber, 'ThermalConductivity_40C'] = kappa_40
    MoleculeDatabase.at[IDNumber, 'ThermalConductivity_100C'] = kappa_100
    MoleculeDatabase.at[IDNumber, 'KVI'] = KVI
    MoleculeDatabase.at[IDNumber, 'DVI'] = DVI

MoleculeDatabase.to_csv(f'{STARTINGDIR}/MoleculeDatabase1.csv', index=False)
print(MoleculeDatabase)

# print(KVI)
# print(DVI)
# print(visc_40)
# print(visc_100)
# print(KVisc40)
# print(KVisc100)
# print(Dens40)
# print(Dens100)


