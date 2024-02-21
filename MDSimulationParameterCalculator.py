"""
Date: 21st February 2024

Script to detemine parameters from simulations benchmarking our viscosity calculations

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
import sys

WORKINGDIR = 'F:/PhD/HIGH_THROUGHPUT_STUDIES/MDSimulationEvaluation'

def get_plot_params(logfile, System, Run, WORKINGDIR, Temp, showplot=False):
    chdir(join(WORKINGDIR, System, Run))

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
                if len(line) != 16:
                    pass
                
                elif line[0].isnumeric():
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
                    kappa = (float(line[13]) + float(line[14]) + float(line[15])) / 30

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
        # Vplot.set_xlim(-0.05, 3)
        plt.savefig(join(WORKINGDIR, System, Run, f'ViscPlot_{Temp}.png'))
        if showplot:
            plt.show()
        plt.close()
        
        # Plot Kappa evolution
        KappaPlt, Kplot = plt.subplots()
        Kplot.set_title(f'Thermal Conductivity - {Temp}')
        Kplot.set_ylabel('Thermal Conductivity (W/m$^2$)')
        Kplot.set_xlabel('Time (ns)')
        Kplot.plot(step, kappalist)
        Kplot.grid(color='grey', linestyle='--', linewidth=0.5)
        Kplot.grid(which="minor", linestyle='--', linewidth=0.2)
        plt.minorticks_on()
        # Kplot.set_xlim(-0.05, 3)
        plt.savefig(join(WORKINGDIR, System, Run, f'KappaPlot_{Temp}.png'))
        if showplot:
            plt.show()
        plt.close()

        # Save final Visc and Kappa                
        visc_final = (v11[-1] + v22[-1] + v33[-1]) / 3
        kappa_final = (k11[-1] + k22[-1] + k33[-1]) / 30

        # Add 
    
    except Exception as E:
        print(E)
        print(System, Run)
        visc_final, kappa_final, step = None, None, None

    return visc_final, kappa_final, step, visclist, kappalist

# Get values from Long Simulations

Studies = ['Trajectory_Studies']
Molecules = ('squalane', '1-methylnapthalene', '11-octyl_13-methyl_henicosane')

for Study in Studies:
    for Molecule in Molecules:
        Runs = os.listdir(f'F:\PhD\HIGH_THROUGHPUT_STUDIES\MDSimulationEvaluation\{Study}\{Molecule}')

        RunFolders = [x for x in Runs if 'Run' in x]

        FinalViscList = []
        FinalKappaList = []

        Dataframe = pd.DataFrame()

        for Run in RunFolders:
            System = f'{Study}/{Molecule}'
            logfile = f'logGKvisc_{Molecule}_T313KP1atm.out'
            visc, kappa, step, visclist, kappalist = get_plot_params(logfile, System, Run, WORKINGDIR, '313K', showplot=False)
            Dens40C = GAF.GetDens(f'eqmDensity_{Molecule}_T313KP1atm.out')
            FinalViscList.append(visc)
            FinalKappaList.append(kappa)
            Dataframe[f'{System}_40C_Viscosity_{Run}'] = pd.Series(visclist)
            Dataframe[f'{System}_40C_ThemalK_{Run}'] = pd.Series(kappalist)

        # Dataframe = Dataframe.dropna()
        Dataframe.to_csv(join(WORKINGDIR, System, 'Data.csv'))
        print(f'{System} average viscosity = {np.average(FinalViscList)}')
        print(f'{System} average thermal conductivity = {np.average(FinalKappaList)}')