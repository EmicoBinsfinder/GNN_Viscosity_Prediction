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

    except Exception as E:
        print(E)
        print(System, Run)
        visc_final, kappa_final, step = None, None, None

    return visc_final, kappa_final, step, visclist, kappalist

# Get values from Long Simulations and Trajectory Studies

Studies = ['Trajectory_Studies', 'LongSimEffects']
Molecules = ['squalane', '1-methylnapthalene', '11-octyl_13-methyl_henicosane']

for Study in Studies:
    for Molecule in Molecules:
        Runs = os.listdir(f'F:\PhD\HIGH_THROUGHPUT_STUDIES\MDSimulationEvaluation\{Study}\{Molecule}')

        RunFolders = [x for x in Runs if 'Run' in x]

        FinalViscList = []
        FinalKappaList = []
        DataframeVisc = pd.DataFrame()
        DataframeKappa = pd.DataFrame()

        for Run in RunFolders:
            System = f'{Study}/{Molecule}'
            logfile = f'logGKvisc_{Molecule}_T373KP1atm.out'
            visc, kappa, step, visclist, kappalist = get_plot_params(logfile, System, Run, WORKINGDIR, '373K', showplot=False)
            Dens40C = GAF.GetDens(f'eqmDensity_{Molecule}_T373KP1atm.out')
            FinalViscList.append(visc)
            FinalKappaList.append(kappa)
            DataframeVisc[f'{System}_40C_Viscosity_{Run}'] = pd.Series(visclist)
            DataframeKappa[f'{System}_40C_ThemalK_{Run}'] = pd.Series(kappalist)

        DataframeVisc.to_csv(join(WORKINGDIR, System, 'DataVisc373K.csv'), index=False)
        DataframeKappa.to_csv(join(WORKINGDIR, System, 'DataKappa373K.csv'), index=False)

        DataframeVisc = pd.read_csv(join(WORKINGDIR, System, 'DataVisc373K.csv'), index_col=False)
        DataframeKappa = pd.read_csv(join(WORKINGDIR, System, 'DataKappa373K.csv'), index_col=False)

        # Remove incomplete experiments
        for column in DataframeVisc.columns:
            if len(DataframeVisc) != len(pd.Series(DataframeVisc[column])):
                DataframeVisc = DataframeVisc.drop([column], axis=1)
        
        for column in DataframeKappa.columns:
            if len(DataframeKappa) != len(pd.Series(DataframeKappa[column])):
                DataframeKappa = DataframeKappa.drop([column], axis=1)

        # Plot average value for each timestep
        DataframeVisc['Average'] = DataframeVisc.mean(axis=1)
        DataframeVisc['STD'] = DataframeVisc.std(axis=1)
        DataframeKappa['Average'] = DataframeKappa.mean(axis=1)
        DataframeKappa['STD'] = DataframeKappa.std(axis=1)

        # Save average from each time step
        AvViscList = DataframeVisc['Average'].to_list()
        AvViscListSTD = DataframeVisc['STD'].to_list()
        AvViscList = [float(x) for x in AvViscList]
        AvViscListSTD = [float(x) for x in AvViscListSTD]
        Visc_UpperSTD = [a + b for a, b in zip(AvViscList, AvViscListSTD)]
        Visc_LowerSTD = [a - b for a, b in zip(AvViscList, AvViscListSTD)]

        AvKappaList = DataframeKappa['Average'].to_list()
        AvKappaListSTD = DataframeKappa['STD'].to_list()
        AvKappaList = [float(x) for x in AvKappaList]
        AvKappaListSTD = [float(x) for x in AvKappaListSTD]
        Kappa_LowerSTD = [a + b for a, b in zip(AvKappaList, AvKappaListSTD)]
        Kappa_UpperSTD = [a - b for a, b in zip(AvKappaList, AvKappaListSTD)]

        # Transform timesteps
        step = list(range(0, len(AvViscList)))
        step = [x/1000 for x in step]

        #Get Standard Deviation 

        showplot = True

        # Plot Visc evolution
        ViscPlt, Vplot = plt.subplots()
        Vplot.set_title(f'Viscosity Average- 373K {Study} {Molecule}')
        Vplot.set_ylabel('Viscosity (Cp)')
        Vplot.set_xlabel('Time (ns)')
        Vplot.plot(step, AvViscList)
        Vplot.fill_between(step, Visc_LowerSTD, Visc_UpperSTD, alpha=0.4)
        Vplot.grid(color='grey', linestyle='--', linewidth=0.5)
        Vplot.grid(which="minor", linestyle='--', linewidth=0.2)
        plt.minorticks_on()
        # Vplot.set_xlim(-0.05, 3)
        plt.savefig(join(WORKINGDIR, f'{Study}',  f'AvViscPlot_{Molecule}_{Study}_373K.png'))
        plt.close()
        
        # Plot Kappa evolution
        KappaPlt, Kplot = plt.subplots()
        Kplot.set_title(f'Thermal Conductivity Average - 373K {Study} {Molecule}')
        Kplot.set_ylabel('Thermal Conductivity (W/m$^2$)')
        Kplot.set_xlabel('Time (ns)')
        Kplot.plot(step, AvKappaList)
        Kplot.fill_between(step, Kappa_LowerSTD, Kappa_UpperSTD, alpha=0.4)
        Kplot.grid(color='grey', linestyle='--', linewidth=0.5)
        Kplot.grid(which="minor", linestyle='--', linewidth=0.2)
        plt.minorticks_on()
        # Kplot.set_xlim(-0.05, 3)
        plt.savefig(join(WORKINGDIR, f'{Study}', f'AvKappaPlot_{Molecule}_{Study}_373K.png'))
        plt.close()
            
        print(f'{System} average viscosity = {np.average(FinalViscList)}')
        print(f'{System} average thermal conductivity = {np.average(FinalKappaList)}')
    
# Get values from FiniteSizeEffect studies

NumMols = ['NumMols_25', 'NumMols_50', 'NumMols_100', 'NumMols_250', 'NumMols_500']

for Molecule in Molecules:
    for NumMol in NumMols:        
        Runs = os.listdir(f'F:\PhD\HIGH_THROUGHPUT_STUDIES\MDSimulationEvaluation\FiniteSizeEffects\{Molecule}\{NumMol}')

        RunFolders = [x for x in Runs if 'Run' in x]

        FinalViscList = []
        FinalKappaList = []
        DataframeVisc = pd.DataFrame()
        DataframeKappa = pd.DataFrame()

        for Run in RunFolders:
            System = f'FiniteSizeEffects/{Molecule}/{NumMol}'
            logfile = f'logGKvisc_{Molecule}_T373KP1atm.out'
            visc, kappa, step, visclist, kappalist = get_plot_params(logfile, System, Run, WORKINGDIR, '373K', showplot=False)
            Dens40C = GAF.GetDens(f'eqmDensity_{Molecule}_T373KP1atm.out')
            FinalViscList.append(visc)
            FinalKappaList.append(kappa)
            DataframeVisc[f'{System}_40C_Viscosity_{Run}'] = pd.Series(visclist)
            DataframeKappa[f'{System}_40C_ThemalK_{Run}'] = pd.Series(kappalist)

        DataframeVisc.to_csv(join(WORKINGDIR, System, 'DataVisc373.csv'), index=False)
        DataframeKappa.to_csv(join(WORKINGDIR, System, 'DataKappa373.csv'), index=False)

        # # Dataframe = Dataframe.dropna()
        DataframeVisc = pd.read_csv(join(WORKINGDIR, System, 'DataVisc373.csv'), index_col=False)
        DataframeKappa = pd.read_csv(join(WORKINGDIR, System, 'DataKappa373.csv'), index_col=False)

        # Remove incomplete experiments
        for column in DataframeVisc.columns:
            if len(DataframeVisc) != len(pd.Series(DataframeVisc[column])):
                DataframeVisc = DataframeVisc.drop([column], axis=1)
        
        for column in DataframeKappa.columns:
            if len(DataframeKappa) != len(pd.Series(DataframeKappa[column])):
                DataframeKappa = DataframeKappa.drop([column], axis=1)

        # Plot average value for each timestep
        DataframeVisc['Average'] = DataframeVisc.mean(axis=1)
        DataframeVisc['STD'] = DataframeVisc.std(axis=1)
        DataframeKappa['Average'] = DataframeKappa.mean(axis=1)
        DataframeKappa['STD'] = DataframeKappa.std(axis=1)

        # Save average from each time step
        AvViscList = DataframeVisc['Average'].to_list()
        AvViscListSTD = DataframeVisc['STD'].to_list()
        AvViscList = [float(x) for x in AvViscList]
        AvViscListSTD = [float(x) for x in AvViscListSTD]
        Visc_UpperSTD = [a + b for a, b in zip(AvViscList, AvViscListSTD)]
        Visc_LowerSTD = [a - b for a, b in zip(AvViscList, AvViscListSTD)]

        AvKappaList = DataframeKappa['Average'].to_list()
        AvKappaListSTD = DataframeKappa['STD'].to_list()
        AvKappaList = [float(x) for x in AvKappaList]
        AvKappaListSTD = [float(x) for x in AvKappaListSTD]
        Kappa_LowerSTD = [a + b for a, b in zip(AvKappaList, AvKappaListSTD)]
        Kappa_UpperSTD = [a - b for a, b in zip(AvKappaList, AvKappaListSTD)]

        # Transform timesteps
        step = list(range(0, len(AvViscList)))
        step = [x/1000 for x in step]

        #Get Standard Deviation 

        showplot = True

        # Plot Visc evolution
        ViscPlt, Vplot = plt.subplots()
        Vplot.set_title(f'Viscosity Average- 373K {NumMol} {Molecule}')
        Vplot.set_ylabel('Viscosity (Cp)')
        Vplot.set_xlabel('Time (ns)')
        Vplot.plot(step, AvViscList)
        Vplot.fill_between(step, Visc_LowerSTD, Visc_UpperSTD, alpha=0.4)
        Vplot.grid(color='grey', linestyle='--', linewidth=0.5)
        Vplot.grid(which="minor", linestyle='--', linewidth=0.2)
        plt.minorticks_on()
        # Vplot.set_xlim(-0.05, 3)
        plt.savefig(join(WORKINGDIR, 'FiniteSizeEffects', f'AvViscPlot_{Molecule}_{NumMol}_373K.png'))
        plt.close()
        
        # Plot Kappa evolution
        KappaPlt, Kplot = plt.subplots()
        Kplot.set_title(f'Thermal Conductivity Average - 373K {NumMol} {Molecule}')
        Kplot.set_ylabel('Thermal Conductivity (W/m$^2$)')
        Kplot.set_xlabel('Time (ns)')
        Kplot.plot(step, AvKappaList)
        Kplot.fill_between(step, Kappa_LowerSTD, Kappa_UpperSTD, alpha=0.4)
        Kplot.grid(color='grey', linestyle='--', linewidth=0.5)
        Kplot.grid(which="minor", linestyle='--', linewidth=0.2)
        plt.minorticks_on()
        # Kplot.set_xlim(-0.05, 3)
        plt.savefig(join(WORKINGDIR, 'FiniteSizeEffects', f'AvKappaPlot_{Molecule}_{NumMol}_373K.png'))
        plt.close()
            
        print(f'{System} average viscosity = {np.average(FinalViscList)}')
        print(f'{System} average thermal conductivity = {np.average(FinalKappaList)}')