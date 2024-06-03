import sys, argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import integrate
from scipy.constants import Boltzmann
from scipy.optimize import curve_fit
from tqdm import trange
from os.path import join as join
from os import chdir
from os import getcwd
from os import listdir
import os

# Define ACF using FFT
def acf(data):
    steps = data.shape[0]
    # print(steps)
    lag = steps 

    # Nearest size with power of 2 (for efficiency) to zero-pad the input data
    size = 2 ** np.ceil(np.log2(2 * steps - 1)).astype('int')

    # Compute the FFT
    FFT = np.fft.fft(data, size)

    # Get the power spectrum
    PWR = FFT.conjugate() * FFT

    # Calculate the auto-correlation from inverse FFT of the power spectrum
    COR = np.fft.ifft(PWR)[:steps].real

    autocorrelation = COR / np.arange(steps, 0, -1)

    return autocorrelation[:lag]

# Viscosity from Einstein relation
def einstein(timestep, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz, volume, kBT, Time):

    Pxxyy = (Pxx - Pyy) / 2
    Pyyzz = (Pyy - Pzz) / 2

    '''
    Calculate the viscosity from the Einstein relation 
    by integrating the components of the pressure tensor
    '''
    timestep = timestep * 10**(-12)

    Pxy_int = integrate.cumtrapz(y=Pxy, dx=timestep, initial=0)
    Pxz_int = integrate.cumtrapz(y=Pxz, dx=timestep, initial=0)
    Pyz_int = integrate.cumtrapz(y=Pyz, dx=timestep, initial=0)

    # Pxxyy_int = integrate.cumtrapz(y=Pxxyy, dx=timestep, initial=0)
    # Pyyzz_int = integrate.cumtrapz(y=Pyyzz, dx=timestep, initial=0)

    # integral = (Pxy_int**2 + Pxz_int**2 + Pyz_int**2 + Pxxyy_int**2 + Pyyzz_int**2) / 5
    integral = (Pxy_int**2 + Pxz_int**2 + Pyz_int**2) / 3

    viscosity = integral[1:] * (volume * 10**(-30) / (2 * kBT * Time[1:] * 10**(-12)) )

    return viscosity

path = 'F:/PhD/HIGH_THROUGHPUT_STUDIES/MDsimulationEvaluation/ValidationStudies12ACutoff_200mols_LOPLS_NoKSPACE/'
chdir(path)
STARTDIR = getcwd()

Names = [x for x in listdir(getcwd()) if os.path.isdir(x)]
Temps = [313, 373]

def GetVisc(STARTDIR, Molecule, Temp):
    chdir(join(STARTDIR, Molecule))
    Runs = [x for x in listdir(getcwd()) if os.path.isdir(x)]

    DataframeEinstein = pd.DataFrame()
    DataframeGK = pd.DataFrame()

    for Run in Runs:
        try:
            chdir(join(STARTDIR, Molecule, Run))
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            df = pd.read_csv(f'Stress_AVGOnespd_{Molecule}_T{Temp}KP1atm.out')
            # print(df.columns)
            # df = df.drop(columns=['TimeStep'])
            # print(df)

            unit = 'atm' #Pressure, bar or temperature
            datafile = f'Stress_AVGOnespd_{Molecule}_T{Temp}KP1atm.out'
            diag = False # Use the diagonal components for viscosity prediction
            steps = len(df) -1 # Num steps to read from the pressure tensor file
            timestep = 1 # What timestep are you using in the pressure tensor file
            temperature = Temp #System temp

            with open(f'logGKvisc_{Molecule}_T{Temp}KP1atm.out', "r") as file:
                content = file.readlines()
                for line in content:
                    linecontent = line.split(' ')
                    linecontent = [x for x in linecontent if x != '']
                    if len(linecontent) == 18:
                        try:
                            vol = linecontent[9]
                            volume = float(vol)
                        except:
                            pass

            # print(volume)
            each = 10 # Sample frequency

            # Conversion ratio from atm/bar to Pa
            if unit == 'Pa':
                conv_ratio = 1
            elif unit == 'atm':
                conv_ratio = 101325
            elif unit == 'bar':
                conv_ratio = 100000

            # Calculate the kBT value
            kBT = Boltzmann * temperature

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Initiate the pressure tensor component lists
            Pxx, Pyy, Pzz, Pxy, Pxz, Pyz = [], [], [], [], [], []

            # Read the pressure tensor elements from data file
            with open(datafile, "r") as file:
                next(file)
                next(file)

                for _ in range(steps):
                    line = file.readline()
                    step = list(map(float, line.split()))
                    Pxx.append(step[1]*conv_ratio)
                    Pyy.append(step[2]*conv_ratio)
                    Pzz.append(step[3]*conv_ratio)
                    Pxy.append(step[4]*conv_ratio)
                    Pxz.append(step[5]*conv_ratio)
                    Pyz.append(step[6]*conv_ratio)

            # Convert lists to numpy arrays
            Pxx = np.array(Pxx)
            Pyy = np.array(Pyy)
            Pzz = np.array(Pzz)
            Pxy = np.array(Pxy)
            Pxz = np.array(Pxz)
            Pyz = np.array(Pyz)

            # Generate the time array
            end_step = steps * timestep
            Time = np.linspace(0, end_step, num=steps, endpoint=False)

            viscosity = einstein(timestep, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz, volume, kBT, Time)

            # Save the running integral of viscosity as a csv file
            df = pd.DataFrame({"time(ps)" : Time[:viscosity.shape[0]:each], "viscosity(Pa.s)" : viscosity[::each]})

            DataframeEinstein[f'Viscosity_{Run}'] = viscosity[:]*1000

            Time = np.linspace(0, end_step, num=steps, endpoint=False)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Viscosity from Green-Kubo relation
            def green_kubo(timestep):
                # Calculate the ACFs
                Pxy_acf = acf(Pxy)
                Pxz_acf = acf(Pxz)
                Pyz_acf = acf(Pyz)

                avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf) / 3

                # Integrate the average ACF to get the viscosity
                timestep = timestep * 10**(-12)
                integral = integrate.cumtrapz(y=avg_acf, dx=timestep, initial=0)
                viscosity = integral * (volume * 10**(-30) / kBT)
                # print(viscosity)

                return avg_acf, viscosity

            avg_acf, viscosity = green_kubo(timestep)

            DataframeGK[f'Viscosity_{Run}'] = viscosity[:]*1000

            # Save running integral of the viscosity as a csv file
            # df = pd.DataFrame({"time(ps)" : Time[:viscosity.shape[0]:each], "viscosity(Pa.s)" : viscosity[::each]})

        except Exception as E:
            print(E)
            ViscosityAv = None
            ViscosityAvEinstein = None
            pass

    try:
        # Plot average value for each timestep

        DataframeEinstein = DataframeEinstein.dropna()
        DataframeGK = DataframeGK.dropna()
        DataframeGK['Average'] = DataframeGK.mean(axis=1)
        DataframeGK['STD'] = DataframeGK.std(axis=1)
        DataframeEinstein['Average'] = DataframeEinstein.mean(axis=1)
        DataframeEinstein['STD'] = DataframeEinstein.std(axis=1)

        DataframeGKViscList_Average = DataframeGK['Average'].to_list()
        DataframeGKViscList_AverageSTD = DataframeGK['STD'].to_list()
        DataframeGKViscList_Average = [float(x) for x in DataframeGKViscList_Average]
        DataframeGKViscList_AverageSTD = [float(x) for x in DataframeGKViscList_AverageSTD]
        DataframeEinsteinList_Average = DataframeEinstein['Average'].to_list()
        DataframeEinsteinList_AverageSTD = DataframeEinstein['STD'].to_list()
        DataframeEinsteinList_Average = [float(x) for x in DataframeEinsteinList_Average]
        DataframeEinsteinList_AverageSTD = [float(x) for x in DataframeEinsteinList_AverageSTD]

        step = list(range(0, len(DataframeGKViscList_Average)))
        step = [x/1000 for x in step]

        ViscosityAv = round((DataframeGKViscList_Average[-1]), 2)
        ViscosityAvEinstein = round((DataframeEinsteinList_Average[-1]), 2)

    except Exception as E:
        print(E)
        ViscosityAv = None
        ViscosityAvEinstein = None

    return ViscosityAv, ViscosityAvEinstein

viscosity = GetVisc(STARTDIR, Names[0], 313)
print(viscosity)

def Bootstrap(numsamples,trjlen,numtrj,viscosity,Time,fv,plot,popt2):
    #Perform calculate the viscosity of one bootstrapping sample
    Bootlist = np.zeros((numsamples,trjlen))
    for j in range(0,numsamples):
        rint=randint(0,numtrj-1)
        for k in range(0,trjlen):
            Bootlist[j][k] = viscosity[rint][k]
    average = np.zeros(trjlen)
    stddev = np.zeros(trjlen)
    for j in range(0,trjlen):
        average[j] = np.average(Bootlist.transpose()[j])
        stddev[j] = np.std(Bootlist.transpose()[j])
    Value = fv.fitvisc(Time,average,stddev,plot,popt2)
    return Value