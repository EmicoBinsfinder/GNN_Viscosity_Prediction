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
    print(autocorrelation)

    return autocorrelation[:lag]

# def acf(data):
#     """
#     Compute the autocorrelation function of the given data.
#     """
#     n = len(data)
#     mean = np.mean(data)
#     var = np.var(data)
#     print(var)
#     acf = np.correlate(data - mean, data - mean, mode='full') / (var * n)
#     return acf[n-1:]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Viscosity from Einstein relation
def einstein(timestep):

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

    viscosity = integral[1:] * ( volume * 10**(-30) / (2 * kBT * Time[1:] * 10**(-12)) )

    return viscosity

chdir('F:/PhD/HIGH_THROUGHPUT_STUDIES/MDsimulationEvaluation/ValidationStudies12ACutoff_200mols_LOPLS_NoKSPACE/')
STARTDIR = getcwd()

Names = [x for x in listdir(getcwd()) if os.path.isdir(x)]
Temps = [313, 373]

for Name in Names:
    print(Name)
    for Temp in Temps:
            print(Temp)
            chdir(join(STARTDIR, Name))
            Runs = [x for x in listdir(getcwd()) if os.path.isdir(x)]

            DataframeEinstein = pd.DataFrame()
            DataframeGK = pd.DataFrame()

            for Run in Runs:
                try:
                    chdir(join(STARTDIR, Name, Run))
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    df = pd.read_csv(f'Stress_AVGOnespd_{Name}_T{Temp}KP1atm.out')
                    # print(df.columns)
                    # df = df.drop(columns=['TimeStep'])
                    # print(df)

                    unit = 'atm' #Pressure, bar or temperature
                    datafile = f'Stress_AVGOnespd_{Name}_T{Temp}KP1atm.out'
                    diag = False # Use the diagonal components for viscosity prediction
                    steps = len(df) -1 # Num steps to read from the pressure tensor file
                    timestep = 1 # What timestep are you using in the pressure tensor file
                    temperature = Temp #System temp

                    with open(f'logGKvisc_{Name}_T{Temp}KP1atm.out', "r") as file:
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
                    each = 2 # Sample frequency
                    plot = True # Create plot for autocorrelation function

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

                    viscosity = einstein(timestep=timestep)

                    # print(f"\nViscosity (Einstein): {round((viscosity[-1] * 1000), 2)} [mPa.s]")
                    viscE = round((viscosity[-1] * 1000), 2)

                    # Plot the running integral of viscosity
                    if plot:
                        plt.figure()
                        plt.plot(Time[:viscosity.shape[0]], viscosity[:]*1000, label='Viscosity')
                        plt.xlabel('Time (ps)')
                        plt.ylabel('Einstein Viscosity (mPa.s)')
                        plt.legend([f'Viscosity Estimate: {viscE} [mPa.s]'])
                        plt.title(f'{Name}_{Run}_{Temp}_Einstein')
                        plt.xlim(0, 1)
                        plt.ylim(0, 35)
                        plt.savefig(join(STARTDIR, Name, f'{Name}_{Run}_{Temp}_Einstein.png'))
                        plt.close()

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

                        # Calculate the shear components of the pressure tensor and their ACF
                        # if diag:
                        #     Pxxyy = (Pxx - Pyy) / 2
                        #     Pyyzz = (Pyy - Pzz) / 2
                        #     Pxxzz = (Pxx - Pzz) / 2

                        #     Pxxyy_acf = acf(Pxxyy)
                        #     Pyyzz_acf = acf(Pyyzz)
                        #     Pxxzz_acf = acf(Pxxzz)

                        # if diag:
                        #     avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf + Pxxyy_acf + Pyyzz_acf + Pxxzz_acf) / 6
                        avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf) / 3


                        # Integrate the average ACF to get the viscosity
                        timestep = timestep * 10**(-12)
                        integral = integrate.cumtrapz(y=avg_acf, dx=timestep, initial=0)
                        viscosity = integral * (volume * 10**(-30) / kBT)
                        # print(viscosity)

                        return avg_acf, viscosity

                    avg_acf, viscosity = green_kubo(timestep)

                    # Plot the normalized average ACF
                    if plot:
                        norm_avg_acf = avg_acf / avg_acf[0]
                        plt.figure()
                        plt.plot(Time[:avg_acf.shape[0]], norm_avg_acf[:], label='Average')
                        plt.xlabel('Time (ps)')
                        plt.ylabel('Green Kubo ACF')
                        plt.legend()
                        plt.title(f'{Name} {Temp}K Green Kubo ACF')
                        # plt.xlim(0, 1)
                        plt.grid(color='grey', linestyle='--', linewidth=0.5)
                        plt.grid(which="minor", linestyle='--', linewidth=0.2)
                        plt.minorticks_on()
                        plt.savefig(join(STARTDIR, Name, f'{Name}_{Run}_{Temp}_2nsGKACF.png'))
                        plt.close()

                    # Save the normalized average ACF as a csv file
                    norm_avg_acf = avg_acf / avg_acf[0]
                    # df = pd.DataFrame({"time (ps)" : Time[:avg_acf.shape[0]], "ACF" : norm_avg_acf[:]})
                    # df.to_csv("avg_acf.csv", index=False)

                    # print(f"Viscosity (Green-Kubo): {round((viscosity[-1] * 1000), 2)} [mPa.s]")


                    # Plot the time evolution of the viscosity estimate
                    if plot:
                        plt.figure()
                        plt.plot(Time[:viscosity.shape[0]], viscosity[:]*1000, label='Viscosity')
                        plt.xlabel('Time (ps)')
                        plt.ylabel('Green Kubo Viscosity (mPa.s)')
                        plt.legend([f'Viscosity Estimate: {round((viscosity[-1] * 1000), 2)} [mPa.s]'])
                        plt.title(f'{Name} {Temp}K Green Kubo Viscosity')
                        plt.xlim(0, 1)
                        plt.ylim(0, 35)
                        plt.grid(color='grey', linestyle='--', linewidth=0.5)
                        plt.grid(which="minor", linestyle='--', linewidth=0.2)
                        plt.minorticks_on()
                        plt.savefig(join(STARTDIR, Name, f'{Name}_{Run}_{Temp}_GreenKubo2ns.png'))
                        plt.close()

                    DataframeGK[f'Viscosity_{Run}'] = viscosity[:]*1000

                    # Save running integral of the viscosity as a csv file
                    # df = pd.DataFrame({"time(ps)" : Time[:viscosity.shape[0]:each], "viscosity(Pa.s)" : viscosity[::each]})

                except Exception as E:
                    print(E)
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
                ViscGK_UpperSTD = [a + b for a, b in zip(DataframeGKViscList_Average, DataframeGKViscList_AverageSTD)]
                ViscGK_LowerSTD = [a - b for a, b in zip(DataframeGKViscList_Average, DataframeGKViscList_AverageSTD)]
                DataframeEinsteinList_Average = DataframeEinstein['Average'].to_list()
                DataframeEinsteinList_AverageSTD = DataframeEinstein['STD'].to_list()
                DataframeEinsteinList_Average = [float(x) for x in DataframeEinsteinList_Average]
                DataframeEinsteinList_AverageSTD = [float(x) for x in DataframeEinsteinList_AverageSTD]
                Einstein_LowerSTD = [a + b for a, b in zip(DataframeEinsteinList_Average, DataframeEinsteinList_AverageSTD)]
                Einstein_UpperSTD = [a - b for a, b in zip(DataframeEinsteinList_Average, DataframeEinsteinList_AverageSTD)]

                print('Einstein Uncert')
                print(DataframeEinsteinList_AverageSTD[-1])
                print('GK Uncert Viscosity')
                print(DataframeGKViscList_AverageSTD[-1])

                step = list(range(0, len(DataframeGKViscList_Average)))
                step = [x/1000 for x in step]

                # Plot Visc evolution - Green Kubo
                ViscPlt, Vplot = plt.subplots()
                Vplot.set_title(f'GK Viscosity Average - {Name} {Temp}K')
                Vplot.set_ylabel('Viscosity [mPa.S]')
                Vplot.set_xlabel('Time (ns)')
                Vplot.plot(step, DataframeGKViscList_Average)
                Vplot.legend([f'Viscosity Estimate: {round((DataframeGKViscList_Average[-1]), 2)} [mPa.s]']) 
                Vplot.fill_between(step, ViscGK_LowerSTD, ViscGK_UpperSTD, alpha=0.4)
                Vplot.grid(color='grey', linestyle='--', linewidth=0.5)
                Vplot.grid(which="minor", linestyle='--', linewidth=0.2)
                plt.minorticks_on()
                Vplot.set_xlim(0, 1)
                Vplot.set_ylim(0, 35)
                plt.savefig(join(STARTDIR, Name,  f'AvViscPlotGK_{Name}{Temp}K_2ns.png'))
                plt.close()            

                step = list(range(0, len(DataframeEinsteinList_Average)))
                step = [x/1000 for x in step]

                # Plot Visc evolution - Einstein
                ViscPlt, Vplot = plt.subplots()
                Vplot.set_title(f'Einstein Viscosity Average - {Name} {Temp}K')
                Vplot.set_ylabel('Viscosity [mPa.S]')
                Vplot.set_xlabel('Time (ns)')
                Vplot.plot(step, DataframeEinsteinList_Average)
                Vplot.legend([f'Viscosity Estimate: {round((DataframeEinsteinList_Average[-1]), 2)} [mPa.s]']) 
                Vplot.fill_between(step, Einstein_LowerSTD, Einstein_UpperSTD, alpha=0.4)
                Vplot.grid(color='grey', linestyle='--', linewidth=0.5)
                Vplot.grid(which="minor", linestyle='--', linewidth=0.2)
                plt.minorticks_on()
                Vplot.set_xlim(0, 1)
                Vplot.set_ylim(0, 35)
                plt.savefig(join(STARTDIR, Name,  f'EinsteinAvViscPlot_{Name}{Temp}K_2ns.png'))
                plt.close() 

                print(f'Einstein Viscosity: {round((DataframeEinsteinList_Average[-1]), 2)}')
                print(f'GK Viscosity: {round((DataframeGKViscList_Average[-1]), 2)}')


            except Exception as E:
                print(E)
                pass


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