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
import traceback
import os

# Define ACF using FFT
def acf(data):
    steps = data.shape[0]
    lag = steps // 2

    size = 2 ** np.ceil(np.log2(2 * steps - 1)).astype('int')
    FFT = np.fft.fft(data, size)
    PWR = FFT.conjugate() * FFT
    COR = np.fft.ifft(PWR)[:steps].real
    autocorrelation = COR / np.arange(steps, 0, -1)

    return autocorrelation[:lag]

# def acf(data):
#     """Computes the autocorrelation function using FFT."""
#     n = len(data)
#     data -= np.mean(data)
#     result = np.correlate(data, data, mode='full')[-n:]
#     result /= result[0]  # Normalize
#     return result

# ThermalConductivity from Einstein relation
def einstein(timestep, Jx, Jy, Jz):

    timestep = timestep * 10**(-15)
    Jx_int = integrate.cumtrapz(y=Jx, dx=timestep, initial=0)
    Jy_int = integrate.cumtrapz(y=Jy, dx=timestep, initial=0)
    Jz_int = integrate.cumtrapz(y=Jz, dx=timestep, initial=0)

    integral = (Jx_int**2 + Jy_int**2 + Jz_int**2) / 3
    ThermalConductivity = integral[1:] * (volume / (2 * kBT * temperature**2 * Time[1:]*10**(-15)))
    # ThermalConductivity = integral[1:] * (volume * 10**(-30) / (2 * kBT * Time[1:] * 10**(-12)))

    return ThermalConductivity

# Define a function to pad the ThermalConductivity array
def pad_ThermalConductivity(ThermalConductivity, min_length, target_length):
    if len(ThermalConductivity) >= min_length and len(ThermalConductivity) <= target_length:
        # Calculate how many elements to add
        elements_to_add = target_length - len(ThermalConductivity)
        # Pad the ThermalConductivity array with NaNs
        ThermalConductivity = np.pad(ThermalConductivity, (0, elements_to_add), mode='constant', constant_values=np.nan)
    return ThermalConductivity

# Bootstrapping function
def bootstrap(data, n_iterations):
    bootstrap_means = []
    for _ in range(n_iterations):
        resampled_data = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(resampled_data))
    return np.array(bootstrap_means)

# Define Green-Kubo function
def green_kubo(timestep):
    Jx_acf = acf(Jx)
    Jy_acf = acf(Jy)
    Jz_acf = acf(Jz)
    avg_acf = (Jx_acf + Jy_acf + Jz_acf) / 3

    timestep = timestep * 10**(-15)
    integral = integrate.cumtrapz(y=avg_acf, dx=timestep, initial=0)
    # ThermalConductivity = integral * (volume * 10**(-30) / kBT)
    ThermalConductivity = integral * (3 * volume / (kBT * temperature**2)) * 10**-1
    

    return avg_acf, ThermalConductivity

# Main processing loop
chdir('/rds/general/ephemeral/user/eeo21/ephemeral/CyclicMoleculeBenchmarkingFeb18/COMPASS')
STARTDIR = getcwd()
Names = [x for x in listdir(getcwd()) if os.path.isdir(x)]
Temps = [313, 373]

# Initialize a list to store the final ThermalConductivity values for each material and temperature
final_viscosities = []

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

                df = pd.read_csv(f'Stress_AVGOnespd_{Name}_T{Temp}KP1atm.out')

                unit = 'atm'
                datafile = f'HeatFlux_AVGOnespd_{Name}_T{Temp}KP1atm.out'
                diag = False
                steps = len(df) - 1
                timestep = 1
                temperature = Temp

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

                each = 2
                plot = True

                if unit == 'Pa':
                    conv_ratio = 1
                elif unit == 'atm':
                    conv_ratio = 101325
                elif unit == 'bar':
                    conv_ratio = 100000

                kBT = Boltzmann 

                Pxx, Pyy, Pzz, Jx, Jy, Jz = [], [], [], [], [], []

                with open(datafile, "r") as file:
                    next(file)
                    next(file)

                    for _ in range(steps):
                        line = file.readline()
                        step = list(map(float, line.split()))
                        Jx.append(step[1])
                        Jy.append(step[2])
                        Jz.append(step[3])

                Jx = np.array(Jx)
                Jy = np.array(Jy)
                Jz = np.array(Jz)

                end_step = steps * timestep
                Time = np.linspace(0, end_step, num=steps, endpoint=False)

                ThermalConductivity = einstein(timestep, Jx, Jy, Jz)
                viscE = round((ThermalConductivity[-1]), 2)

                if plot:
                    plt.figure()
                    plt.plot(Time[:ThermalConductivity.shape[0]], ThermalConductivity[:], label='ThermalConductivity (Average)', linewidth=1)  # thinner line
                    plt.xlabel('Time (ps)')
                    plt.ylabel('Einstein ThermalConductivity W/m.K')
                    plt.legend([f'ThermalConductivity Estimate: {viscE} [W/m.K]'])
                    plt.title(f'{Name}_{Run}_{Temp}_Einstein')
                    plt.xlim(0, 1)
                    # plt.ylim(0, 50)  # y-axis range between 0 and 50
                    plt.savefig(join(STARTDIR, Name, f'{Name}_{Run}_{Temp}_Einstein.png'))
                    plt.close()

                # Add to DataframeEinstein only if the ThermalConductivity array length is at least 1000
                if len(ThermalConductivity) > 1000:
                    ThermalConductivity_padded = pad_ThermalConductivity(ThermalConductivity, min_length=1000, target_length=1500)
                    DataframeEinstein[f'ThermalConductivity_{Run}'] = ThermalConductivity_padded * 1000

                # Perform bootstrapping for Einstein ThermalConductivity
                bootstrap_results_einstein = bootstrap(ThermalConductivity, n_iterations=20)
                mean_einstein = np.mean(bootstrap_results_einstein)
                std_einstein = np.std(bootstrap_results_einstein)
                ci_lower_einstein = np.percentile(bootstrap_results_einstein, 2.5)
                ci_upper_einstein = np.percentile(bootstrap_results_einstein, 97.5)

                # Similarly for Green-Kubo ThermalConductivity
                avg_acf, ThermalConductivity_gk = green_kubo(timestep)
                bootstrap_results_gk = bootstrap(ThermalConductivity_gk, n_iterations=20)
                mean_gk = np.mean(bootstrap_results_gk)
                std_gk = np.std(bootstrap_results_gk)
                ci_lower_gk = np.percentile(bootstrap_results_gk, 2.5)
                ci_upper_gk = np.percentile(bootstrap_results_gk, 97.5)

                # Add to DataframeGK only if the ThermalConductivity array length is at least 500
                if len(ThermalConductivity_gk) >= 500:
                    ThermalConductivity_padded = pad_ThermalConductivity(ThermalConductivity_gk, min_length=500, target_length=750)
                    DataframeGK[f'ThermalConductivity_{Run}'] = ThermalConductivity_padded * 1000

            except Exception as E:
                print(E)
                traceback.print_exc()
                pass

        try:
            DataframeEinstein = DataframeEinstein.dropna()
            DataframeGK = DataframeGK.dropna()

            DataframeGK['Average'] = DataframeGK.mean(axis=1)
            DataframeGK['STD'] = DataframeGK.std(axis=1)
            DataframeEinstein['Average'] = DataframeEinstein.mean(axis=1)
            DataframeEinstein['STD'] = DataframeEinstein.std(axis=1)

            # Save the ThermalConductivity data to CSV files
            DataframeEinstein['Bootstrap_Mean'] = mean_einstein
            DataframeEinstein['Bootstrap_STD'] = std_einstein
            DataframeEinstein.to_csv(join(STARTDIR, Name, f'{Name}_{Temp}K_Einstein_ThermalConductivity.csv'), index=False)
            DataframeGK['Bootstrap_Mean'] = mean_gk
            DataframeGK['Bootstrap_STD'] = std_gk
            DataframeGK.to_csv(join(STARTDIR, Name, f'{Name}_{Temp}K_GreenKubo_ThermalConductivity.csv'), index=False)

            # Extract the final ThermalConductivity values
            ThermalConductivityAvList = DataframeGK['Average'].tolist()
            ThermalConductivityAv = ThermalConductivityAvList[-1]
            ThermalConductivityAvListEinstein = DataframeEinstein['Average'].tolist()
            ThermalConductivityAvEinstein = ThermalConductivityAvListEinstein[-1]

            # Add the final viscosities to the master list
            final_viscosities.append({
                "Material": Name,
                "Temperature (K)": Temp,
                "Einstein ThermalConductivity (W/m.K)": ThermalConductivityAvEinstein,
                "Green-Kubo ThermalConductivity (W/m.K)": ThermalConductivityAv,
                "Einstein ThermalConductivity Bootstrapped Mean (W/m.K)": mean_einstein,  # Bootstrapped mean for Einstein ThermalConductivity
                "Einstein ThermalConductivity Bootstrapped Std (W/m.K)": std_einstein,    # Bootstrapped std for Einstein ThermalConductivity
                "Green-Kubo ThermalConductivity Bootstrapped Mean (W/m.K)": mean_gk,      # Bootstrapped mean for Green-Kubo ThermalConductivity
                "Green-Kubo ThermalConductivity Bootstrapped Std (W/m.K)": std_gk         # Bootstrapped std for Green-Kubo ThermalConductivity
            })
            # Plot for Einstein ThermalConductivity with uncertainty and confidence intervals
            plt.figure()
            plt.plot(Time[:len(ThermalConductivity)], ThermalConductivity[:]*1000, label=f'ThermalConductivity (Mean: {mean_einstein:.2f} W/m.K, Std: {std_einstein:.2f} W/m.K)', linewidth=1)  # thinner line
            plt.fill_between(Time[:len(ThermalConductivity)], ci_lower_einstein * 1000, ci_upper_einstein * 1000, color='blue', alpha=0.2, label='95% Confidence Interval')
            plt.xlabel('Time (ps)')
            plt.ylabel('Einstein ThermalConductivity (W/m.K)')
            plt.title(f'{Name}_{Run}_{Temp}_Einstein with Uncertainty and Confidence Intervals')
            plt.legend()
            plt.grid(True)
            # plt.ylim(0, 50)  # y-axis range between 0 and 50
            plt.savefig(join(STARTDIR, Name, f'{Name}_{Run}_{Temp}_Einstein_with_uncertainty.png'))
            plt.close()

            # Plot for Green-Kubo ThermalConductivity with uncertainty and confidence intervals
            plt.figure()
            plt.plot(Time[:len(ThermalConductivity_gk)], ThermalConductivity_gk, label=f'ThermalConductivity (Mean: {mean_gk:.2f} W/m.K, Std: {std_gk:.2f} W/m.K)', linewidth=1)  # thinner line
            plt.fill_between(Time[:len(ThermalConductivity_gk)], ci_lower_gk, ci_upper_gk, color='red', alpha=0.2, label='95% Confidence Interval')
            plt.xlabel('Time (ps)')
            plt.ylabel('Green-Kubo ThermalConductivity (W/m.K)')
            plt.title(f'{Name}_{Run}_{Temp}_Green-Kubo with Uncertainty and Confidence Intervals')
            plt.legend()
            plt.grid(True)
            # plt.ylim(0, 50)  # y-axis range between 0 and 50
            plt.savefig(join(STARTDIR, Name, f'{Name}_{Run}_{Temp}_Green_Kubo_with_uncertainty.png'))
            plt.close()

        except Exception as E:
            print(E)
            traceback.print_exc()
            pass

# After processing all materials and temperatures, save the final viscosities to a master CSV
master_df = pd.DataFrame(final_viscosities)
master_df.to_csv(join(STARTDIR, 'Final_ThermalConductivity.csv'), index=False)
