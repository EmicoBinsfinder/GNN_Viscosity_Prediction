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

# Viscosity from Einstein relation
def einstein(timestep):
    Pxxyy = (Pxx - Pyy) / 2
    Pyyzz = (Pyy - Pzz) / 2

    timestep = timestep * 10**(-12)
    Pxy_int = integrate.cumtrapz(y=Pxy, dx=timestep, initial=0)
    Pxz_int = integrate.cumtrapz(y=Pxz, dx=timestep, initial=0)
    Pyz_int = integrate.cumtrapz(y=Pyz, dx=timestep, initial=0)

    integral = (Pxy_int**2 + Pxz_int**2 + Pyz_int**2) / 3
    viscosity = integral[1:] * (volume * 10**(-30) / (2 * kBT * Time[1:] * 10**(-12)))

    return viscosity

# Define a function to pad the viscosity array
def pad_viscosity(viscosity, min_length, target_length):
    if len(viscosity) >= min_length and len(viscosity) <= target_length:
        # Calculate how many elements to add
        elements_to_add = target_length - len(viscosity)
        # Pad the viscosity array with NaNs
        viscosity = np.pad(viscosity, (0, elements_to_add), mode='constant', constant_values=np.nan)
    return viscosity

# Bootstrapping function
def bootstrap(data, n_iterations):
    bootstrap_means = []
    for _ in range(n_iterations):
        resampled_data = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(resampled_data))
    return np.array(bootstrap_means)

# Define Green-Kubo function
def green_kubo(timestep):
    Pxy_acf = acf(Pxy)
    Pxz_acf = acf(Pxz)
    Pyz_acf = acf(Pyz)
    avg_acf = (Pxy_acf + Pxz_acf + Pyz_acf) / 3

    timestep = timestep * 10**(-12)
    integral = integrate.cumtrapz(y=avg_acf, dx=timestep, initial=0)
    viscosity = integral * (volume * 10**(-30) / kBT)

    return avg_acf, viscosity

# Main processing loop
chdir('/rds/general/ephemeral/user/eeo21/ephemeral/Benchmarking/COMPASS')
STARTDIR = getcwd()
Names = [x for x in listdir(getcwd()) if os.path.isdir(x)]
Temps = [313, 373]

# Initialize a list to store the final viscosity values for each material and temperature
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
                datafile = f'Stress_AVGOnespd_{Name}_T{Temp}KP1atm.out'
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

                kBT = Boltzmann * temperature

                Pxx, Pyy, Pzz, Pxy, Pxz, Pyz = [], [], [], [], [], []

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

                Pxx = np.array(Pxx)
                Pyy = np.array(Pyy)
                Pzz = np.array(Pzz)
                Pxy = np.array(Pxy)
                Pxz = np.array(Pxz)
                Pyz = np.array(Pyz)

                end_step = steps * timestep
                Time = np.linspace(0, end_step, num=steps, endpoint=False)

                viscosity = einstein(timestep=timestep)
                viscE = round((viscosity[-1] * 1000), 2)

                if plot:
                    plt.figure()
                    plt.plot(Time[:viscosity.shape[0]], viscosity[:]*1000, label='Viscosity (Average)', linewidth=1)  # thinner line
                    plt.xlabel('Time (ps)')
                    plt.ylabel('Einstein Viscosity (mPa.s)')
                    plt.legend([f'Viscosity Estimate: {viscE} [mPa.s]'])
                    plt.title(f'{Name}_{Run}_{Temp}_Einstein')
                    plt.xlim(0, 1)
                    plt.ylim(0, 50)  # y-axis range between 0 and 50
                    plt.savefig(join(STARTDIR, Name, f'{Name}_{Run}_{Temp}_Einstein.png'))
                    plt.close()

                # Add to DataframeEinstein only if the viscosity array length is at least 1000
                if len(viscosity) > 1000:
                    viscosity_padded = pad_viscosity(viscosity, min_length=1000, target_length=1500)
                    DataframeEinstein[f'Viscosity_{Run}'] = viscosity_padded * 1000

                # Perform bootstrapping for Einstein viscosity
                bootstrap_results_einstein = bootstrap(viscosity, n_iterations=20)
                mean_einstein = np.mean(bootstrap_results_einstein)
                std_einstein = np.std(bootstrap_results_einstein)
                ci_lower_einstein = np.percentile(bootstrap_results_einstein, 2.5)
                ci_upper_einstein = np.percentile(bootstrap_results_einstein, 97.5)

                # Similarly for Green-Kubo viscosity
                avg_acf, viscosity_gk = green_kubo(timestep)
                bootstrap_results_gk = bootstrap(viscosity_gk, n_iterations=20)
                mean_gk = np.mean(bootstrap_results_gk)
                std_gk = np.std(bootstrap_results_gk)
                ci_lower_gk = np.percentile(bootstrap_results_gk, 2.5)
                ci_upper_gk = np.percentile(bootstrap_results_gk, 97.5)

                # Add to DataframeGK only if the viscosity array length is at least 500
                if len(viscosity_gk) >= 500:
                    viscosity_padded = pad_viscosity(viscosity_gk, min_length=500, target_length=750)
                    DataframeGK[f'Viscosity_{Run}'] = viscosity_padded * 1000

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

            # Save the viscosity data to CSV files
            DataframeEinstein['Bootstrap_Mean'] = mean_einstein
            DataframeEinstein['Bootstrap_STD'] = std_einstein
            DataframeEinstein.to_csv(join(STARTDIR, Name, f'{Name}_{Temp}K_Einstein_Viscosity.csv'), index=False)
            DataframeGK['Bootstrap_Mean'] = mean_gk
            DataframeGK['Bootstrap_STD'] = std_gk
            DataframeGK.to_csv(join(STARTDIR, Name, f'{Name}_{Temp}K_GreenKubo_Viscosity.csv'), index=False)

            # Extract the final viscosity values
            ViscosityAvList = DataframeGK['Average'].tolist()
            ViscosityAv = ViscosityAvList[-1]
            ViscosityAvListEinstein = DataframeEinstein['Average'].tolist()
            ViscosityAvEinstein = ViscosityAvListEinstein[-1]

            # Add the final viscosities to the master list
            final_viscosities.append({
                "Material": Name,
                "Temperature (K)": Temp,
                "Einstein Viscosity (mPa.s)": ViscosityAvEinstein,
                "Green-Kubo Viscosity (mPa.s)": ViscosityAv,
                "Einstein Viscosity Bootstrapped Mean (mPa.s)": mean_einstein,  # Bootstrapped mean for Einstein viscosity
                "Einstein Viscosity Bootstrapped Std (mPa.s)": std_einstein,    # Bootstrapped std for Einstein viscosity
                "Green-Kubo Viscosity Bootstrapped Mean (mPa.s)": mean_gk,      # Bootstrapped mean for Green-Kubo viscosity
                "Green-Kubo Viscosity Bootstrapped Std (mPa.s)": std_gk         # Bootstrapped std for Green-Kubo viscosity
            })
            # Plot for Einstein Viscosity with uncertainty and confidence intervals
            plt.figure()
            plt.plot(Time[:len(viscosity)], viscosity[:]*1000, label=f'Viscosity (Mean: {mean_einstein:.2f} mPa.s, Std: {std_einstein:.2f} mPa.s)', linewidth=1)  # thinner line
            plt.fill_between(Time[:len(viscosity)], ci_lower_einstein * 1000, ci_upper_einstein * 1000, color='blue', alpha=0.2, label='95% Confidence Interval')
            plt.xlabel('Time (ps)')
            plt.ylabel('Einstein Viscosity (mPa.s)')
            plt.title(f'{Name}_{Run}_{Temp}_Einstein with Uncertainty and Confidence Intervals')
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 50)  # y-axis range between 0 and 50
            plt.savefig(join(STARTDIR, Name, f'{Name}_{Run}_{Temp}_Einstein_with_uncertainty.png'))
            plt.close()

            # Plot for Green-Kubo Viscosity with uncertainty and confidence intervals
            plt.figure()
            plt.plot(Time[:len(viscosity_gk)], viscosity_gk * 1000, label=f'Viscosity (Mean: {mean_gk:.2f} mPa.s, Std: {std_gk:.2f} mPa.s)', linewidth=1)  # thinner line
            plt.fill_between(Time[:len(viscosity_gk)], ci_lower_gk * 1000, ci_upper_gk * 1000, color='red', alpha=0.2, label='95% Confidence Interval')
            plt.xlabel('Time (ps)')
            plt.ylabel('Green-Kubo Viscosity (mPa.s)')
            plt.title(f'{Name}_{Run}_{Temp}_Green-Kubo with Uncertainty and Confidence Intervals')
            plt.legend()
            plt.grid(True)
            plt.ylim(0, 50)  # y-axis range between 0 and 50
            plt.savefig(join(STARTDIR, Name, f'{Name}_{Run}_{Temp}_Green_Kubo_with_uncertainty.png'))
            plt.close()

        except Exception as E:
            print(E)
            traceback.print_exc()
            pass

# After processing all materials and temperatures, save the final viscosities to a master CSV
master_df = pd.DataFrame(final_viscosities)
master_df.to_csv(join(STARTDIR, 'Final_Viscosities.csv'), index=False)
