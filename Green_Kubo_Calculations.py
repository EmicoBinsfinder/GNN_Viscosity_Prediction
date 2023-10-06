"""
Author: Egheosa Ogbomo
Date: 06/10/2023

A script to calculate the viscosity of a fluid from a LAMMPS simulation

Steps:

1. Load off-diagonal stress tensors in
2. Plot evolution of these stress tensors
3. Find autocorrelation of each of these tensors
4.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.tsatools import lagmat
import matplotlib.pyplot as plt
import os
import csv
import ast

working_dir = 'F:/PhD/HIGH_THROUGHPUT_STUDIES/Ashlie_Martini_Study/API529/Viscosity/'
filename = 'Stress_AVGOnespd_API529_T300FP1atm.out'

filepath = os.path.join(working_dir, filename)

with open(f'{filepath}', 'r') as file:
    ReaderObject = csv.reader(file)
    firstrow = next(ReaderObject) # Skipping first line of output which is useless
    ColumnHeaders = next(ReaderObject)[0].split(' ')[1:] # Isolating names of pressure tensors and Timestep column

    print(ColumnHeaders)

    Tensors_Dataframe = pd.DataFrame(columns=ColumnHeaders)

    for row in ReaderObject:
        row = row[0].split(' ')
        row = [float(x) for x in row]
        Tensors_Dataframe.loc[len(Tensors_Dataframe)] = row

ax=plt.subplot()
plt.plot(Tensors_Dataframe['TimeStep'], Tensors_Dataframe['v_myPxz'])
ax.set_ylim(-350, 350)
ax.set_xlim(0, 1.1e6)
plt.show()


"""
Seems to fluctuate around 0 showing weak or no autocorrelation?, or cyclic
"""