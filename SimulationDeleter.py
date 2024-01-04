"""
Script to automatically delete simulations in an array job

"""

import subprocess
import os

def runcmd(cmd, verbose = False, *args, **kwargs):
    #bascially allows python to run a bash command, and the code makes sure 
    #the error of the subproceess is communicated if it fails
    process = subprocess.run(
        cmd,
        text=True,
        shell=True)
    
runcmd('qstat > sims.txt')

sims = []
CWD = os.getcwd()

with open('sims.txt', 'r') as file:
    next(file)
    next(file)
    filelist = file.readlines()
    for x in filelist:
        sim = x.split(' ')[0]
        if '[]' in sim:
            sims.append(sim)

for sim in sims:
    runcmd(f'qdel {sim}')