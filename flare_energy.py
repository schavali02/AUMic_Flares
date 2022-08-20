### CALCULATING FLARE ENERGIES
import numpy as np
import os

file_array = np.array([])
energy_array = np.array([])
rootdir = '/Users/shashankchavali/projects/Python/ResearchProject/shifted_files_AY'
for path, subdirs, files in os.walk(rootdir):
    for file in files:
        if file[-9:] == 'shift.txt':
            filen = rootdir + '/' + file
            txt = np.loadtxt(filen)
            wave = txt[:,0]
            flux = txt[:,1]
            mask = (wave >= 2500) & (wave <= 3200)
            energy = np.trapz(flux[mask], wave[mask])
            file_array = np.append(file_array, file)
            energy_array = np.append(energy_array, energy)

flare_energy_array = np.vstack((file_array, energy_array)).T
np.savetxt("flare_energy_array.txt", flare_energy_array, fmt="%s")

           