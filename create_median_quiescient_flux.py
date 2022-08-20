### CREATING MEDIAN QUIESCIENT FLUX
import csv
import numpy as np

file = '/Users/shashankchavali/projects/Python/ResearchProject/quiescent_master_array.txt'
txt = np.loadtxt(file)
#median = np.median(txt[0,1:])
#print(median)
#print(txt[0:])

wave_array = np.array([])
median_array = np.array([])
for x in txt:
    wave_array = np.append(wave_array, x[0])
    median = np.median(x[1:])
    median_array = np.append(median_array, median)

file_array = np.vstack((wave_array, median_array)).T
print(file_array)
np.savetxt("median_quiescent_flux.txt", file_array)



