### CREATING MASTER_ARRAY
import numpy as np
import os


rootdir = '/Users/shashankchavali/projects/Python/ResearchProject/shifted_files_AY'
#master_array = np.array([])
for path, subdirs, files in os.walk(rootdir):
    for file in files:
        if file[-16:-4] == 'shift_interp':
            filen = rootdir + "/" +file
            txt = np.loadtxt(filen)
            flux = txt[:,1]
            master_array = np.column_stack((master_array, flux))
            
np.savetxt('quiescent_master_array.txt', master_array)