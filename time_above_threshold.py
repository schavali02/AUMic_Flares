### TIME ABOVE ABIOGENESIS THRESHOLD
import os
import numpy as np
counter48 = 0
counter812 = 0
counter12up = 0
rootdir = '/Users/shashankchavali/projects/Python/ResearchProject/shifted_files_AY'
for path, subdirs, files in os.walk(rootdir):
    for file in files:
        if file[-9:] == 'shift.txt' and file != '.DS_Store' and file != 'average_spectra_maker.py':
            filen = rootdir + '/' + file
            txt = np.loadtxt(filen)
            wave = txt[:,0]
            flux = txt[:,1]
            mask = (wave >= 2500) & (wave <= 3200)
            flux_mean = np.mean(flux[mask])
            abio_threshold_mean = 3.3945000000000002e-15
            ratio = flux_mean / abio_threshold_mean
            print(file)
            print(ratio)
            if (ratio  >= 4) & (ratio <= 8):
                counter48 = counter48 + 1
            elif (ratio  > 8) & (ratio <= 12):
                counter812 = counter812 + 1
            elif (ratio  > 12):
                counter12up = counter12up + 1


            


