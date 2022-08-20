### LOADING MASTER_ARRAY WITH WAVELENGTH VALUES
### FILES MUST BE RUN IN THIS ORDER: load_master_array.py, create_master_array.py, create_median_quiescient_flux.py
import numpy as np

master_array = np.array([])
file = '/Users/shashankchavali/projects/Python/ResearchProject/shifted_files_AY/00095500002_1_shift_interp.txt'
txt = np.loadtxt(file)
wave = txt[:,0]
wave = np.vstack(wave)
master_array = np.hstack((wave))