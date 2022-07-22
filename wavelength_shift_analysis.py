from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
from matplotlib import gridspec
from PyAstronomy import pyasl
import matplotlib
import astropy.io.fits as pyfits
from astropy.time import Time
from matplotlib import cm
import glob
from lmfit import Model
from astropy.modeling.models import Gaussian1D

plt.ion()

rc('text', usetex=True)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans']})
rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)


def Gaussian(x, x0, am, stddev, c0):  # , c1):

    g = Gaussian1D(amplitude=10.**am, mean=x0, stddev=stddev)
    cont = 10.**c0  # + x*c1

    return g(x) + cont


fmodel = Model(Gaussian)

params = fmodel.make_params(x0=2800, am=-13, stddev=8, c0=-14)  # , c1=0.1)

df = pd.DataFrame(data=None, columns=[
                  'filename', 'center', 'am', 'stddev', 'c0'])

files_special_attention = ['00095500005_3_shiftv3.txt',
                           '00095500005_5_shiftv3.txt', '00095500005_4_shiftv3.txt']
files_special_attention2 = ['00095500005_1_shiftv3.txt']

# 00095500001_1_shift.txt is garbage
print('hello')
for i, fn in enumerate(glob.glob('/Users/shashankchavali/projects/Python/ResearchProject/shifted_files_v3/*txt')):
    fig = plt.figure()
    print(fn[72:])
    txt = np.loadtxt(fn, skiprows=5)

    '''
    if (fn[72:]) in files_special_attention:

        mask = (txt[:, 0] > 2900) & (txt[:, 0] < 3100)
        x0 = 3000

    elif (fn[72:]) in files_special_attention2:

        mask = (txt[:, 0] > 2450) & (txt[:, 0] < 2652)
        x0 = 2568

    else:
	'''
    mask = (txt[:, 0] > 2700) & (txt[:, 0] < 2900)
    x0 = 2800

    params = fmodel.make_params(x0=x0, am=-13, stddev=8, c0=-14)

    plt.plot(txt[:, 0][mask], txt[:, 1][mask], color='k')

    result = fmodel.fit(txt[:, 1][mask], params, x=txt[:, 0][mask])

    plt.plot(txt[:, 0][mask], result.best_fit, 'r--')
    shifted_wave_array = txt[:, 0] - result.params['x0'].value + 2799.116
    np.savetxt('/Users/shashankchavali/projects/Python/ResearchProject/shifted_files_AY/' +
               fn[72:], np.transpose(np.array([shifted_wave_array, txt[:, 1], txt[:, 2]])))

    df.loc[i] = [fn[72:], result.params['x0'].value, result.params['am'].value,
                 result.params['stddev'].value, result.params['c0'].value]
    fig.savefig('/Users/shashankchavali/projects/Python/ResearchProject/wavelength_shift_figures/' +
                fn[72:-12] + '_fitted.png')
    #import pdb; pdb.set_trace()
    plt.close()

print('hello')

df.to_csv(
    '/Users/shashankchavali/projects/Python/ResearchProject/wavelength_shifts.csv')


# these three need special attention
