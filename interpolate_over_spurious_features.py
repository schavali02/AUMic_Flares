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
import numpy.polynomial.polynomial as poly
import time

plt.ion()

rc('text', usetex=True)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans']})
rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)

path = '/Users/shashankchavali/projects/Python/ResearchProject/shifted_files_v2'

"""
test_fn = '00095500005_3_shift.txt'

txt = np.loadtxt(path+test_fn)

fig=plt.figure()
plt.plot(txt[:,0],txt[:,1],color='k')
plt.vlines(2800,-1e-13,5e-13,color='m',linestyle=':')
plt.vlines(2616,-1e-13,5e-13,color='m',linestyle=':')
plt.vlines(2373,-1e-13,5e-13,color='m',linestyle=':')

plt.ylim([-1e-13,5e-13])
plt.xlim([1800,3500])
plt.xlabel('Wavelength (\\AA)',fontsize=18)
plt.ylabel('Flux Density (erg cm$^{-2}$ s$^{-1}$ \\AA$^{-1}$)',fontsize=18)
# plt.title(test_fn,fontsize=18)
plt.tight_layout()
"""


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


coords = []


def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d' % (
        ix, iy))

    global coords
    coords.append((ix, iy))

    if len(coords) == 4:
        fig.canvas.mpl_disconnect(cid)

    return coords


def remove_line(wave, flux):
    dummy = fig.ginput(n=1, timeout=-1, show_clicks=True,
                       mouse_add=1, mouse_pop=3, mouse_stop=2)

    print("dummy click done")

    coords = fig.ginput(n=4, timeout=-1, show_clicks=True,
                        mouse_add=1, mouse_pop=3, mouse_stop=2)

    line_region_mask = (wave >= coords[0][0]) & (wave <= coords[1][0])

    fit_region_mask1 = (wave >= coords[2][0]) & (wave <= coords[0][0])

    fit_region_mask2 = (wave >= coords[1][0]) & (wave <= coords[3][0])

    fit_region_mask = fit_region_mask1 + fit_region_mask2

    # , w=1./(error[fit_region_mask]**2))
    coefs = poly.polyfit(wave[fit_region_mask], flux[fit_region_mask], deg=2)
    ffit = poly.polyval(wave, coefs)
    plt.plot(wave[fit_region_mask], ffit[fit_region_mask], color='c')

    flux[line_region_mask] = ffit[line_region_mask]

    plt.plot(wave[fit_region_mask + line_region_mask],
             flux[fit_region_mask + line_region_mask], color='k', alpha=0.4)

    return flux


 # need to add mask for where features were replaced
for i, fn in enumerate(glob.glob(
        '/Users/shashankchavali/projects/Python/ResearchProject/interpolate/*txt')):

    txt = np.loadtxt(fn)

    fig = plt.figure()
    plt.plot(txt[:, 0], txt[:, 1], color='k')
    plt.vlines(2800, -1e-13, 5e-13, color='m', linestyle=':')
    plt.vlines(2616, -1e-13, 5e-13, color='m', linestyle=':')
    plt.vlines(2373, -1e-13, 5e-13, color='m', linestyle=':')

    plt.ylim([-1e-13, 5e-13])
    plt.xlim([1800, 3500])
    plt.xlabel('Wavelength (\\AA)', fontsize=18)
    plt.ylabel(
        'Flux Density (erg cm$^{-2}$ s$^{-1}$ \\AA$^{-1}$)',
        fontsize=18)
    # plt.title(test_fn,fontsize=18)
    plt.tight_layout()
    print(i)
    print(fn)
    print(txt)

    continue_param = input(
        "Need to subtract features from this spectrum? type 'yes', otherwise just hit enter or type anything else: ")

    any_changes = False

    while continue_param == 'yes':
        print("starting while loop")
        flux = remove_line(txt[:, 0], txt[:, 1])
        plt.plot(txt[:, 0], flux, color='r', linestyle='--')

        any_changes = True

        continue_param = input(
            "Need to subtract features from this spectrum? type 'yes', otherwise just hit enter or type anything else: ")
        print("end while loop")
    dummy = fig.ginput(n=1, timeout=-1, show_clicks=True,
                       mouse_add=1, mouse_pop=3, mouse_stop=2)
    print('post dummy')
    looks_good = input("Looks ok?")

    if any_changes:
        np.savetxt('/Users/shashankchavali/projects/Python/ResearchProject/zeroth_order_subtracted_files/'+fn[-23:],
                   np.transpose(np.array([txt[:, 0], flux])))
    else:
        np.savetxt('/Users/shashankchavali/projects/Python/ResearchProject/zeroth_order_subtracted_files/'+fn[-23:],
                   np.transpose(np.array([txt[:, 0], txt[:, 1]])))

    if any_changes:
        print(fn[-23:] + ' changed')

    # time.sleep(5)

    plt.close()
