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
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import glob
import os


from matplotlib import rc

plt.ion()

rc('text', usetex=True)
rc('font', **{'family': 'sans-serif', 'sans-serif': ['DejaVu Sans']})
rc('text.latex', preamble=r'\usepackage[cm]{sfmath}')
rc('xtick', labelsize=8)
rc('ytick', labelsize=8)

fname = "/Users/shashankchavali/Downloads/Swift_UVOT.UVM2.dat"
x = pd.read_table(fname, header=1, delim_whitespace=True,
                  names=['wavelength', 'effective area'])
wave_ea = np.array(x['wavelength'])
ea = np.array(x['effective area'])

photometry = pd.read_csv(
    '/Users/shashankchavali/projects/Python/ResearchProject/AUMic_UVgrism_photometry_timesorted.csv')
synthetic_UVM2 = photometry.loc[photometry['Type'] == 'UV grism']
real_UVM2 = photometry.loc[photometry['Type'] == 'UVM2']


# synthetic_UVM2 = pd.read_csv('/Users/aayoungb/Data/Swift/AUMic_grism_flares/Shashank_code_UVM2_synthetic_photometry/AUMic_UVM2_synthetic_photometry.txt')
# real_UVM2 = pd.read_csv('/Users/aayoungb/Data/Swift/AUMic_grism_flares/AUMic_UVM2_photometry.txt')

# synthetic_UVM2 = synthetic_UVM2.sort_values(by=['# time(jd)'])
# real_UVM2 = real_UVM2.sort_values(by=['# time(jd)'])


# Full obs set!
plt.figure()
plt.errorbar((synthetic_UVM2['Time (MJD)'] - np.min(real_UVM2['Time (MJD)']))*24.*3600., synthetic_UVM2['Flux (counts/s)'],
             xerr=synthetic_UVM2['exptime (s)']/2., yerr=synthetic_UVM2['Error (counts/s)'], fmt='o', color='r', label='simulated flux')
plt.errorbar((real_UVM2['Time (MJD)'] - np.min(real_UVM2['Time (MJD)']))*24.*3600., real_UVM2['Flux (counts/s)'],
             xerr=real_UVM2['exptime (s)']/2., yerr=real_UVM2['Error (counts/s)'], fmt='s', color='k', label='UVM2 flux')

plt.legend()
plt.xlabel('Time from ' +
           str(np.min(real_UVM2['Time (MJD)'])) + 'JD in seconds', fontsize=8)
plt.ylabel('Flux (counts/s)', fontsize=8)
plt.title('Simulated Photometric Flux and UVM2 Light Curve', fontsize=8)
# plt.tight_layout()


# Let's automate these obs set plots

obs_set_numbers = np.arange(42) + 1

obs_set_start_jd = np.array([2458944.95335648,  # 1
                             2.458945422482638620e+06,  # 2
                             2.458958503159722313e+06,  # 3
                             2.458965537910879590e+06,  # 4
                             2.458973166215277743e+06,  # 5
                             2.458979544247685000e+06,  # 6
                             2.458986845063657500e+06,  # 7
                             2.458993686574073974e+06,  # 8
                             2.459001389716435224e+06,  # 9
                             2.459007561018518638e+06,  # 10
                             2.459015395231481642e+06,  # 11
                             2.459022033634259365e+06,  # 12
                             2.459029200555555522e+06,  # 13
                             2.459035716660879552e+06,  # 14
                             2.459043421030092519e+06,  # 15
                             2.459050451226851903e+06,  # 16
                             2.459057089525463060e+06,  # 17
                             2.459062133697916754e+06,  # 18
                             2.459070830185185187e+06,  # 19
                             2.459077811568287201e+06,  # 20
                             2.459085180960648227e+06,  # 21
                             2.459094069479166530e+06,  # 22
                             2.459099069554397836e+06,  # 23
                             2.459103246990740765e+06,  # 24
                             2.459105830324074253e+06,  # 25
                             2.459112870520833414e+06,  # 26
                             2.459121027997685131e+06,  # 27
                             2.459126742256944533e+06,  # 28
                             2.459137891221065074e+06,  # 29
                             2.459140941435185261e+06,  # 30
                             2.459148451076388825e+06,  # 31
                             2.459155159959490411e+06,  # 32
                             2.459161800891203806e+06,  # 33
                             2.459168900243055541e+06,  # 34
                             2.459175798883101903e+06,  # 35
                             2.459182706990740728e+06,  # 36
                             2.459189810422453564e+06,  # 37
                             2.45928552568287e+06,  # 38
                             2.4592925621875e+06,  # 39
                             2.45930039797453e+06,  # 40
                             2.45930185760995e+06,  # 41
                             2.45930736930555e+06])  # 42


obs_set_end_jd = np.array([2.458944963541666511e+06,  # 1
                           2.458945425798611250e+06,  # 2
                           2.458958512881944422e+06,  # 3
                           2.458965546388888732e+06,  # 4
                           2458973.17675925,  # 5
                           2.458979555254629813e+06,  # 6
                           2.458986855254629627e+06,  # 7
                           2.458993695046296343e+06,  # 8
                           2.459001399687499739e+06,  # 9
                           2.459007569421296474e+06,  # 10
                           2.459015405243055429e+06,  # 11
                           2.459022043460648041e+06,  # 12
                           2.459029209027777892e+06,  # 13
                           2.459035726770833135e+06,  # 14
                           2.459043429513888899e+06,  # 15
                           2.459050459722222295e+06,  # 16
                           2.459057098032407463e+06,  # 17
                           2.459062144953703508e+06,  # 18
                           2.459070838680555578e+06,  # 19
                           2.459077822627314832e+06,  # 20
                           2.459085191342592705e+06,  # 21
                           2.459094077025462873e+06,  # 22
                           2.459099069554397836e+06,  # 23
                           2.459103255480323918e+06,  # 24
                           2.459105838784722146e+06,  # 25
                           2.459112880231481511e+06,  # 26
                           2.459121036481481511e+06,  # 27
                           2.459126753136574291e+06,  # 28
                           2.459137899710648227e+06,  # 29
                           2.459140952442129608e+06,  # 30
                           2.459148459560185205e+06,  # 31
                           2.459155171215277631e+06,  # 32
                           2.459161809363425709e+06,  # 33
                           2.459168911527778022e+06,  # 34
                           2.459175807361111045e+06,  # 35
                           2.459182716782407369e+06,  # 36
                           2.459189820532407612e+06,  # 37
                           2.45928553417245e+06,  # 38
                           2459292.57185185,  # 39
                           2459300.40802083,  # 40
                           2459301.8677662,  # 41
                           2459307.37778935])  # 42


buffer_begin = -60  # seconds - must be negative
buffer_end = 600  # seconds

for i in range(len(obs_set_numbers)):

    t_begin = obs_set_start_jd[i]
    t_end = obs_set_end_jd[i]

    mask_synthetic = (synthetic_UVM2['Time (MJD)'] >= (t_begin + buffer_begin/(
        24.*3600.))) & (synthetic_UVM2['Time (MJD)'] <= (t_end + buffer_end/(24.*3600.)))
    mask_real = (real_UVM2['Time (MJD)'] >= (t_begin + buffer_begin/(24.*3600.))
                 ) & (real_UVM2['Time (MJD)'] <= (t_end + buffer_end/(24.*3600.)))

    cmap = cm.get_cmap('viridis', mask_synthetic.sum())

    t_begin_isot = Time(t_begin, format='jd').isot

    fig = plt.figure(figsize=(20, 9))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax3 = inset_axes(ax1, width='30%', height=1.1, loc=2, bbox_to_anchor=(
        1.23, -0.06, 1.1, 1), bbox_transform=ax1.transAxes, borderpad=0)

    sorted_indices = np.argsort(synthetic_UVM2['Time (MJD)'][mask_synthetic])

    for c, j in enumerate(sorted_indices.index):
        ax1.errorbar((synthetic_UVM2['Time (MJD)'][mask_synthetic][j] - t_begin)*24.*3600., synthetic_UVM2['Flux (counts/s)'][mask_synthetic][j],
                     xerr=synthetic_UVM2['exptime (s)'][mask_synthetic][j]/2., yerr=synthetic_UVM2['Error (counts/s)'][mask_synthetic][j], fmt='o', color=cmap(c))  # ,label='simulated flux')
    ax1.errorbar((real_UVM2['Time (MJD)'][mask_real] - t_begin)*24.*3600., real_UVM2['Flux (counts/s)'][mask_real],
                 xerr=real_UVM2['exptime (s)'][mask_real]/2., yerr=real_UVM2['Error (counts/s)'][mask_real], fmt='s', color='k', label='UVM2 flux')

    ax1.legend()
    ax1.set_xlabel('Time (s) since ' + str(t_begin_isot), fontsize=8)
    ax1.set_ylabel('UVM2 Flux (counts/s)', fontsize=8)
    # ax1.set_title('Obs Set ' + str(obs_set_numbers[i]), fontsize=8)
    ax1.axhline(y=11.423399925231934, color='g',
                linestyle='dashed', label="Median Photometric Flux")
    ax1.legend()

    grism_filenames = synthetic_UVM2['filename'][mask_synthetic]
    file = '/Users/shashankchavali/projects/Python/ResearchProject/AUMic_Grism_integrated_flux_2200_to_3200.csv'
    max_flux_mean = 0
    for j, fn in enumerate(grism_filenames):

        txt = np.loadtxt(
            '/Users/shashankchavali/projects/Python/ResearchProject/shifted_files_AY/' + fn)

        ax2.plot(txt[:, 0], txt[:, 1], color=cmap(j), linewidth=0.7, alpha=0.7)
        ea_interp = np.interp(txt[:, 0], wave_ea, ea)
        wave = txt[:, 0]
        flux = txt[:, 1]
        mask = (wave >= 2000) & (wave <= 2800)
        flux_mean = np.mean(flux[mask])
        ax3.axhline(y=flux_mean, color=cmap(j))
        if flux_mean > max_flux_mean:
            max_flux_mean = flux_mean

    #ax2.set_ylim([-3e-14, 8e-13])
        ax2.set_ylim([-3e-14, 2e-13])
    ax2.set_xlim([1750, 4000])
    ax2.set_xlabel('Wavelength (\AA)', fontsize=8)
    ax2.set_ylabel(
        'Flux Density (erg cm$^{-2}$ s$^{-1}$ \AA$^{-1}$)', fontsize=8)
    # ax2.add_patch(Rectangle((2000, 1.599e-15), 800, 3.591e-15,fill = True, color = 'r', alpha = 0.5, zorder = 100, figure = fig))
    median_quiescient_flux_file = '/Users/shashankchavali/projects/Python/ResearchProject/median_quiescent_flux.txt'
    median_file = np.loadtxt(median_quiescient_flux_file)
    ax2.plot(median_file[:, 0], median_file[:, 1], color='black',
             linewidth=0.7, alpha=0.7, label='median quiescent flux')
    mask = (median_file[:, 0] >= 2000) & (median_file[:, 0] <= 2800)
    mean_median_flux = np.mean(median_file[:, 1][mask])

    ax3.add_patch(Rectangle((2000, 1.599e-15), 800, 3.591e-15,
                  fill=True, color='r', alpha=0.5, zorder=100))
    ax3.set_xlim(2000, 2800)
    upper_limit = 1e-9
    ax3.set_ylim([1e-16, 1e-11])
    ax3.yaxis.tick_right()
    ax3.tick_params(axis='both', which='major', labelsize=4)
    ax3.axhline(y=mean_median_flux, color='black')
    ax3.set_yscale('log')

    ax2.set_xlabel('Wavelength (\AA)', fontsize=8)
    ax2.set_ylabel(
        'Flux Density (erg cm$^{-2}$ s$^{-1}$ \AA$^{-1}$)', fontsize=8)

    if obs_set_numbers[i] == 1:
        ax1.set_ylim([7, 13])

    ax1.minorticks_on()
    ax2.minorticks_on()

    # fig.tight_layout()

    fig.set_size_inches(7, 4)
    fig.subplots_adjust(bottom=0.15, right=0.95, left=0.1, top=0.8)

    fig.savefig('/Users/shashankchavali/projects/Python/ResearchProject/obs_set_figures_v2/obs_set_' +
                str(obs_set_numbers[i]) + '.png')

    plt.close()
