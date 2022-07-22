import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams, ticker, gridspec
import glob
import astropy.io.fits as pyfits
from astropy.time import Time
import pandas as pd
import os

plt.ion()

rc('font',**{'family':'sans-serif'})
rc('text', usetex=True)

label_size = 16
rcParams['xtick.labelsize'] = label_size 
rcParams['ytick.labelsize'] = label_size




obsids = np.sort(glob.glob("/Users/shashankchavali/Downloads/AUMic_Swift_data/000955000*"))
subdirectory_path = "/uvot/image/"

save_path = "/Users/shashankchavali/Downloads/AUMic_Swift_data/"

time_array = np.array([])
isot_array = np.array([])
exposure_time_array = np.array([])
filename_array = np.array([],dtype=str)
obsid_array = np.array([],dtype=str)
ra_array = np.array([])
dec_array = np.array([])
roll_angle_array = np.array([])


for i, obsid in enumerate(obsids):

    print(i, obsid.split('/')[-1])
    filename = obsid + subdirectory_path + 'sw' + obsid.split('/')[-1] + 'um2_sk.img.gz'
    if not os.path.exists(filename):
        filename = filename.replace('.gz','')
    

    if os.path.exists(filename):
        hdu = pyfits.open(filename)

        n_hdu_extensions = len(hdu)

        for j in range(n_hdu_extensions-1):


            header = hdu[j+1].header

            exptime = hdu[j+1].header['TELAPSE']
            start_time = Time(hdu[j+1].header["DATE-OBS"], format = "isot", scale = "utc")
            end_time = Time(hdu[j+1].header["DATE-END"], format = "isot", scale = "utc")
            t_mid_jd = np.mean([start_time.jd, end_time.jd])
            mean_time = Time(t_mid_jd, format = "jd")
            RA = hdu[j+1].header['RA_PNT']
            Dec = hdu[j+1].header['DEC_PNT']
            roll_angle = hdu[j+1].header['PA_PNT']


            time_array = np.append(time_array, mean_time.jd)
            isot_array = np.append(isot_array, mean_time.isot)
            exposure_time_array = np.append(exposure_time_array, exptime)
            filename_array = np.append(filename_array, 'sw' + obsid.split('/')[-1] + 'ugu_sk.img.gz')
            obsid_array = np.append(obsid_array, obsid.split('/')[-1] + '_' + str(j+1))
            ra_array = np.append(ra_array, hdu[j+1].header['RA_PNT'])
            dec_array = np.append(dec_array, hdu[j+1].header['DEC_PNT'])
            roll_angle_array = np.append(roll_angle_array, hdu[j+1].header['PA_PNT'])




df = pd.DataFrame(data={'Time (MJD)': time_array, 'Time (ISOT)': isot_array, 'RA (deg)': ra_array, 
                          'Dec (deg)': dec_array, 'Roll Angle (deg)': roll_angle_array, 'exptime (s)': exposure_time_array,
                        'filename': filename_array,'obsid': obsid_array})
df['Type'] = 'UV grism'





time_array = np.array([])
isot_array = np.array([])
exposure_time_array = np.array([])
filename_array = np.array([],dtype=str)
obsid_array = np.array([],dtype=str)
ra_array = np.array([])
dec_array = np.array([])
roll_angle_array = np.array([])


for i, obsid in enumerate(obsids):

    print(i, obsid.split('/')[-1])

    filename = obsid + subdirectory_path + 'sw' + obsid.split('/')[-1] + 'um2_sk.img.gz'
    if not os.path.exists(filename):
        filename = filename.replace('.gz','')
    

    if os.path.exists(filename):
        hdu = pyfits.open(filename)

        n_hdu_extensions = len(hdu)

        for j in range(n_hdu_extensions-1):


            header = hdu[j+1].header

            exptime = hdu[j+1].header['TELAPSE']
            start_time = Time(hdu[j+1].header["DATE-OBS"], format = "isot", scale = "utc")
            end_time = Time(hdu[j+1].header["DATE-END"], format = "isot", scale = "utc")
            t_mid_jd = np.mean([start_time.jd, end_time.jd])
            mean_time = Time(t_mid_jd, format = "jd")
            RA = hdu[j+1].header['RA_PNT']
            Dec = hdu[j+1].header['DEC_PNT']
            roll_angle = hdu[j+1].header['PA_PNT']


            time_array = np.append(time_array, mean_time.jd)
            isot_array = np.append(isot_array, mean_time.isot)
            exposure_time_array = np.append(exposure_time_array, exptime)
            filename_array = np.append(filename_array, 'sw' + obsid.split('/')[-1] + 'um2_sk.img.gz')
            obsid_array = np.append(obsid_array, obsid.split('/')[-1] + '_' + str(j+1))
            ra_array = np.append(ra_array, hdu[j+1].header['RA_PNT'])
            dec_array = np.append(dec_array, hdu[j+1].header['DEC_PNT'])
            roll_angle_array = np.append(roll_angle_array, hdu[j+1].header['PA_PNT'])




df2 = pd.DataFrame(data={'Time (MJD)': time_array, 'Time (ISOT)': isot_array, 'RA (deg)': ra_array, 
                          'Dec (deg)': dec_array, 'Roll Angle (deg)': roll_angle_array, 'exptime (s)': exposure_time_array,
                        'filename': filename_array,'obsid': obsid_array})
df2['Type'] = 'UVM2'


df_both = pd.concat([df, df2])
df_both=df_both.sort_values('Time (MJD)')
df_both.reset_index(inplace=True)
df_both.to_csv(save_path + 'obs_info.csv')

